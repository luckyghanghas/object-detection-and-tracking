"""
SORT: Simple Online and Realtime Tracking
Custom NumPy/SciPy implementation with class-aware matching.

Key parameters tuned for stable IDs:
  - iou_threshold = 0.15  (low → tolerates partial occlusion & uncertain Kalman predictions)
  - max_age       = 50    (long → track survives brief disappearances instead of respawning)
  - min_hits      = 1     (label appears immediately on first detection)
  - Class blocking deferred until a tracker has >= 5 hits (class label unreliable early on)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    bb_gt   = np.expand_dims(bb_gt,   0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w     = np.maximum(0., xx2 - xx1)
    h     = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = ((bb_test[..., 2] - bb_test[..., 0]) *
                 (bb_test[..., 3] - bb_test[..., 1]))
    area_gt   = ((bb_gt[...,   2] - bb_gt[...,   0]) *
                 (bb_gt[...,   3] - bb_gt[...,   1]))

    return inter / (area_test + area_gt - inter + 1e-9)


class KalmanFilter:
    def __init__(self, x_init):
        self.x = np.zeros((7, 1))
        self.x[:4] = x_init.reshape(4, 1)

        self.F = np.eye(7)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1

        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.
        self.P *= 10.

        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z.reshape(4, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class KalmanBoxTracker:
    _count = 0

    def __init__(self, bbox, class_id=-1):
        self.kf               = KalmanFilter(self._bbox_to_z(bbox))
        self.id               = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.class_id          = int(class_id)
        self.time_since_update = 0
        self.hits              = 0
        self.hit_streak        = 0
        self.age               = 0
        self.history           = []

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.trail = [(cx, cy)]

    def update(self, bbox, class_id=-1):
        self.time_since_update = 0
        self.history           = []
        self.hits             += 1
        self.hit_streak       += 1
        self.class_id          = int(class_id)
        self.kf.update(self._bbox_to_z(bbox))

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.trail.append((cx, cy))
        if len(self.trail) > 40:
            self.trail.pop(0)

    def predict(self):
        if (self.kf.x[6, 0] + self.kf.x[2, 0]) <= 0:
            self.kf.x[6, 0] = 0.
        x_pred = self.kf.predict()
        if x_pred[2, 0] <= 0:
            x_pred[2, 0]    = 1e-3
            self.kf.x[2, 0] = 1e-3
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self._x_to_bbox(self.kf.x)

    @staticmethod
    def _bbox_to_z(bbox):
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s  = w * h
        r  = float(w) / float(h) if h > 0 else 1.0
        return np.array([cx, cy, s, r])

    @staticmethod
    def _x_to_bbox(x):
        x = x.flatten()          # (7,1) → (7,) so x[i] is always a scalar
        s = float(x[2])
        r = float(x[3])
        r = max(r, 1e-6)
        w = np.sqrt(max(s * r, 0.))
        h = (s / w) if w > 0 else 0.
        return np.array([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.,
        ]).reshape((1, 4))


def associate_detections_to_trackers(detections, trackers,
                                     det_classes, trk_classes,
                                     active_trackers,
                                     iou_threshold=0.15):
    n_det = len(detections)
    n_trk = len(trackers)

    if n_det == 0:
        return (np.empty((0, 2), dtype=int),
                np.empty((0,),   dtype=int),
                np.arange(n_trk, dtype=int))
    if n_trk == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(n_det, dtype=int),
                np.empty((0,),   dtype=int))

    iou_matrix = iou_batch(detections, trackers)

    for d in range(n_det):
        dc = int(det_classes[d])
        for t in range(n_trk):
            tc  = int(trk_classes[t])
            trk = active_trackers[t]
            # Enforce class only after tracker has >= 5 reliable hits
            if dc >= 0 and tc >= 0 and dc != tc and trk.hits >= 5:
                iou_matrix[d, t] = 0.0

    matched_indices = (linear_assignment(-iou_matrix)
                       if iou_matrix.size > 0
                       else np.empty((0, 2), dtype=int))

    matched_det_set = (set(matched_indices[:, 0].tolist())
                       if len(matched_indices) else set())
    matched_trk_set = (set(matched_indices[:, 1].tolist())
                       if len(matched_indices) else set())

    unmatched_dets = [d for d in range(n_det) if d not in matched_det_set]
    unmatched_trks = [t for t in range(n_trk) if t not in matched_trk_set]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(int(m[0]))
            unmatched_trks.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    matches = (np.concatenate(matches, axis=0)
               if matches else np.empty((0, 2), dtype=int))

    return (matches,
            np.array(unmatched_dets, dtype=int),
            np.array(unmatched_trks, dtype=int))


class Sort:
    def __init__(self, max_age=50, min_hits=1, iou_threshold=0.15):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def reset(self):
        self.trackers    = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0

    def update(self, dets=np.empty((0, 6))):
        self.frame_count += 1

        trks   = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk_obj in enumerate(self.trackers):
            pos = trk_obj.predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)

        dets_boxes  = dets[:, :4] if len(dets) else np.empty((0, 4))
        det_classes = (dets[:, 5] if len(dets) and dets.shape[1] > 5
                       else np.full(len(dets), -1))
        trk_classes = np.array([tr.class_id for tr in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_boxes, trks, det_classes, trk_classes,
            self.trackers, self.iou_threshold
        )

        for m in matched:
            cls = int(dets[m[0], 5]) if dets.shape[1] > 5 else -1
            self.trackers[m[1]].update(dets[m[0], :4], cls)

        for i in unmatched_dets:
            cls = int(dets[i, 5]) if dets.shape[1] > 5 else -1
            self.trackers.append(KalmanBoxTracker(dets[i, :4], cls))

        ret = []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                d = trk.get_state()[0]
                ret.append(
                    np.concatenate((d, [trk.id + 1, trk.class_id])).reshape(1, 6)
                )

        self.trackers = [
            tr for tr in self.trackers
            if tr.time_since_update <= self.max_age
        ]

        return np.concatenate(ret, axis=0) if ret else np.empty((0, 6))
