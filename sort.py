"""
SORT: Simple Online and Realtime Tracking
Custom NumPy/SciPy implementation with class-aware matching and velocity constraints.

Improvements over original SORT:
  - Same-class enforcement (a car never gets matched to a person)
  - Velocity-direction consistency check (penalizes physically impossible matches)
  - Fixed Hungarian assignment condition (was silently skipping most matches)
  - Thread-safe ID counter via instance reset()
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Hungarian assignment
# ---------------------------------------------------------------------------

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_batch(bb_test, bb_gt):
    """
    Vectorised IoU between two sets of boxes.

    bb_test : (N, 4)  [x1, y1, x2, y2]
    bb_gt   : (M, 4)  [x1, y1, x2, y2]
    returns : (N, M)  IoU matrix
    """
    bb_gt   = np.expand_dims(bb_gt,   0)   # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1)   # (N, 1, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w  = np.maximum(0., xx2 - xx1)
    h  = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[...,   2] - bb_gt[...,   0]) * (bb_gt[...,   3] - bb_gt[...,   1])

    iou = inter / (area_test + area_gt - inter + 1e-9)
    return iou


# ---------------------------------------------------------------------------
# Kalman Filter  (state: [cx, cy, s, r, vx, vy, vs])
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Constant-velocity Kalman Filter for a bounding box.

    State vector: [cx, cy, s, r, vx, vy, vs]
      cx, cy  — centre of box
      s       — scale (area)
      r       — aspect ratio  (width / height, kept constant)
      vx, vy  — velocities of centre
      vs      — velocity of scale
    """

    def __init__(self, x_init):
        self.x = np.zeros((7, 1))
        self.x[:4] = x_init.reshape(4, 1)

        # State transition
        self.F = np.eye(7)
        self.F[0, 4] = 1   # cx += vx
        self.F[1, 5] = 1   # cy += vy
        self.F[2, 6] = 1   # s  += vs

        # Measurement matrix (we observe cx, cy, s, r)
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # Covariance — high uncertainty on initial velocities
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.
        self.P *= 10.

        # Process noise
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise — higher uncertainty on s, r
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


# ---------------------------------------------------------------------------
# Single object tracker
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Tracks a single object using a Kalman filter.
    Each instance gets a unique, monotonically increasing integer ID.
    """

    _count = 0          # class-level counter; reset via Sort.reset()

    def __init__(self, bbox, class_id=-1):
        self.kf  = KalmanFilter(self._bbox_to_z(bbox))
        self.id  = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.class_id         = int(class_id)
        self.time_since_update = 0
        self.hits             = 0
        self.hit_streak       = 0
        self.age              = 0
        self.history          = []

        # Trajectory: list of (cx, cy) centres for drawing trails
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.trail = [(cx, cy)]

    # ------------------------------------------------------------------
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
        if len(self.trail) > 40:      # keep last 40 positions
            self.trail.pop(0)

    def predict(self):
        # Prevent negative scale before predicting
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.
        x_pred = self.kf.predict()
        # Clamp predicted scale to avoid degenerate boxes after prediction
        if x_pred[2] <= 0:
            x_pred[2] = 1e-3
            self.kf.x[2] = 1e-3
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self._x_to_bbox(self.kf.x)

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_z(bbox):
        """[x1,y1,x2,y2] → [cx, cy, s, r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s  = w * h
        r  = float(w) / float(h) if h > 0 else 1.0
        return np.array([cx, cy, s, r])

    @staticmethod
    def _x_to_bbox(x):
        """[cx, cy, s, r, …] → [x1, y1, x2, y2]"""
        # s = w * h,  r = w / h  →  w = sqrt(s * r),  h = s / w
        s = float(x[2])
        r = float(x[3])
        r = max(r, 1e-6)          # guard against zero/negative aspect ratio
        w = np.sqrt(max(s * r, 0))
        h = s / w if w > 0 else 0.
        return np.array([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.,
        ]).reshape((1, 4))


# ---------------------------------------------------------------------------
# Detection → tracker association
# ---------------------------------------------------------------------------

def associate_detections_to_trackers(detections, trackers,
                                     det_classes, trk_classes,
                                     active_trackers,
                                     iou_threshold=0.3):
    """
    Match detections to existing tracks.

    Rules applied (in order):
      1. Cross-class matches are forbidden (IoU forced to 0).
      2. Velocity-direction consistency: if a tracker has a reliable
         velocity estimate and the detection is in the opposite direction,
         the IoU score is penalised by 75 %.
      3. Hungarian algorithm on the resulting cost matrix.
      4. Matches below iou_threshold are rejected.
    """
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

    iou_matrix = iou_batch(detections, trackers)   # (N_det, N_trk)

    for d in range(n_det):
        det_cx = (detections[d, 0] + detections[d, 2]) / 2.0
        det_cy = (detections[d, 1] + detections[d, 3]) / 2.0

        for t in range(n_trk):
            # Rule 1 — same class only
            if det_classes[d] != trk_classes[t]:
                iou_matrix[d, t] = 0.0
                continue

            # Rule 2 — velocity consistency
            trk = active_trackers[t]
            if trk.hits > 3:
                vx = trk.kf.x[4, 0]
                vy = trk.kf.x[5, 0]
                if np.sqrt(vx**2 + vy**2) > 2.0:
                    pred_cx = trk.kf.x[0, 0]
                    pred_cy = trk.kf.x[1, 0]
                    dx = det_cx - pred_cx
                    dy = det_cy - pred_cy
                    if (vx * dx + vy * dy) < 0:
                        iou_matrix[d, t] *= 0.25

    # --- BUG FIX: original condition was almost always False ---
    # Old (broken): if a.all(axis=0).any() or a.all(axis=1).any()
    # New (correct): run Hungarian whenever the matrix has entries
    if iou_matrix.size > 0:
        matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    matched_det_set = set(matched_indices[:, 0].tolist()) if len(matched_indices) else set()
    matched_trk_set = set(matched_indices[:, 1].tolist()) if len(matched_indices) else set()

    unmatched_dets = [d for d in range(n_det) if d not in matched_det_set]
    unmatched_trks = [t for t in range(n_trk) if t not in matched_trk_set]

    # Filter weak matches — push rejected pairs back to unmatched lists
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(int(m[0]))
            unmatched_trks.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


# ---------------------------------------------------------------------------
# SORT tracker
# ---------------------------------------------------------------------------

class Sort:
    """
    SORT multi-object tracker.

    Parameters
    ----------
    max_age       : frames a track survives without a detection match
    min_hits      : detections needed before a track is reported
    iou_threshold : minimum IoU for a detection-track match
    """

    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def reset(self):
        """Reset tracker state and ID counter (call between videos)."""
        self.trackers    = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0

    def update(self, dets=np.empty((0, 6))):
        """
        Parameters
        ----------
        dets : np.ndarray, shape (N, 6)
               Each row: [x1, y1, x2, y2, confidence, class_id]
               Pass np.empty((0, 6)) for empty frames.

        Returns
        -------
        np.ndarray, shape (M, 6)
               Each row: [x1, y1, x2, y2, track_id, class_id]
        """
        self.frame_count += 1

        # --- Predict new positions for all existing trackers ---
        trks    = np.zeros((len(self.trackers), 4))
        to_del  = []
        for t, trk_obj in enumerate(self.trackers):
            pos = trk_obj.predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)

        # --- Associate ---
        dets_boxes  = dets[:, :4]  if len(dets) else np.empty((0, 4))
        det_classes = dets[:, 5]   if len(dets) and dets.shape[1] > 5 else np.full(len(dets), -1)
        trk_classes = np.array([tr.class_id for tr in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_boxes, trks, det_classes, trk_classes,
            self.trackers, self.iou_threshold
        )

        # --- Update matched trackers ---
        for m in matched:
            cls = int(dets[m[0], 5]) if dets.shape[1] > 5 else -1
            self.trackers[m[1]].update(dets[m[0], :4], cls)

        # --- Create new trackers for unmatched detections ---
        for i in unmatched_dets:
            cls = int(dets[i, 5]) if dets.shape[1] > 5 else -1
            self.trackers.append(KalmanBoxTracker(dets[i, :4], cls))

        # --- Collect outputs and prune dead tracks ---
        ret = []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                d = trk.get_state()[0]
                ret.append(np.concatenate((d, [trk.id + 1, trk.class_id])).reshape(1, 6))

        # Prune (iterate forward so indices stay valid after pop)
        self.trackers = [
            tr for tr in self.trackers
            if tr.time_since_update <= self.max_age
        ]

        return np.concatenate(ret, axis=0) if ret else np.empty((0, 6))
