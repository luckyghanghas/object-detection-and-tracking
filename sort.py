import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    """
    Solves the linear sum assignment problem using SciPy's Hungarian algorithm.
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    bb_test: array of shape [N, 4], format [x1, y1, x2, y2]
    bb_gt: array of shape [M, 4], format [x1, y1, x2, y2]
    Returns an array of shape [N, M] with IoU values.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


class KalmanFilter:
    """
    A simple Kalman Filter implementation in NumPy.
    State representation: [x, y, s, r, vx, vy, vs]^T
    where (x, y) is center of bounding box, s is scale (area), r is aspect ratio,
    and vx, vy, vs are respective velocities.
    """
    def __init__(self, x_init):
        # State: 7x1
        self.x = np.zeros((7, 1))
        self.x[:4] = x_init.reshape(4, 1)

        # State transition matrix F
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Covariance matrix P
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.  # High uncertainty in initial velocities
        self.P *= 10.

        # Process noise covariance matrix Q
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise covariance matrix R
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        # Measurement update step
        y = z.reshape(4, 1) - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)


class KalmanBoxTracker:
    """
    Represents the state of individual tracked objects observed as bounding boxes.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        """
        self.kf = KalmanFilter(self.convert_bbox_to_z(bbox))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the center of the box, s is the scale/area and r is
        the aspect ratio.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = float(w) / float(h) if h > 0 else 0
        return np.array([x, y, s, r])

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a state vector in the form [x,y,s,r,...] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top-left and x2,y2 is the bottom-right.
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w > 0 else 0
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, det_classes, trk_classes, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes).
    Only allows matches between the same class.
    """
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    # Enforce same-class matching by zeroing out IoU for mismatched classes
    for d in range(len(detections)):
        for t in range(len(trackers)):
            if det_classes[d] != trk_classes[t]:
                iou_matrix[d, t] = 0.0

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold)
        if a.all(axis=0).any() or a.all(axis=1).any():
            matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty((0, 2), dtype=int)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the tracks as a numpy array where each row contains [x1,y1,x2,y2,id,class_id].
        """
        self.frame_count += 1
        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.delete(trks, to_del, axis=0)
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Separate detections into bounding boxes and confidence/class info if present
        if dets.shape[1] >= 5:
            dets_boxes = dets[:, :4]
        else:
            dets_boxes = dets

        # Extract classes for same-class matching constraints
        det_classes = dets[:, 5] if dets.shape[1] > 5 else np.array([-1] * len(dets))
        trk_classes = np.array([tracker.class_id for tracker in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_boxes, trks[:, :4], det_classes, trk_classes, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for m in matched:
            # Update matching tracker.
            # If original dets has class ID at index 5, store it in KalmanBoxTracker or use it
            self.trackers[m[1]].update(dets[m[0], :4])
            # If the original det had class info, we attach it here
            self.trackers[m[1]].class_id = dets[m[0], 5] if dets.shape[1] > 5 else -1

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.class_id = dets[i, 5] if dets.shape[1] > 5 else -1
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Format: x1, y1, x2, y2, id, class_id
                ret.append(np.concatenate((d, [trk.id + 1, trk.class_id])).reshape(1, 6))
            i -= 1
            # Remove dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret, axis=0)
        return np.empty((0, 6))
