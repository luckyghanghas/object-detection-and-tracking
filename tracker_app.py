"""
Real-time Object Detection and Tracking
========================================
YOLOv8  +  Custom SORT (Kalman Filter + Hungarian Assignment)

Usage
-----
  python tracker_app.py                          # webcam
  python tracker_app.py --source video.mp4       # video file
  python tracker_app.py --source video.mp4 --benchmark
  python tracker_app.py --classes 0 2            # person + car only
  python tracker_app.py --model yolov8s.pt --conf 0.4
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sort import Sort


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time Object Detection & Tracking — YOLOv8 + SORT"
    )
    p.add_argument("--source",    type=str,   default="0",
                   help="Webcam index (0) or path to video file")
    p.add_argument("--model",     type=str,   default="yolov8n.pt",
                   help="YOLO model name/path  (yolov8n.pt, yolov8s.pt, ...)")
    p.add_argument("--conf",      type=float, default=0.30,
                   help="YOLO confidence threshold  (0-1)")
    p.add_argument("--iou",       type=float, default=0.15,
                   help="SORT IoU threshold for track matching  (0-1)")
    p.add_argument("--max-age",   type=int,   default=50,
                   help="Frames a track can go unmatched before deletion")
    p.add_argument("--min-hits",  type=int,   default=1,
                   help="Detections required before a track is displayed")
    p.add_argument("--classes",   type=int,   nargs="+", default=None,
                   help="COCO class indices to track (default: all)")
    p.add_argument("--trail",     type=int,   default=20,
                   help="Trail length in frames  (0 = off)")
    p.add_argument("--benchmark", action="store_true",
                   help="Print summary stats at the end")
    p.add_argument("--no-count",  action="store_true",
                   help="Hide the per-class object count overlay")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLOR_CACHE: "dict[int, tuple]" = {}

def get_color(track_id: int) -> tuple:
    if track_id not in _COLOR_CACHE:
        rng = np.random.default_rng(int(track_id))
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        _COLOR_CACHE[track_id] = color
    return _COLOR_CACHE[track_id]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box(img, x1, y1, x2, y2, label, color, line_len=14, thickness=2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    H, W = img.shape[:2]

    x1, x2 = max(0, x1), min(W - 1, x2)
    y1, y2 = max(0, y1), min(H - 1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    for sx, ex in [(x1, x1 + line_len), (x2, x2 - line_len)]:
        for sy, ey in [(y1, y1 + line_len), (y2, y2 - line_len)]:
            cv2.line(img, (sx, sy), (ex, sy), color, thickness)
            cv2.line(img, (sx, sy), (sx, ey), color, thickness)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    font_thick = 1
    (tw, th), bl = cv2.getTextSize(label, font, font_scale, font_thick)
    pad = 5

    tag_x1 = max(0, min(x1, W - tw - pad * 2))
    tag_x2 = tag_x1 + tw + pad * 2
    tag_y2 = max(th + bl + pad, y1)
    tag_y1 = tag_y2 - th - bl - pad

    cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.putText(img, label,
                (tag_x1 + pad, tag_y2 - bl - 2),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def draw_trail(img, trail, color, length):
    pts = trail[-length:]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        t_color = tuple(int(c * alpha) for c in color)
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]),     int(pts[i][1]))
        cv2.line(img, p1, p2, t_color, 2, cv2.LINE_AA)


def draw_hud(img, fps, class_counts, class_names, show_counts):
    if class_names is None:
        class_names = {}
    lines = [f"FPS: {fps:.1f}"]
    if show_counts and class_counts:
        for cls_id, cnt in sorted(class_counts.items()):
            name = class_names.get(cls_id, f"cls{cls_id}")
            lines.append(f"{name}: {cnt}")

    pad, lh = 8, 20
    box_h = pad * 2 + lh * len(lines)
    box_w = 160

    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for i, line in enumerate(lines):
        color = (0, 255, 127) if i == 0 else (220, 220, 220)
        cv2.putText(img, line,
                    (16, 8 + pad + (i + 1) * lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video source helper
# ---------------------------------------------------------------------------

SAMPLE_URL = (
    "https://raw.githubusercontent.com/intel-iot-devkit/"
    "sample-videos/master/person-bicycle-car-detection.mp4"
)

def open_source(source_str: str):
    source = int(source_str) if source_str.isdigit() else source_str

    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"[INFO] Opened webcam index {source}")
            return cap
        print(f"[WARN] Could not open webcam {source}. Falling back to sample video.")
        source = "sample.mp4"

    if isinstance(source, str) and not os.path.exists(source):
        print(f"[INFO] '{source}' not found — downloading sample video ...")
        try:
            urllib.request.urlretrieve(SAMPLE_URL, source)
            print("[INFO] Download complete.")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {source}")
        return None

    print(f"[INFO] Opened: {source}")
    return cap


# ---------------------------------------------------------------------------
# Benchmark stats
# ---------------------------------------------------------------------------

class BenchmarkStats:
    def __init__(self):
        self.fps_samples      = []
        self.max_simultaneous = 0
        self.id_switches      = 0
        self._prev_ids: set   = set()
        self._first_frame     = True

    def update(self, fps, track_ids: set):
        self.fps_samples.append(fps)
        self.max_simultaneous = max(self.max_simultaneous, len(track_ids))
        if not self._first_frame:
            self.id_switches += len(track_ids - self._prev_ids)
        self._first_frame = False
        self._prev_ids    = track_ids

    def report(self):
        if not self.fps_samples:
            return
        avg = np.mean(self.fps_samples)
        mn  = np.min(self.fps_samples)
        mx  = np.max(self.fps_samples)
        print("\n" + "=" * 45)
        print("  BENCHMARK RESULTS")
        print("=" * 45)
        print(f"  Average FPS        : {avg:.1f}")
        print(f"  Min / Max FPS      : {mn:.1f} / {mx:.1f}")
        print(f"  Max simultaneous   : {self.max_simultaneous} objects")
        print(f"  New track IDs seen : {self.id_switches}")
        print("=" * 45 + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cap = open_source(args.source)
    if cap is None:
        return

    print(f"[INFO] Loading model: {args.model} ...")
    model = YOLO(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running inference on: {device.upper()}")

    class_names: dict = model.names or {}

    tracker = Sort(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou,
    )
    tracker.reset()

    stats       = BenchmarkStats() if args.benchmark else None
    prev_time   = time.time()
    out_writer  = None
    show_counts = not args.no_count

    print("[INFO] Processing ... press Q in the window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Detection ----
        results = model(
            frame,
            verbose=False,
            conf=args.conf,
            classes=args.classes,
            device=device,
            imgsz=640,
        )[0]

        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            cls  = float(box.cls[0].cpu())
            dets.append([x1, y1, x2, y2, conf, cls])

        dets = np.array(dets) if dets else np.empty((0, 6))

        # ---- Tracking ----
        tracks = tracker.update(dets)

        # ---- Draw ----
        class_counts: "dict[int, int]" = {}
        current_ids:  "set[int]"       = set()

        for trk in tracks:
            x1, y1, x2, y2, tid, cid = trk
            tid = int(tid)
            cid = int(cid)
            current_ids.add(tid)
            class_counts[cid] = class_counts.get(cid, 0) + 1

            color = get_color(tid)
            label = f"ID {tid} | {class_names.get(cid, '?')}"
            draw_box(frame, x1, y1, x2, y2, label, color)

            if args.trail > 0:
                trk_obj = next(
                    (t for t in tracker.trackers if t.id + 1 == tid), None
                )
                if trk_obj and len(trk_obj.trail) > 1:
                    draw_trail(frame, trk_obj.trail, color, args.trail)

        # ---- HUD ----
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now
        draw_hud(frame, fps, class_counts, class_names, show_counts)

        if stats:
            stats.update(fps, current_ids)

        # ---- Display or write ----
        try:
            cv2.imshow("Object Detection & Tracking  (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error:
            if out_writer is None:
                h, w    = frame.shape[:2]
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                fps_out = src_fps if src_fps > 0 else 25.0

                # Try H.264 first (plays on Windows/Mac/Linux without extra codecs)
                fourcc     = cv2.VideoWriter_fourcc(*"avc1")
                out_writer = cv2.VideoWriter("output.mp4", fourcc, fps_out, (w, h))
                if not out_writer.isOpened():
                    out_writer.release()
                    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
                    out_writer = cv2.VideoWriter("output.mp4", fourcc, fps_out, (w, h))
                    print("[INFO] Headless mode — writing output.mp4 (mp4v codec)")
                else:
                    print("[INFO] Headless mode — writing output.mp4 (H.264 codec)")
            out_writer.write(frame)

    # ---- Cleanup ----
    if out_writer:
        out_writer.release()
        print("[INFO] Saved output.mp4")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    if stats:
        stats.report()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
