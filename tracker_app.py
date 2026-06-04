import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Tracking with YOLO & SORT")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to video file or webcam index (e.g. 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or name (e.g. yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO detections",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter detections by class index (e.g. 0 for person)",
    )
    return parser.parse_args()


def draw_elegant_box(img, x1, y1, x2, y2, label, color, thickness=2, line_length=15):
    """
    Draws a premium bounding box with modern corner accents and a clean tag.
    """
    # Cast coordinate variables to integer
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw semi-transparent background fill for the box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Draw main thin bounding box outline
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    # Draw thick corners for high-end look
    # Top Left
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
    # Top Right
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
    # Bottom Left
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)

    # Draw text tag above the box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Ensure label background doesn't go out of bounds
    tag_y1 = max(0, y1 - text_h - 8)
    tag_y2 = y1

    # Draw solid tag background
    cv2.rectangle(img, (x1, tag_y1), (x1 + text_w + 10, tag_y2), color, -1)
    # Draw text inside tag
    cv2.putText(
        img,
        label,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    # Determine input source
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"[INFO] Initializing webcam/video source: {source}...")
    
    # Use cv2.CAP_DSHOW backend on Windows for reliable webcam access
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    
    # Fallback to sample.mp4 if camera index fails to open
    if not cap.isOpened() and isinstance(source, int):
        print(f"[WARNING] Could not open webcam at index {source} with DirectShow. Falling back to 'sample.mp4'...")
        source = "sample.mp4"
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print(f"[INFO] Loading YOLO model: {args.model}...")
    model = YOLO(args.model)

    # Get class names dictionary from YOLO
    class_names = model.names

    # Initialize SORT tracker
    print("[INFO] Initializing SORT tracker...")
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

    # Define color map for different track IDs for visualization
    # Using HSL/harmonious colors converted to BGR
    def get_color(track_id):
        np.random.seed(int(track_id))
        color = np.random.randint(50, 255, size=3).tolist()
        return tuple(color)

    prev_time = 0
    print("[INFO] Start processing. Press 'q' on the output window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended or failed to read frame.")
            break

        # Run YOLO detection
        results = model(frame, verbose=False, conf=args.conf, classes=args.classes)[0]

        # Extract detections: [x1, y1, x2, y2, confidence, class_id]
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = box.cls[0].cpu().item()
            dets.append([x1, y1, x2, y2, conf, cls])

        dets = np.array(dets)
        if len(dets) == 0:
            dets = np.empty((0, 6))

        # Update SORT tracker
        # sort.update accepts [[x1, y1, x2, y2, score, class_id], ...]
        track_bbs_ids = tracker.update(dets)

        # Draw bounding boxes and tracks
        for track in track_bbs_ids:
            x1, y1, x2, y2, track_id, class_id = track
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Map class ID to class name
            label_name = class_names.get(class_id, "Unknown")
            label = f"ID {track_id} | {label_name}"
            
            color = get_color(track_id)
            draw_elegant_box(frame, x1, y1, x2, y2, label, color)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
        prev_time = curr_time

        # Draw FPS overlay
        cv2.rectangle(frame, (10, 10), (160, 45), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 127),
            2,
            cv2.LINE_AA,
        )

        # Display window
        cv2.imshow("Real-time Object Detection and Tracking (SORT)", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
