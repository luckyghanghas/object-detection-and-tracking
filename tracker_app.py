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
        default=0.25,
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
    Handles boundary clamping to prevent tags from clipping off-screen.
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

    # Draw text tag above the box (or inside if near the top edge)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Get frame dimensions to clamp horizontally
    img_h, img_w = img.shape[:2]
    text_x1 = max(0, min(x1, img_w - text_w - 10))

    # Adjust vertical tag placement to prevent clipping at the top
    if y1 < text_h + 15:
        tag_y1 = y1
        tag_y2 = y1 + text_h + 8
        text_y = y1 + text_h + 3
    else:
        tag_y1 = y1 - text_h - 8
        tag_y2 = y1
        text_y = y1 - 5

    # Draw solid tag background
    cv2.rectangle(img, (text_x1, tag_y1), (text_x1 + text_w + 10, tag_y2), color, -1)
    # Draw text inside tag
    cv2.putText(
        img,
        label,
        (text_x1 + 5, text_y),
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
        
        # Automatically download sample.mp4 if it does not exist locally
        import os
        if not os.path.exists(source):
            print("[INFO] 'sample.mp4' not found. Downloading sample video...")
            url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, source)
                print("[INFO] Download completed successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to download sample video: {e}")
                
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print(f"[INFO] Loading YOLO model: {args.model}...")
    model = YOLO(args.model)

    # Get class names dictionary from YOLO
    class_names = model.names

    # Initialize SORT tracker with corrected parameters
    print("[INFO] Initializing SORT tracker...")
    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

    # Define color map using a cache to avoid slow random-seed recomputations
    COLOR_CACHE = {}
    def get_color(track_id):
        if track_id not in COLOR_CACHE:
            np.random.seed(int(track_id))
            color = np.random.randint(50, 255, size=3).tolist()
            COLOR_CACHE[track_id] = tuple(color)
        return COLOR_CACHE[track_id]

    prev_time = 0
    out_writer = None
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

        # Display window or fallback to saving video in headless environments
        try:
            cv2.imshow("Real-time Object Detection and Tracking (SORT)", frame)
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        except cv2.error:
            # We are in a headless environment (like Codespaces)
            if out_writer is None:
                height, width, _ = frame.shape
                out_writer = cv2.VideoWriter(
                    "output.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    25.0,  # default FPS for output
                    (width, height)
                )
                print("[INFO] Headless environment detected. Saving output to 'output.mp4'...")
            
            out_writer.write(frame)
            # Check if we should quit (non-blocking in headless mode)
            # In headless mode we process the whole video and then exit
            pass

    if out_writer is not None:
        out_writer.release()
        print("[INFO] Finished writing output to 'output.mp4'. You can now download and view it!")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print("[INFO] Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
