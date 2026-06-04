# Object Detection and Tracking — YOLOv8 + SORT

Real-time multi-object detection and tracking for webcam or video files.
**YOLOv8** detects objects in each frame; a custom **SORT** tracker (Kalman Filter + Hungarian algorithm) links those detections across frames, giving every object a stable ID.

---

 

## How it works — the full pipeline

```
Video frame
    │
    ▼
┌─────────────────────────────┐
│  YOLOv8  (per-frame)        │  ← no memory, just detects objects in this frame
│  outputs: [x1,y1,x2,y2,     │
│            confidence, class]│
└─────────────┬───────────────┘
              │  list of detections
              ▼
┌─────────────────────────────┐
│  SORT Tracker               │
│                             │
│  1. Kalman predict          │  ← where is each existing track NOW?
│     each active track       │
│                             │
│  2. IoU matrix              │  ← how much do detections overlap tracks?
│     (+ class constraint     │     cross-class matches forbidden
│      + velocity constraint) │     opposite-direction matches penalised
│                             │
│  3. Hungarian algorithm     │  ← optimal 1-to-1 assignment
│                             │
│  4. Update matched tracks   │  ← Kalman filter refines position estimate
│     Create new tracks       │
│     Delete lost tracks      │
└─────────────┬───────────────┘
              │  [x1,y1,x2,y2, track_id, class_id]
              ▼
┌─────────────────────────────┐
│  Visualiser                 │  ← draw boxes, trails, HUD
└─────────────────────────────┘
```

### Why SORT needs a Kalman Filter

YOLOv8 runs independently on each frame — it has no memory. Between frames, objects move, get partially occluded, or briefly disappear. The Kalman Filter maintains a **state estimate** for every active track:

```
State vector:  [cx, cy, s, r, vx, vy, vs]
                ──────────────  ─────────────
                  position       velocity
```

- `cx, cy` — centre of bounding box
- `s` — scale (area of box)
- `r` — aspect ratio (held constant)
- `vx, vy, vs` — rate of change of each position variable

Each frame: **predict** where the object moved → **update** with the new detection. Even if the detector misses an object for a few frames, the Kalman prediction keeps the track alive.

### Why Hungarian algorithm

After predicting positions, we have N detections and M existing tracks. We need to decide which detection belongs to which track. This is the **assignment problem** — solved optimally in O(N³) by the Hungarian algorithm using IoU as the cost.

### What this repo adds on top of vanilla SORT

| Enhancement | Where | Effect |
|---|---|---|
| **Same-class constraint** | `sort.py` | A car can never be matched to a person — IoU forced to 0 across classes |
| **Velocity-direction check** | `sort.py` | If a detection is in the *opposite* direction to a track's velocity, its IoU score is penalised 75% |
| **Fixed assignment condition** | `sort.py` | Original code had a logic error that skipped Hungarian assignment most frames — fixed |
| **Per-ID isolated RNG** | `tracker_app.py` | Each track ID gets a deterministic colour without mutating the global random state |
| **Motion trail** | `tracker_app.py` | Last N centre positions drawn as a fading polyline |
| **Per-class count HUD** | `tracker_app.py` | Live count overlay per detected class |
| **Benchmark mode** | `tracker_app.py` | Prints FPS stats, max simultaneous tracks, new track IDs created |
| **Edge-clamped labels** | `tracker_app.py` | Labels never overflow the frame edge |
| **Tracker reset between runs** | `sort.py` | IDs always start at 1 — no bleed-over between videos |

---

## Setup

### Requirements

- Python 3.8+
- (Optional but recommended) NVIDIA GPU with CUDA for real-time performance

```bash
pip install -r requirements.txt
```

`requirements.txt` installs: `numpy`, `scipy`, `opencv-python`, `ultralytics` (YOLOv8), `torch`.

---

## Usage

### Webcam (default)
```bash
python tracker_app.py
```

### Video file
```bash
python tracker_app.py --source path/to/video.mp4
```

### Track specific classes only
```bash
# Person only (COCO class 0)
python tracker_app.py --classes 0

# Person + car + bicycle
python tracker_app.py --classes 0 2 1
```

### Use a larger / more accurate model
```bash
python tracker_app.py --model yolov8s.pt --conf 0.4
```

### Benchmark mode — prints stats at the end
```bash
python tracker_app.py --source video.mp4 --benchmark
```

### All options
```
--source     Webcam index or video path          (default: 0)
--model      YOLO model name or path             (default: yolov8n.pt)
--conf       Detection confidence threshold      (default: 0.30)
--iou        SORT IoU threshold for matching     (default: 0.30)
--max-age    Frames before a lost track dies     (default: 30)
--min-hits   Detections before track is shown    (default: 2)
--classes    COCO class IDs to track             (default: all)
--trail      Trail length in frames, 0=off       (default: 20)
--benchmark  Print FPS + tracking stats at end
--no-count   Hide the per-class count overlay
```

### Controls
Press **`Q`** in the output window to stop.

---

## Parameter guide

### `--conf` — detection confidence

| Value | Effect |
|---|---|
| `0.20` | More detections, more false positives, more ID switches |
| `0.30` | Good balance (default) |
| `0.50` | Fewer detections, fewer false positives, misses dim/distant objects |

### `--iou` — SORT matching threshold

| Value | Effect |
|---|---|
| `0.10` | Almost any overlap triggers a match → ghost tracks, ID inflation |
| `0.30` | Standard SORT value, works well for typical scenes (default) |
| `0.50` | Strict matching → tracks break during fast motion or occlusion |

### `--max-age` — track survival

How many consecutive frames a track can go unmatched before being deleted. Increase this if objects frequently pass behind obstacles.

### `--min-hits` — track confirmation

Number of consecutive matched detections before a track is displayed. Increase to suppress single-frame false detections.

---

## Known limitations

| Limitation | Cause | Workaround |
|---|---|---|
| ~1–2 FPS on CPU | YOLOv8 is GPU-optimised | Use a CUDA GPU or `yolov8n.pt` on CPU |
| Misclassification on overhead footage | COCO training data is mostly ground-level | Fine-tune on aerial data or use a top-down specific model |
| ID switch after long occlusion | SORT has no appearance model | Switch to BoT-SORT or ByteTrack (drop-in via `ultralytics`) |
| All COCO classes detected by default | YOLO sees 80 classes | Use `--classes` to filter to what you need |

---

## Project structure

```
object-detection-and-tracking/
├── tracker_app.py   — main entry point: video loop, drawing, CLI
├── sort.py          — SORT tracker: Kalman filter, Hungarian assignment
├── requirements.txt — pinned dependencies
└── README.md
```

### `sort.py` — key classes

| Class / function | Role |
|---|---|
| `KalmanFilter` | 7-state constant-velocity filter for one bounding box |
| `KalmanBoxTracker` | Wraps KalmanFilter, holds ID, class, hit history, trail |
| `associate_detections_to_trackers()` | Builds IoU matrix, applies class + velocity constraints, runs Hungarian |
| `Sort` | Orchestrates predict → associate → update → prune each frame |

### `tracker_app.py` — key functions

| Function | Role |
|---|---|
| `open_source()` | Opens webcam or video file, downloads sample if missing |
| `get_color()` | Deterministic per-ID colour from an isolated RNG |
| `draw_box()` | Corner-accent box + edge-safe label tag |
| `draw_trail()` | Fading polyline of past centres |
| `draw_hud()` | FPS + per-class count overlay |
| `BenchmarkStats` | Accumulates FPS, max tracks, new ID count |
| `main()` | CLI → model load → frame loop → cleanup |

---

## Possible next improvements

- **ByteTrack / BoT-SORT** — better occlusion handling via a second association pass using low-confidence detections. Available in Ultralytics: `model.track(source, tracker="bytetrack.yaml")`
- **Re-identification** — add an appearance embedding (e.g. OSNet) so IDs survive long occlusions
- **GPU export** — export YOLOv8 to TensorRT for 10× faster inference: `yolo export model=yolov8n.pt format=engine`
- **Zone counting** — define polygon zones and count objects entering/exiting
- **CSV/JSON logging** — write track IDs, classes, positions, and timestamps to a file per run

---

## License

MIT
