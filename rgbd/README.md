# Marco person detector

RGB-D person detector for Marco, the hide-and-seek robot.

## Step 1: camera setup

- `sources/realsense.py`: `RealSenseSource` streams colour (BGR uint8) and depth aligned to colour (float32 metres, 0 = invalid) from a live D435i or a recording (`.db3`).
- `scripts/view.py`: shows colour and colourised depth side by side (0.3–5 m, black = no depth).

## Step 2: 2D person detection

- `core/detector.py`: `PersonDetector` wraps Ultralytics YOLO (`yolo26n.pt` by default) and returns person boxes `(x1, y1, x2, y2)` + confidence in original image pixels. Uses numpy + ultralytics only, so it can be reused in a ROS node.
- `scripts/detect_live.py`: live (or bag) colour stream with person boxes, detector time and loop FPS.

The first run downloads `yolo26n.pt` into this folder (needs internet once).

## Step 3: 3D position from box + depth

- `core/types.py`: `Intrinsics` and `PersonDetection` (no pyrealsense2 dependency).
- `core/localise.py`: `localise(boxes, depth_m, intrinsics, LocaliseConfig())`, pure numpy. For each box:
  depth from the central 50% x 60% of the box, kept if in [0.2, 8.0] m; with >= 50 valid pixels,
  Z = 30th percentile and the box centre is back-projected -> `POSITION`, else `BEARING` (direction only).
  Every detection gets a unit `bearing` and `bearing_std = atan((box_width/2)/fx)`.
  Camera optical frame: x right, y down, z forward, metres (front surface of the body).
- `scripts/detect_live.py`: green box = POSITION (label: conf, Z, X), orange = BEARING ("no depth").
  The thin inner rectangle is the depth region used. `d` toggles the depth view, `--print` prints detections.
- `tests/test_localise.py`: synthetic-data tests.

## Step 4: per-frame evidence

- `core/pipeline.py`: `Pipeline(detector).process(color_bgr, depth_m, intrinsics, stamp_ms, frame_number)` returns one
  `FrameEvidence` per frame, even when nobody is seen. No camera, GUI or ROS imports; the ROS node will call this.
- Covariance (POSITION only, 3x3, camera optical frame): `s_lat = a + b*r`, `s_rng = c + d*r^2`,
  `cov = s_lat^2 * I + (s_rng^2 - s_lat^2) * outer(d, d)`. The defaults are placeholders until measured. With them
  the along-ray sigma only exceeds the lateral one beyond 1 m.
- `truncated`: the box is within 2 px of an image edge.
- `FrameEvidence`: `stamp_ms`, `frame_number`, `source`, `detections`, `sensor_pose` (identity for now, from
  `identity_sensor_pose`, which ROS TF will replace), `h_fov` (radians), `max_range` (5 m). `max_range` is only
  reported; detections out to the 8 m depth limit are kept.
- `scripts/detect_live.py`: labels add `lat` and `ray` sigmas (m) and `EDGE`; the overlay shows the number of
  detections and h_fov. `--print` prints the whole `FrameEvidence` every frame, including empty ones.
- `tests/test_pipeline.py`: synthetic data with YOLO mocked.

## Setup (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Detection runs on the CPU (`DetectorConfig.device = "cpu"`).

## Run (from this folder)

```powershell
python -m scripts.view                          # live camera
python -m scripts.view --record recordings\test.db3   # live + record
python -m scripts.view --bag recordings\test.db3      # play back a recording

python -m scripts.detect_live                         # person detection, live
python -m scripts.detect_live --bag recordings\test.db3   # person detection on a recording
python -m scripts.detect_live --model yolo26s.pt --conf 0.5 --imgsz 480   # other settings
python -m scripts.detect_live --bag recordings\test.db3 --print           # also print 3D results

python -m pytest tests -v                             # run the tests
```

`detect_live` keys: `q` quit, `space` pause, `d` toggle depth view, `s` save `snapshots/det_<time>_color.png`, `_annotated.png`, `_depth.npy`.

The `recordings` folder must exist before you record (`mkdir recordings`). pyrealsense2 2.59 only records to `.db3` (a `.bag` path is refused), so recordings use that extension.

## Viewer controls

| Input | Action |
|---|---|
| Left-click | Print `(u, v)` and depth in metres (median of 5x5 patch, zeros ignored) |
| `space` | Pause / resume |
| `s` | Save `snapshots/snap_<time>_color.png` and `_depth.npy` (metres) |
| `q` | Quit |

## Using the source from other code

```python
from sources import RealSenseSource

with RealSenseSource(bag="recordings/test.db3") as src:
    print(src.intrinsics, src.depth_scale)
    while (frame := src.read()) is not None:
        color, depth_m, timestamp_ms, frame_number = frame
```
