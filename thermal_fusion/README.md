# thermal_fusion: OAK Thermal + RealSense D415

Puts RealSense depth into the OAK Thermal image, so that every thermal pixel or thermal
detection gets a distance. It also compares person distances from RGB YOLO (`rgbd/`) and
thermal YOLO (`thermal/`).

This is different from `sensor_fusion/` (audio + RGB-D belief maps, `fusion` branch): this
folder only deals with the two cameras.

## Run (from this folder, with the `rgbd` venv)

```
python distance_compare.py --height 1.75   # RGB YOLO vs thermal YOLO distances, live
python align_view.py                       # check / fine-tune the calibration by eye
```

`distance_compare.py` keys: `[` / `]` ground truth distance, `l` log to `logs/*.csv`
(prints a summary when you stop), `s` screenshot, `q` quit.

## Calibration (`calibration/thermal_to_color.yaml`)

- **Thermal lens:** OAK factory calibration (fx 183.8, OpenCV rational distortion model),
  read from the camera with `read_oak_factory.py`. The datasheet values (fx 128) are badly
  wrong because the lens has strong barrel distortion.
- **Pose (thermal relative to the D415 colour camera):** tuned by eye in `align_view.py`
  for the mounting on 2026-11-18. **It is only valid for that mounting.** If either camera
  moves, re-tune it, or measure it with a checkerboard:
  `python make_board.py` (print A3 at 100%), heat the board (sunlight works; LED lamps do
  not), then `python calib_capture.py --mode pair` and `python calib_solve.py <session> --square 0.045`.

Result on 2026-11-18 (354 logged rows, 0.6-1.6 m): thermal YOLO + projected depth vs RGB-D
had a mean difference of -0.1 cm, std 1.7 cm, mean |difference| 1.0 cm. Below about 0.6 m,
the centre of the thermal box can miss the body.

## Notes

- The D415 is on USB 2 here: `sources.py` falls back to modes that keep the full field of
  view (640x480 colour crops it to 55 deg).
- Everything is in the colour camera's frame; depth is aligned to colour, as in `rgbd/`.
- Tests: `python -m pytest` (33 tests, synthetic data, no cameras needed).

| File | What |
|---|---|
| `geometry.py` | Back-projection, transforms, projection with lens distortion |
| `calibration.py` | Calibration file format |
| `sources.py` | RealSense colour / aligned depth and OAK temperature readers |
| `align_view.py` | Depth drawn over thermal, with sliders to tune the calibration |
| `distance_compare.py` | RGB YOLO vs thermal YOLO distances |
| `calib_capture.py`, `calib_solve.py`, `checkerboard.py`, `make_board.py` | Checkerboard calibration |
| `read_oak_factory.py` | Copies the OAK factory lens calibration into the YAML |
| `make_test_scene.py` | Ray-traced synthetic scene for the tests |
