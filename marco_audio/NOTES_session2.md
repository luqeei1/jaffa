# Session 2 (Wed 28 Oct 2026) - what's new and how to run it

Copy these folders into `C:\marco_audio` (they only add files; nothing existing is replaced).

## tools/
- `doa_phat.py` - GCC-PHAT, least-squares and robust direction fits, SRP-PHAT, and the
  SoundEvent fields (confidence, azimuth_std_rad). Same geometry and angle convention as
  `mic_array.py` (0 deg = USB edge, anticlockwise from above).
- `test_doa_phat.py` - self-tests with ideal delays. Run: `python tools\test_doa_phat.py`
  (should end with `ALL PASSED`).

## sim/  (needs `pip install pyroomacoustics`)
- `benchmark.py` - simulated room; compares plain cross-correlation (re-implementation of the
  current method), GCC-PHAT, GCC-PHAT robust and SRP-PHAT. Run: `python sim\benchmark.py`
  (a few minutes; `--quick` for a short check). Writes `sim\out\events.csv` and `summary.csv`.
- `tune.py` - analysis window length and PHAT strength (beta). Writes `sim\out\tune.txt`.
- `distance.py` - error at 1, 2 and 4 m in a bigger room. Writes `sim\out\distance.txt`.
- `calibration.py` - does confidence / azimuth_std match the real errors?
- `explain_figure.py` - the three-panel "why PHAT works" figure (`sim\out\explain_phat.png`).
- `plot_benchmark.py` - error vs reverberation figure (`sim\out\sim_error_vs_rt60.png`).
- `plot_style.py` - shared figure style and the colour for each method.

## Using it on a recording (until it is wired into analyse_directions.py)
```python
import numpy as np, doa_phat as dp
# x: (n_samples, 6) int16 recording; raw mics are channels 2-5
seg = x[onset - 80 : onset + 480, 2:6]          # 5 ms before to 30 ms after the onset
r = dp.localise(seg)
print(np.degrees(r["gcc_phat"]["azimuth_rad"]), r["gcc_phat"]["confidence"])
print(np.degrees(r["srp_phat"]["azimuth_rad"]), r["srp_phat"]["confidence"])
```
