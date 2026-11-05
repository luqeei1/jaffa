# SLAM back-end study (office, VLP-16 sim)

Keyframe pose-graph SLAM on top of each odometry front-end, run offline but causally on the
`sim_runs/vlp16` office data (86 m route, the core loop driven twice).

- `slam.py` is the back-end. It turns odometry into keyframes (every 1 m or 15°), finds loop candidates,
  verifies them with point-to-plane ICP, builds a GTSAM pose graph and solves it.
  - Loop candidates: `prox` (within 3 m of the current estimate and >10 m back along the path),
    `sc` (Scan Context descriptor), or both.
  - Optional odometry-consistency `gate`.
  - Solvers: `none`, `lm` (batch), `isam2` (incremental). Robust options: `cauchy`, `gnc`.
- `experiments.py` runs every front-end × strategy and the false-loop stress test (`--aliasing`), then
  writes `results.md/.csv`, `aliasing.md/.csv`, `table_slam.tex` and `fig_slam_{ape,traj,error}.pdf`.

Metrics:
- *final*: the trajectory after the last optimisation.
- *online*: the pose the robot had at the time (the latest keyframe's map→odom correction), which is
  what navigation uses.
- Map accuracy: keyframe clouds against the true world surfaces.
- Loop correctness: the loop's relative pose agrees with ground truth to 0.3 m and 3°.

```bash
# needs numpy<2 (the gtsam 4.2 wheel segfaults with numpy 2): an isolated venv
python3 -m venv ~/.venvs/hcr_slam && ~/.venvs/hcr_slam/bin/pip install "numpy<2" scipy pyyaml rosbags evo matplotlib gtsam==4.2
cd hcr_ws/sim_runs/slam_study
~/.venvs/hcr_slam/bin/python experiments.py ../vlp16 office_vlp16              # main matrix  (~10 min)
~/.venvs/hcr_slam/bin/python experiments.py ../vlp16 office_vlp16 --aliasing   # false loops  (~6 min)
```

`kiss_icp_drift` is synthetic: KISS-ICP odometry with 1 % scale error and 0.5°/m heading drift, as a
stand-in for the worse odometry of the real robot.

## Report figures and tables

```bash
~/.venvs/hcr_slam/bin/python report_figs.py office_vlp16      # after both experiment runs (~3 min, maps included)
```
- `fig_pipeline`: the system and the options compared at each stage.
- `fig_heatmap`: APE (final, online), strategy × front-end.
- `fig_loops`: correct vs false loops per strategy.
- `fig_robustness`: APE vs injected false loops.
- `fig_runtime`: ms per keyframe by stage, against the keyframe period.
- `fig_maps`: maps coloured by distance to the true walls.
- `table_summary.{tex,md}`: per front-end, odometry only vs the recommended back-end.
- `table_all.{tex,md}`: every strategy and every metric in one table (`table*`, needs `booktabs` and `graphicx`).
