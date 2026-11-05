# Simulation benchmark runs

One folder per run, made by `HCR_RUN=sim_runs/<run> ros2 run p3at_sim benchmark.sh <tag> [worlds]`.
Sensor: simulated Unitree L2 (`jaffa/simulator/p3at_sim`, `l2_lidar.py`) + its built-in IMU.

```
<run>/
├── run_info.yaml                 date, code commit, ROS/Gazebo version, sensor settings, methods.yaml copy
├── benchmark.log                 console log of the whole run
├── bags/<world>_<tag>/           dataset bag: /points /imu /ground_truth /odom /tf /tf_static /clock
│                                 (+ sim.log, rec.log from the recording)
└── results/
    ├── <world>_<tag>/
    │   ├── <method>/             metrics.json, traj_xy.png, traj/ (estimate + GT bag), method/play logs
    │   ├── maps/                 map_<source>.ply, map_metrics.json, maps/errors/trajectory plots
    │   └── scans/                every raw scan <idx>.ply (sensor frame, x y z intensity ring time)
    │                             + poses_gt.tum (lidar pose per scan: t x y z qx qy qz qw)
    ├── summary.csv               one row per (world, method) as it was scored
    ├── benchmark_<tag>/          summary.md, odometry.csv, maps.csv, overview plots
    └── paper_<tag>/              fig_*.pdf (+ .png previews), table_odometry.tex
```

Metrics: APE RMSE (SE(3)-aligned), RPE over 1 m (translation, rotation), end drift % (start-aligned),
z RMSE, map point-to-true-surface distance. See `src/p3at_sim/scripts/evaluate.py` and `build_map.py`.
