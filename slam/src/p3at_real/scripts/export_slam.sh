#!/usr/bin/env bash
# Export a SLAM run's RTAB-Map database into plain files.
#
#   ros2 run p3at_real export_slam.sh ~/hcr_runs/latest
#
# Writes <run>/slam/: rtabmap_cloud.ply (3D map from the optimised poses, 5 cm voxels) and
# rtabmap_poses.txt (optimised keyframe poses: stamp x y z qx qy qz qw, i.e. TUM, for evo).
set -euo pipefail
run=$(readlink -f "${1:?usage: export_slam.sh <run folder>}")
mkdir -p "$run/slam"
rtabmap-export --scan --cloud --voxel 0.05 --output rtabmap --output_dir "$run/slam" "$run/rtabmap.db"
rtabmap-export --poses --poses_format 10 --output rtabmap --output_dir "$run/slam" "$run/rtabmap.db"
ls -la "$run/slam"
