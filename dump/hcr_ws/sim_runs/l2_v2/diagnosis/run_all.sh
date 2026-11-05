#!/usr/bin/env bash
# Run the four mini-odometry variants on one world in parallel; results in out_<residual>_<prior>.txt
cd "$(dirname "$0")/../../.."   # workspace root
world=${1:-office}
for v in plane:gyro point:gyro plane:cv point:cv; do
  python3 sim_runs/l2_v2/diagnosis/mini_odom.py sim_runs/l2_v2 "$world" "$v" \
    > "sim_runs/l2_v2/diagnosis/out_${world}_${v/:/_}.txt" 2>&1 &
done
wait
grep -h -E "APE|invalid|failed" sim_runs/l2_v2/diagnosis/out_${world}_*.txt
