#!/usr/bin/env bash
# Replay a recorded dataset bag into one method and score its trajectory.
#
#   run_method.sh <dataset_bag_dir> <method> [play rate]
#   e.g. run_method.sh bags/office_base kiss_icp 0.5
#
# Methods are defined in config/methods.yaml. Output: results/<dataset>/<method>/
# (traj bag with /ground_truth + estimate, method log, metrics.json), and a row in
# results/summary.csv.
set -euo pipefail
set -m  # job control: background jobs keep default SIGINT handling, so kill -INT works
bag=$(readlink -f "${1:?usage: run_method.sh <bag_dir> <method> [rate]}")
method=${2:?method name from config/methods.yaml}
rate=${3:-1.0}

prefix=$(ros2 pkg prefix p3at_sim)
share=$prefix/share/p3at_sim
ws=$(cd "$(dirname "$(readlink -f "$0")")" && while [[ ! -d install && $PWD != / ]]; do cd ..; done; pwd)
read -r odom offset launch < <(python3 - "$share/config/methods.yaml" "$method" <<'EOF'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]
print(m['odom'], m.get('time_offset', 0.0), m.get('launch') or '')
EOF
)
out=${HCR_RESULTS:-$ws/results}/$(basename "$bag")/$method
rm -rf "$out"; mkdir -p "$out"
summary=$(dirname "$(dirname "$out")")/summary.csv
evaluate() { python3 "$prefix/lib/p3at_sim/evaluate.py" "$1" "$odom" --time-offset "$offset" --plot --out "$out" \
  --name "$(basename "$bag")" --method "$method" --summary "$summary"; }

if [[ -z $launch ]]; then  # estimate already in the dataset (e.g. wheel odometry): no replay needed
  evaluate "$bag"; exit
fi

cleanup() { kill -INT ${rec:-} ${meth:-} 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

bash -c "exec $launch" > "$out/method.log" 2>&1 &
meth=$!
sleep 5
ros2 bag record -o "$out/traj" /ground_truth "$odom" > "$out/rec.log" 2>&1 &
rec=$!
sleep 2
echo "replaying $(basename "$bag") into $method at ${rate}x..."
ros2 bag play "$bag" --rate "$rate" > "$out/play.log" 2>&1
sleep 3
cleanup
evaluate "$out/traj"
