#!/usr/bin/env bash
# Full benchmark: per world, record the route once, replay it into every method, build the
# maps (scored against the true world geometry), then summarise across worlds.
#
#   benchmark.sh <tag> [world ...]            (default: all five worlds)
#   METHODS="wheel_odom kiss_icp" benchmark.sh quick office
#
# Output: bags/<world>_<tag>/, results/<world>_<tag>/<method>/, results/<world>_<tag>/maps/,
#         results/benchmark_<tag>/ (tables + plots). A world whose bag exists is not re-recorded.
set -euo pipefail
tag=${1:?usage: benchmark.sh <tag> [world ...]}
shift
if (($#)); then worlds=("$@"); else worlds=(office warehouse corridor hall split_level); fi
methods=(${METHODS:-wheel_odom kiss_icp kiss_icp_indoor})

prefix=$(ros2 pkg prefix p3at_sim)
lib=$prefix/lib/p3at_sim
share=$prefix/share/p3at_sim
ws=$(cd "$(dirname "$(readlink -f "$0")")" && while [[ ! -d install && $PWD != / ]]; do cd ..; done; pwd)
bags=${HCR_BAGS:-$ws/bags}
results=${HCR_RESULTS:-$ws/results}

for w in "${worlds[@]}"; do
  ds=${w}_${tag}
  echo "=== $ds"
  [[ -d $bags/$ds ]] || "$lib/record_route.sh" "$w" "$tag" || { echo "recording $ds failed, skipping"; continue; }
  map_args=()
  for m in "${methods[@]}"; do
    "$lib/run_method.sh" "$bags/$ds" "$m" | grep -E '"(ape_rmse|drift_pct|best_offset_s)"' || true
    read -r odom offset launch < <(python3 -c "
import sys, yaml
m = yaml.safe_load(open('$share/config/methods.yaml'))['$m']
print(m['odom'], m.get('time_offset', 0.0), m.get('launch') or '')")
    src=$results/$ds/$m/traj
    [[ -z $launch ]] && src=$bags/$ds  # estimate already in the dataset bag
    map_args+=(--method "$m" "$src" "$odom" "$offset")
  done
  mkdir -p "$results/$ds/maps"
  echo "building maps..."
  python3 "$lib/build_map.py" "$bags/$ds" "${map_args[@]}" --world "$share/worlds/$w.sdf" \
    --out "$results/$ds/maps" > "$results/$ds/maps/build_map.log" 2>&1 \
    || echo "build_map failed for $ds (see $results/$ds/maps/build_map.log)"
done
python3 "$lib/summarize.py" "$results" "$tag"
