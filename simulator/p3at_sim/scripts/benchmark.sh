#!/usr/bin/env bash
# Full benchmark: per world, record the route once, replay it into every method, build the
# maps (scored against the true world geometry), then summarise across worlds.
#
#   benchmark.sh <tag> [world ...]            (default: all five worlds)
#   METHODS="wheel_odom kiss_icp" benchmark.sh quick office
#   HCR_RUN=sim_runs/l2_baseline benchmark.sh l2 (everything for the run under one folder)
#   SIM_ARGS="lidar:=vlp16" HCR_RUN=sim_runs/vlp16 benchmark.sh vlp16  (sim launch args for recording)
#
# Output (HCR_RUN set, else bags/ and results/ in the workspace):
#   <run>/run_info.yaml                   date, code version, sensor + sim settings, methods
#   <run>/bags/<world>_<tag>/             dataset bag (/points /imu /ground_truth /odom /tf)
#   <run>/results/<world>_<tag>/<method>/ metrics.json, traj/ (estimate + GT bag), logs
#   <run>/results/<world>_<tag>/maps/     map_<source>.ply, map_metrics.json, plots
#   <run>/results/<world>_<tag>/scans/    every raw scan as <idx>.ply + poses_gt.tum (EXPORT_SCANS=0 skips)
#   <run>/results/benchmark_<tag>/        summary.md, odometry.csv, maps.csv
#   <run>/results/paper_<tag>/            paper figures (PDF) + LaTeX table
# A world whose bag exists is not re-recorded.
set -euo pipefail
tag=${1:?usage: benchmark.sh <tag> [world ...]}
shift
if (($#)); then worlds=("$@"); else worlds=(office warehouse corridor hall split_level); fi
methods=(${METHODS:-wheel_odom kiss_icp kiss_icp_indoor})
sim_args=(${SIM_ARGS:-})

prefix=$(ros2 pkg prefix p3at_sim)
lib=$prefix/lib/p3at_sim
share=$prefix/share/p3at_sim
ws=$(cd "$(dirname "$(readlink -f "$0")")" && while [[ ! -d install && $PWD != / ]]; do cd ..; done; pwd)
if [[ -n ${HCR_RUN:-} ]]; then
  mkdir -p "$HCR_RUN"; run=$(readlink -f "$HCR_RUN")
  export HCR_BAGS=$run/bags HCR_RESULTS=$run/results
else
  run=$ws
fi
bags=${HCR_BAGS:-$ws/bags}
results=${HCR_RESULTS:-$ws/results}
mkdir -p "$bags" "$results"

pkg_src=$(readlink -f "$share/urdf/p3at.urdf.xacro"); pkg_src=${pkg_src%/urdf/*}
{
  echo "tag: $tag"
  echo "started: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "package_source: $pkg_src"
  echo "git_commit: $(git -C "$pkg_src" rev-parse --short HEAD 2>/dev/null || echo none)"
  echo "git_dirty: $([[ -n $(git -C "$pkg_src" status --porcelain . 2>/dev/null) ]] && echo true || echo false)"
  echo "ros_distro: ${ROS_DISTRO:-}"
  echo "gazebo: $(ign gazebo --versions 2>/dev/null | head -1)"
  echo "worlds: [${worlds[*]}]"
  echo "methods: [${methods[*]}]"
  echo "sensor_defaults: {$(grep -oP "SENSOR_ARGS = \{\K[^}]*" -z "$share/launch/sim.launch.py" | tr -d '\n\0' | tr -s ' ')}"
  echo "sim_args: [${sim_args[*]}]"
  lidar=$(printf '%s\n' "${sim_args[@]}" | sed -n 's/^lidar:=//p'); lidar=${lidar:-l2}
  echo "lidar_model: $lidar ($([[ $lidar == vlp16 ]] && echo '16-ring spinning, cloud_fields.py' || echo 'Unitree L2, l2_lidar.py'))"
  echo "methods_yaml: |"; sed 's/^/  /' "$share/config/methods.yaml"
} > "$run/run_info.yaml"

for w in "${worlds[@]}"; do
  ds=${w}_${tag}
  echo "=== $ds  ($(date +%H:%M))"
  [[ -d $bags/$ds ]] || "$lib/record_route.sh" "$w" "$tag" "${sim_args[@]}" || { echo "recording $ds failed, skipping"; continue; }
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
  scan_args=()
  [[ ${EXPORT_SCANS:-1} == 1 ]] && scan_args=(--export-scans "$results/$ds/scans")
  echo "building maps..."
  python3 "$lib/build_map.py" "$bags/$ds" "${map_args[@]}" --world "$share/worlds/$w.sdf" \
    --out "$results/$ds/maps" "${scan_args[@]}" > "$results/$ds/maps/build_map.log" 2>&1 \
    || echo "build_map failed for $ds (see $results/$ds/maps/build_map.log)"
done
echo "finished: $(date -Iseconds)" >> "$run/run_info.yaml"
python3 "$lib/summarize.py" "$results" "$tag"
python3 "$lib/paper_plots.py" "$results" "$tag" --bags "$bags" || echo "paper_plots failed"
