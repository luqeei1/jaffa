#!/usr/bin/env bash
# Drive a world's route headless and record a dataset bag for offline LIO replay.
#
#   record_route.sh <world> [tag] [launch args...] [-- route_driver params...]
#   e.g. record_route.sh office base
#        record_route.sh corridor fast lidar_noise:=0.03 -- -p max_v:=1.0 -p wobble:=0.4
#
# Output: $HCR_BAGS/<world>_<tag>/ (default HCR_BAGS=<ws>/bags)
set -euo pipefail
set -m  # job control: background jobs keep default SIGINT handling, so kill -INT works
world=${1:?usage: record_route.sh <world> [tag] [launch args...] [-- driver params...]}
tag=${2:-base}
shift $(( $# >= 2 ? 2 : 1 ))
launch_args=(); driver_args=()
while (($#)); do
  if [[ $1 == -- ]]; then shift; driver_args=("$@"); break; fi
  launch_args+=("$1"); shift
done

ws=$(cd "$(dirname "$(readlink -f "$0")")" && while [[ ! -d install && $PWD != / ]]; do cd ..; done; pwd)
out=${HCR_BAGS:-$ws/bags}/${world}_${tag}
[[ -e $out ]] && { echo "exists: $out" >&2; exit 1; }
mkdir -p "$(dirname "$out")"

cleanup() {
  kill -INT ${rec:-} ${sim:-} 2>/dev/null || true; wait 2>/dev/null || true
  # The ign ruby wrapper can outlive `ros2 launch`; a stale server corrupts the next run
  pkill -f '[g]azebo.*p3at_sim/worlds' 2>/dev/null || true
}
pgrep -f '[g]azebo.*p3at_sim/worlds' > /dev/null && { echo "a p3at_sim Gazebo is already running; stop it first" >&2; exit 1; }
trap cleanup EXIT

ros2 launch p3at_sim sim.launch.py world:="$world" gui:=false "${launch_args[@]}" > "$out.sim.log" 2>&1 &
sim=$!
echo "waiting for sensors..."
timeout 90 ros2 topic echo --once --qos-reliability best_effort /points sensor_msgs/msg/PointCloud2 --field width > /dev/null
# A stale lidar node from an earlier sim also publishes /points and silently doubles every scan
pubs=$(ros2 topic info /points | sed -n 's/^Publisher count: //p')
[[ $pubs == 1 ]] || { echo "/points has $pubs publishers (a stale l2_lidar/cloud_fields node?): stop it first" >&2; exit 1; }
ros2 bag record -o "$out.wall" /points /imu /ground_truth /odom /tf /tf_static /clock > "$out.rec.log" 2>&1 &
rec=$!
sleep 3
echo "driving route '$world' (sim runs slower than real time with software rendering)..."
ros2 run p3at_sim route_driver.py --ros-args -p use_sim_time:=true -p world:="$world" "${driver_args[@]}"
sleep 2
cleanup
# Log times are wall clock; re-time onto /clock so a 1x replay runs at simulated real time
python3 "$(ros2 pkg prefix p3at_sim)/lib/p3at_sim/retime_bag.py" "$out.wall" "$out"
rm -rf "$out.wall"
mv "$out.sim.log" "$out.rec.log" "$out/"
echo "saved $out"
ros2 bag info "$out" | grep -E 'Duration|Size|Messages'
