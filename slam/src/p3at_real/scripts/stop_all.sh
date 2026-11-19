#!/usr/bin/env bash
# Stop every robot / SLAM / Nav2 / Foxglove process on the Jetson, including ones orphaned by a
# closed terminal or SSH session, so the next run starts clean.
#
#   ros2 run p3at_real stop_all.sh            # Ctrl-C everything, wait, then force what is left
#   ros2 run p3at_real stop_all.sh --dry-run  # only list what would be stopped
#
# Order: launchers first (they shut their nodes down cleanly), then any nodes still alive, both
# with SIGINT (= Ctrl-C, so bags, rtabmap.db and RKO-LIO results get written). Whatever survives
# GRACE seconds is killed with SIGKILL. Finally the ROS 2 daemon is restarted (stale node/topic
# lists) and the base serial port is checked.
#
# Run folders in slam/runs/ are not touched: nothing loads them, each run starts a fresh map.
set -u

GRACE=${GRACE:-20}   # s to wait after Ctrl-C (rtabmap may take a while to save its database)
DRY=0
[[ ${1:-} == --dry-run || ${1:-} == -n ]] && DRY=1

# Launchers / wrappers: stopping these first lets ros2 launch stop its own children.
LAUNCHERS='run_slam\.sh|run_odometry\.sh|ros2 launch (p3at_real|foxglove_bridge)|ros2 bag record'
# Everything our launch files start (robot, lidar, odometry, SLAM, Nav2, Foxglove, logging).
NODES='cmd_vel_mux\.py|teleop_receiver\.py|wasd_teleop\.py|run_logger\.py|wait_for_data\.py'
NODES+='|p2os_driver/p2os_driver|unitree_lidar_ros2_node|static_transform_publisher.*unilidar'
NODES+='|rko_lio/online_node|kiss_icp_node|rtabmap_slam/rtabmap|rtabmap_util/|map_saver_cli'
NODES+='|pointcloud_to_laserscan_node|nav2_[a-z_]+/[a-z_]+|foxglove_bridge/foxglove_bridge'
NODES+='|ros2 bag record'

# PIDs matching a pattern, minus this script and its parents (ros2 run, the shell).
find_pids() {
    local skip=" $$ $PPID $(ps -o ppid= -p "$PPID" 2>/dev/null | tr -d ' ') "
    pgrep -f -- "$1" | while read -r pid; do
        [[ $skip == *" $pid "* ]] || echo "$pid"
    done
}

show() { [[ -n $1 ]] && ps -o pid=,etime=,args= -p "$(echo $1 | tr ' ' ',')" | cut -c1-140; }

wait_gone() {  # wait_gone <seconds> <pattern>: wait until nothing matches, or time runs out
    local t=0
    while [[ -n $(find_pids "$2") && $t -lt $1 ]]; do sleep 1; t=$((t + 1)); done
}

all=$(find_pids "$LAUNCHERS|$NODES" | xargs)
if [[ -z $all ]]; then
    echo "Nothing running."
else
    echo "Running now:"
    show "$all"
    if [[ $DRY -eq 1 ]]; then
        echo "(dry run: nothing stopped)"
        exit 0
    fi

    launchers=$(find_pids "$LAUNCHERS" | xargs)
    if [[ -n $launchers ]]; then
        echo "Ctrl-C to launchers: $launchers"
        kill -INT $launchers 2>/dev/null
        wait_gone $((GRACE / 2)) "$LAUNCHERS"
    fi

    nodes=$(find_pids "$NODES" | xargs)
    if [[ -n $nodes ]]; then
        echo "Ctrl-C to remaining nodes: $nodes"
        kill -INT $nodes 2>/dev/null
        wait_gone "$GRACE" "$LAUNCHERS|$NODES"
    fi

    left=$(find_pids "$LAUNCHERS|$NODES" | xargs)
    if [[ -n $left ]]; then
        echo "Still alive after ${GRACE}s, forcing (SIGKILL):"
        show "$left"
        kill -9 $left 2>/dev/null
        sleep 1
    fi
fi

[[ $DRY -eq 1 ]] && exit 0

ros2 daemon stop > /dev/null 2>&1 && echo "ROS 2 daemon restarted (stale node/topic lists cleared)."

status=0
left=$(find_pids "$LAUNCHERS|$NODES" | xargs)
if [[ -n $left ]]; then
    echo "WARNING: could not stop:"; show "$left"; status=1
fi
for port in /dev/ttyUSB0; do
    if [[ -e $port ]] && users=$(fuser "$port" 2>/dev/null) && [[ -n ${users// /} ]]; then
        echo "WARNING: $port still held by PID(s):$users"; status=1
    fi
done
[[ $status -eq 0 ]] && echo "All stopped. Start again with: ros2 run p3at_real run_slam.sh <name>"
exit $status
