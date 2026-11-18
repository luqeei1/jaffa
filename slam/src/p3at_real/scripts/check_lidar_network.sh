#!/usr/bin/env bash
# Check the Ethernet path from the Unitree L2 to the Jetson: address, link, who owns the UDP port,
# and what the lidar actually sends (source, destination, packet size). Passive: it can run while
# the driver is up, e.g. during the start-up gap when IMU data arrives but no clouds.
#
#   ros2 run p3at_real check_lidar_network.sh [seconds]      (default 5; needs sudo for tcpdump)
#
# Packet sizes: 80 B = IMU, 1044 B = 3D points, 1124 B = both in one datagram (see l2_packet_dump.py),
# 40 B = lidar's reply to a driver command (32 B). Traffic the Jetson sends (driver commands, ROS 2 DDS
# discovery to 239.255.0.1:7400) is listed but not judged. Right after the driver resets the L2, the
# link drops (carrier=0) and only IMU arrives for a while: the lidar is restarting, not misaddressed.
set -uo pipefail
IF=enP8p1s0
secs=${1:-5}
want_src=192.168.1.63.6101   # lidar_ip:lidar_port in lidar.launch.py
want_dst=192.168.1.2.6202    # local_ip:local_port

echo "== 1. Jetson address and link ($IF)"
ip -br addr show "$IF"
echo "carrier=$(cat /sys/class/net/$IF/carrier 2>/dev/null) speed=$(cat /sys/class/net/$IF/speed 2>/dev/null) Mb/s"
echo "active connections on $IF:"; nmcli -t -f NAME,DEVICE con show --active | grep "$IF" || echo "  none!"

echo; echo "== 2. Devices seen on the link (ARP)"
ip neigh show dev "$IF"

echo; echo "== 3. Processes bound to the lidar ports (want: one unitree_lidar_ros2_node on 6202, or nothing)"
sudo ss -ulpn | grep -E ':(6101|620[0-9])\b' || echo "  nothing bound"

echo; echo "== 4. What arrives on $IF for $secs s: count  source  ->  destination  size"
sudo timeout "$secs" tcpdump -i "$IF" -n -q udp 2>/dev/null |
  awk '/ UDP, length [0-9]+$/ {sub(":", "", $5); print $3, "->", $5, $NF " B"}'   | sort | uniq -c | sort -rn | tee /tmp/l2_traffic.txt
[[ -s /tmp/l2_traffic.txt ]] || echo "  no UDP at all: lidar off, cable, or not in Ethernet mode"

echo; echo "== Verdict"
imu=$(awk -v s=$want_src -v d=$want_dst '$2==s && $4==d && $5==80 {n+=$1} END {print n+0}' /tmp/l2_traffic.txt)
pts=$(awk -v s=$want_src -v d=$want_dst '$2==s && $4==d && ($5==1044 || $5==1124) {n+=$1} END {print n+0}' /tmp/l2_traffic.txt)
# lidar traffic (from its IP) that does not go to the driver's address, or anything else sent to the driver's port
other=$(awk -v s=$want_src -v d=$want_dst -v ip=${want_src%.*} '
  (index($2, ip ".") == 1 && !($2 == s && $4 == d)) || ($4 == d && $2 != s) {n+=$1} END {print n+0}' /tmp/l2_traffic.txt)
echo "to $want_dst: IMU packets $imu, point packets $pts; lidar packets going elsewhere: $other"
if (( pts > 0 )); then
  echo "OK: point data reaches the driver's port ($(( pts / secs ))/s, expect ~215/s)."
  echo "    If clouds are still missing or empty, the problem is in the data, not the network: run l2_packet_dump.py --watch"
elif (( other > 0 )); then
  echo "PORT/ADDRESS PROBLEM: lidar traffic goes elsewhere (see section 4). Set lidar_ip/local_port to match."
elif (( imu > 0 )); then
  echo "IMU arrives but NO point packets at all: the lidar is not measuring yet."
  echo "    Normal for a while after the driver resets it (wait_for_lidar waits for this). If it lasts > 2 min, power-cycle."
else
  echo "NOTHING from the lidar on the expected addresses."
fi
