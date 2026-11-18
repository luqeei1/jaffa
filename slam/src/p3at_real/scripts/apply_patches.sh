#!/usr/bin/env bash
# Apply our local changes to the third-party submodules in slam/src. Safe to run twice.
#
#   slam/src/p3at_real/scripts/apply_patches.sh [path to slam/src]
#
# p2os2: cmd_vel keep-alive (cmd_vel_resend_period) and cmd_vel_wait_sip, used by robot.launch.py.
#        Without the resend, a held constant speed stalls into short bursts (ARCOS watchdog).
# unilidar_sdk2: start the L2 like Unitree's UDP example (see patch_unilidar_driver.sh).
set -euo pipefail
here=$(dirname "$(readlink -f "$0")")
src=${1:-$(readlink -f "$here/../..")}
patch=$here/../patches/p2os2_cmd_vel_resend.patch

[[ -d $src/p2os2/p2os_driver ]] || { echo "p2os2 missing in $src: run 'git submodule update --init'"; exit 1; }
if git -C "$src/p2os2" apply --reverse --check "$patch" 2>/dev/null; then
  echo "already patched: p2os2"
else
  git -C "$src/p2os2" apply "$patch"
  echo "patched: p2os2"
fi

bash "$here/patch_unilidar_driver.sh" "$src/unilidar_sdk2"
