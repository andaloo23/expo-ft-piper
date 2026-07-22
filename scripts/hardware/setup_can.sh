#!/usr/bin/env bash
# Name and raise the rig's CAN interfaces on the HOST (no container needed).
# Idempotent: safe to re-run; skips interfaces already named and up.
#
#   sudo scripts/hardware/setup_can.sh
#
# Identity comes from the USB adapter serial numbers recorded in
# configs/hardware/can_adapters.yaml (this script embeds the same map so it
# has no dependencies — update BOTH places if an adapter is ever replaced).

set -euo pipefail

BITRATE=1000000

declare -A SERIAL_TO_NAME=(
  ["002300304148571420343133"]="can_left"
  ["003F003C4148571320343133"]="can_right"
  ["003D002E4148571320343133"]="can_left_mst"
  ["004E00244148571420343133"]="can_right_mst"
)

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (sudo $0)" >&2
  exit 1
fi

found=0
for dev in /sys/class/net/*; do
  iface=$(basename "$dev")
  [[ -e "$dev/device" ]] || continue
  # Only consider CAN link types
  [[ $(cat "$dev/type" 2>/dev/null) == "280" ]] || continue
  usb_dir=$(dirname "$(readlink -f "$dev/device")")
  serial=$(cat "$usb_dir/serial" 2>/dev/null || true)
  name=${SERIAL_TO_NAME[$serial]:-}
  if [[ -z "$name" ]]; then
    echo "WARN: CAN interface $iface (serial '${serial:-none}') not in adapter map — leaving untouched"
    continue
  fi
  found=$((found + 1))

  if [[ "$iface" != "$name" ]]; then
    echo "renaming $iface -> $name (serial $serial)"
    ip link set "$iface" down
    ip link set "$iface" name "$name"
    iface="$name"
  fi

  state=$(cat "/sys/class/net/$iface/operstate" 2>/dev/null || echo down)
  if [[ "$state" != "up" ]]; then
    echo "bringing up $iface @ ${BITRATE} bit/s"
    ip link set "$iface" down 2>/dev/null || true
    ip link set "$iface" type can bitrate "$BITRATE"
    ip link set "$iface" up
  else
    echo "$iface already up"
  fi
done

if [[ $found -ne 4 ]]; then
  echo "WARN: expected 4 known adapters, found $found — check USB connections" >&2
  exit 2
fi
echo "all CAN interfaces named and up"
