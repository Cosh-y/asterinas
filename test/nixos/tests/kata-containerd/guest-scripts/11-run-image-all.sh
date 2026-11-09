#!/bin/sh
set -u

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${KATA_DEBUG_OUT:-/tmp/kata-debug-out}
mkdir -p "$OUT_DIR"

echo "=== check kata env ==="
sh "$DIR/00-check-kata-env.sh" | tee "$OUT_DIR/check.log"

echo
echo "=== start debug containerd ==="
sh "$DIR/01-start-containerd-debug.sh"

echo
echo "=== start qemu serial watcher ==="
sh "$DIR/03-watch-qemu-serial.sh" > "$OUT_DIR/qemu-serial-driver.log" 2>&1 &
echo $! > "$OUT_DIR/qemu-serial-watcher.pid"
echo "watcher pid: $(cat "$OUT_DIR/qemu-serial-watcher.pid")"

echo
echo "=== run kata image probe ==="
sh "$DIR/20-run-image-probe.sh"

echo
echo "=== collect logs ==="
sleep 2
COLLECT_TIMEOUT=${KATA_DEBUG_COLLECT_TIMEOUT:-60s}
timeout "$COLLECT_TIMEOUT" sh "$DIR/04-collect-logs.sh" \
  > "$OUT_DIR/collect.stdout.log" 2>&1
COLLECT_STATUS=$?
echo "collect-timeout:$COLLECT_TIMEOUT status:$COLLECT_STATUS" \
  | tee -a "$OUT_DIR/collect.stdout.log"
tail -240 "$OUT_DIR/collect.stdout.log" 2>/dev/null || true

echo
echo "done; outputs are under $OUT_DIR"
