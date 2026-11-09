#!/bin/sh
set -u

OUT_DIR=${KATA_DEBUG_OUT:-/tmp/kata-debug-out}
ADDR=${CONTAINERD_ADDRESS:-/tmp/containerd-debug.sock}
STATE=${CONTAINERD_STATE:-/tmp/containerd-state}
IMAGE_TAR=${SECURITY_DEMO_IMAGE_TAR:-/etc/kata-debug/security-demo.tar}
IMAGE_REF=${SECURITY_DEMO_IMAGE_REF:-agent-security-demo:fixture-v1}
CID=${SECURITY_DEMO_CID:-agent-security-demo}
RECOVERY_CID=${SECURITY_DEMO_RECOVERY_CID:-agent-security-demo-recovery}
HOST_CANARY=${SECURITY_DEMO_HOST_CANARY:-/run/security-demo-host-canary}
RUN_TIMEOUT=${SECURITY_DEMO_RUN_TIMEOUT:-300s}
ATTACK_MODE=${SECURITY_DEMO_ATTACK_MODE:-fixture}

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/security-demo.log"

log() {
  echo "$*" | tee -a "$LOG"
}

if [ ! -S "$ADDR" ]; then
  if [ -S /run/containerd/containerd.sock ]; then
    ADDR=/run/containerd/containerd.sock
  else
    log "SECURITY_DEMO_ERROR=containerd_socket_missing"
    exit 1
  fi
fi

printf 'host-canary-original\n' > "$HOST_CANARY"
log "SECURITY_DEMO_HOST_CANARY=$HOST_CANARY"
log "SECURITY_DEMO_CONTAINERD=$ADDR"
log "SECURITY_DEMO_ATTACK_MODE=$ATTACK_MODE"

ctr --address "$ADDR" tasks kill "$CID" >/dev/null 2>&1 || true
ctr --address "$ADDR" containers rm "$CID" >/dev/null 2>&1 || true
rm -f /tmp/kata-security-demo-run.txt /tmp/kata-security-demo-import.txt
rm -rf "$STATE/io.containerd.runtime.v2.task/default/$CID" \
  "/run/kata/$CID" \
  "/run/kata-containers/shared/sandboxes/$CID"

if [ ! -r "$IMAGE_TAR" ]; then
  log "SECURITY_DEMO_ERROR=image_tar_missing path=$IMAGE_TAR"
  exit 1
fi

timeout 120s ctr --debug --address "$ADDR" image import "$IMAGE_TAR" \
  > /tmp/kata-security-demo-import.txt 2>&1
import_status=$?
cat /tmp/kata-security-demo-import.txt >> "$LOG"
log "SECURITY_DEMO_IMPORT_STATUS=$import_status"
if [ "$import_status" -ne 0 ]; then
  exit 1
fi

if ! ctr --address "$ADDR" images ls -q 2>/dev/null \
  | grep -Fqx "$IMAGE_REF"; then
  imported_ref=$(ctr --address "$ADDR" images ls -q 2>/dev/null \
    | awk '/agent|security/ { print; exit }')
  if [ -n "${imported_ref:-}" ]; then
    IMAGE_REF=$imported_ref
  fi
fi
log "SECURITY_DEMO_IMAGE_REF=$IMAGE_REF"

timeout "$RUN_TIMEOUT" ctr --debug --address "$ADDR" run \
  --runtime io.containerd.kata.v2 \
  --user 1000:1000 \
  --env "SECURITY_DEMO_ATTACK_MODE=$ATTACK_MODE" \
  --env CVE_PAYLOAD_TARGET=/opt/guest-canary \
  "$IMAGE_REF" "$CID" /opt/agent-task.sh \
  > /tmp/kata-security-demo-run.txt 2>&1
run_status=$?
cat /tmp/kata-security-demo-run.txt | tee -a "$LOG"
log "SECURITY_DEMO_TASK_EXIT=$run_status"

if [ "$run_status" -eq 0 ]; then
  log '=== security demo recovery task ==='
  ctr --address "$ADDR" tasks kill "$RECOVERY_CID" >/dev/null 2>&1 || true
  ctr --address "$ADDR" containers rm "$RECOVERY_CID" >/dev/null 2>&1 || true
  timeout "$RUN_TIMEOUT" ctr --debug --address "$ADDR" run \
    --runtime io.containerd.kata.v2 \
    --user 1000:1000 \
    --env AGENT_TASK_ID=security-demo-recovery \
    --env AGENT_TASK_ROLE=recovery \
    "$IMAGE_REF" "$RECOVERY_CID" /opt/agent-task.sh \
    > /tmp/kata-security-demo-recovery.txt 2>&1
  recovery_status=$?
  cat /tmp/kata-security-demo-recovery.txt | tee -a "$LOG"
  log "SECURITY_DEMO_RECOVERY_TASK_EXIT=$recovery_status"

  if [ "$recovery_status" -eq 0 ] \
    && grep -q '^AGENT_TASK_ROLE=recovery$' "$LOG" \
    && grep -q '^RECOVERY_TASK=normal-agent-task$' "$LOG"; then
    log 'SECURITY_DEMO_RECOVERY=pass'
  else
    log 'SECURITY_DEMO_RECOVERY=fail'
  fi
else
  recovery_status=1
  log 'SECURITY_DEMO_RECOVERY=skipped'
fi

cve_result=pass
if [ "$ATTACK_MODE" = cve-2022-0847 ] \
  && ! grep -q '^CVE_REPRODUCTION=success$' "$LOG"; then
  cve_result=fail
fi
log "SECURITY_DEMO_CVE_RESULT=$cve_result"

if grep -qx 'host-canary-original' "$HOST_CANARY"; then
  log 'SECURITY_DEMO_HOST_CANARY=unchanged'
else
  log 'SECURITY_DEMO_HOST_CANARY=modified'
fi

log '=== security demo task state ==='
find "$STATE/io.containerd.runtime.v2.task/default/$CID" -maxdepth 6 -ls \
  2>&1 | tee -a "$LOG" || true

if [ "$run_status" -eq 0 ] \
  && [ "$recovery_status" -eq 0 ] \
  && [ "$cve_result" = pass ] \
  && grep -q '^SECURITY_DEMO_TASK_DONE=1$' "$LOG" \
  && grep -q '^SECURITY_DEMO_RECOVERY=pass$' "$LOG" \
  && grep -q '^SECURITY_DEMO_HOST_CANARY=unchanged$' "$LOG"; then
  log 'SECURITY_DEMO_RESULT=pass'
else
  log 'SECURITY_DEMO_RESULT=fail'
  exit 1
fi
