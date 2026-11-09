#!/bin/sh
set -eu

OUT_DIR=${SECURITY_DEMO_OUT:-/tmp/security-demo}
HOST_CANARY=${SECURITY_DEMO_HOST_CANARY:-/run/security-demo-host-canary}
ATTACK_MODE=${SECURITY_DEMO_ATTACK_MODE:-fixture}
TASK_ID=${AGENT_TASK_ID:-security-demo-fixture}
TASK_ROLE=${AGENT_TASK_ROLE:-attack-probe}
BB=/bin/busybox

if ! mkdir -p "$OUT_DIR" 2>/dev/null; then
  # Some Kata rootfs configurations replace /tmp with a non-writable mount.
  # Keep the task result on stdout so the outer runner can still collect it.
  OUT_DIR=
fi

log() {
  if [ -n "$OUT_DIR" ]; then
    printf '%s\n' "$*" | tee -a "$OUT_DIR/task.log" 2>/dev/null || printf '%s\n' "$*"
  else
    printf '%s\n' "$*"
  fi
}

probe_path() {
  path=$1
  if [ -e "$path" ] || [ -L "$path" ]; then
    log "BOUNDARY_VISIBLE path=$path"
  else
    log "BOUNDARY_BLOCKED path=$path"
  fi
}

log "SECURITY_DEMO_TASK_ID=$TASK_ID"
log "AGENT_TASK_ROLE=$TASK_ROLE"
log "SECURITY_DEMO_ATTACK_MODE=$ATTACK_MODE"
log "TASK_UID=$($BB id -u)"
log "TASK_GID=$($BB id -g)"
log "TASK_KERNEL=$($BB uname -sr)"
log "TASK_HOSTNAME=$($BB hostname 2>/dev/null || printf unknown)"

if [ "$(id -u)" -eq 0 ]; then
  log "TASK_STARTED_AS_ROOT=1"
else
  log "TASK_STARTED_AS_ROOT=0"
fi

if [ "$TASK_ROLE" = recovery ]; then
  log "RECOVERY_TASK=normal-agent-task"
  log "SECURITY_DEMO_TASK_DONE=1"
  exit 0
fi

log "ATTACK_TARGET_CANARY=$HOST_CANARY"
probe_path "$HOST_CANARY"
probe_path /host-secret
probe_path /run/containerd/containerd.sock
probe_path /run/kata-containers
probe_path /var/run/docker.sock
probe_path /dev/kvm
probe_path /dev/mem

case "$ATTACK_MODE" in
  fixture)
    # This fixture validates task dispatch and the boundary-observation path.
    # It deliberately does not claim a kernel compromise.
    log "ATTACK_FIXTURE=task-boundary-probe-only"
    log "GUEST_COMPROMISED=not_claimed"
    ;;
  cve-2022-0847)
    cve_target=${CVE_PAYLOAD_TARGET:-/opt/guest-canary}
    log "CVE_ID=CVE-2022-0847"
    probe_path "$cve_target"
    if [ ! -x /opt/dirty-pipe ]; then
      log "CVE_PAYLOAD=missing"
      exit 2
    fi

    payload_status=0
    payload_output=$(/opt/dirty-pipe "$cve_target" 2>&1) || payload_status=$?
    if [ -n "$payload_output" ]; then
      log "$payload_output"
    fi
    log "CVE_PAYLOAD_EXIT=$payload_status"
    if [ -r "$cve_target" ]; then
      cve_content=$($BB cat "$cve_target" | $BB tr '\n' ' ')
      log "CVE_TARGET_CONTENT=$cve_content"
      case "$cve_content" in
        gXest-canary-original\ *)
          log "CVE_REPRODUCTION=success"
          ;;
        *)
          log "CVE_REPRODUCTION=not_observed"
          ;;
      esac
    fi
    log "CVE_TASK_DONE=1"
    ;;
  payload)
    if [ ! -x /opt/attack-payload.sh ]; then
      log "ATTACK_PAYLOAD=missing"
      exit 2
    fi
    if [ -n "$OUT_DIR" ]; then
      /opt/attack-payload.sh 2>&1 | tee -a "$OUT_DIR/payload.log"
    else
      /opt/attack-payload.sh
    fi
    ;;
  *)
    log "SECURITY_DEMO_ERROR=unsupported_attack_mode"
    exit 2
    ;;
esac

log "SECURITY_DEMO_TASK_DONE=1"
