#!/bin/sh
set -u

ADDR=${CONTAINERD_ADDRESS:-/tmp/containerd-debug.sock}

usage() {
  echo "usage: $0 lifecycle|memory kvm|tcg" >&2
  exit 2
}

wait_task_stopped() {
  task_id=$1
  wait_seconds=$2
  state_log=$3
  deadline=$(( $(date +%s) + wait_seconds ))
  task_state=

  while [ "$(date +%s)" -lt "$deadline" ]; do
    task_state=$(ctr --address "$ADDR" tasks ls 2>/dev/null \
      | grep -F "$task_id" || true)
    if [ -z "$task_state" ] || printf '%s\n' "$task_state" | grep -q STOPPED; then
      printf '%s\n' "$task_state" > "$state_log"
      return 0
    fi
    sleep 0.05
  done

  printf '%s\n' "$task_state" > "$state_log"
  return 124
}

run_lifecycle() {
  accel=$1
  out=/tmp/kata-lifecycle-$accel.tsv
  image_ref=$(cat /tmp/kata-benchmark-image-ref)

  printf 'mode\titeration\tto_workload_ms\tto_quit_ms\ttotal_ms\trun_status\tready_status\tstop_status\ttask_rm_status\tcontainer_rm_status\n' \
    > "$out"
  i=1
  while [ "$i" -le 3 ]; do
    id=kata-lifecycle-$accel-$i
    ready_file=kata-lifecycle-ready-$accel-$i
    launch_start_ns=$(date +%s%N)

    timeout 180s ctr --address "$ADDR" run --detach \
      --runtime io.containerd.kata.v2 "$image_ref" "$id" \
      /bin/sh -c "echo ready > /$ready_file; sleep 2" \
      > "/tmp/kata-lifecycle-run-$accel-$i.stdout" 2>&1 &
    ctr_pid=$!
    run_status=124

    ready_status=124
    ready_deadline=$(( $(date +%s) + 180 ))
    while [ "$(date +%s)" -lt "$ready_deadline" ]; do
      marker=$(find /tmp/containerd-root \
        -path "*/snapshots/*/fs/$ready_file" -type f -print -quit 2>/dev/null)
      if [ -n "$marker" ]; then
        ready_status=0
        break
      fi
      if [ -n "$ctr_pid" ] && ! kill -0 "$ctr_pid" 2>/dev/null; then
        wait "$ctr_pid"
        run_status=$?
        ctr_pid=
        if [ "$run_status" -ne 0 ]; then
          break
        fi
        ready_deadline=$(( $(date +%s) + 10 ))
      fi
      sleep 0.05
    done
    ready_end_ns=$(date +%s%N)

    if [ -n "$ctr_pid" ]; then
      run_deadline=$(( $(date +%s) + 10 ))
      while kill -0 "$ctr_pid" 2>/dev/null && \
            [ "$(date +%s)" -lt "$run_deadline" ]; do
        sleep 0.05
      done
      if kill -0 "$ctr_pid" 2>/dev/null; then
        kill -KILL "$ctr_pid" >/dev/null 2>&1 || true
        wait "$ctr_pid" 2>/dev/null || true
        run_status=124
      else
        wait "$ctr_pid"
        run_status=$?
      fi
    fi

    if [ "$ready_status" -ne 0 ]; then
      timeout 10s ctr --address "$ADDR" tasks kill -s SIGKILL "$id" \
        > "/tmp/kata-lifecycle-kill-$accel-$i.stdout" 2>&1 || true
    fi
    wait_task_stopped "$id" 30 "/tmp/kata-lifecycle-stop-$accel-$i.stdout"
    stop_status=$?
    if [ "$stop_status" -ne 0 ]; then
      timeout 10s ctr --address "$ADDR" tasks kill -s SIGKILL "$id" \
        >> "/tmp/kata-lifecycle-stop-$accel-$i.stdout" 2>&1 || true
    fi

    task_state=$(ctr --address "$ADDR" tasks ls 2>/dev/null \
      | grep -F "$id" || true)
    if [ -z "$task_state" ]; then
      echo "task already absent" > "/tmp/kata-lifecycle-task-rm-$accel-$i.stdout"
      task_rm_status=0
    else
      timeout 30s ctr --address "$ADDR" tasks rm "$id" \
        > "/tmp/kata-lifecycle-task-rm-$accel-$i.stdout" 2>&1
      task_rm_status=$?
    fi
    if ctr --address "$ADDR" containers ls -q 2>/dev/null | grep -Fxq "$id"; then
      timeout 30s ctr --address "$ADDR" containers rm "$id" \
        > "/tmp/kata-lifecycle-container-rm-$accel-$i.stdout" 2>&1
      container_rm_status=$?
    else
      echo "container already absent" \
        > "/tmp/kata-lifecycle-container-rm-$accel-$i.stdout"
      container_rm_status=0
    fi
    quit_end_ns=$(date +%s%N)

    to_workload_ms=$(( (ready_end_ns - launch_start_ns) / 1000000 ))
    to_quit_ms=$(( (quit_end_ns - ready_end_ns) / 1000000 ))
    total_ms=$(( (quit_end_ns - launch_start_ns) / 1000000 ))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$accel" "$i" "$to_workload_ms" "$to_quit_ms" "$total_ms" \
      "$run_status" "$ready_status" "$stop_status" "$task_rm_status" \
      "$container_rm_status" >> "$out"

    if [ "$run_status" -ne 0 ] || [ "$ready_status" -ne 0 ] || \
       [ "$stop_status" -ne 0 ] || \
       [ "$task_rm_status" -ne 0 ] || [ "$container_rm_status" -ne 0 ]; then
      break
    fi
    i=$((i + 1))
  done
}

run_memory() {
  accel=$1
  id=kata-memory-$accel
  image_ref=$(cat /tmp/kata-benchmark-image-ref)

  timeout 180s ctr --address "$ADDR" run --detach \
    --runtime io.containerd.kata.v2 "$image_ref" "$id" /bin/busybox sleep 300 \
    > "/tmp/kata-memory-run-$accel.stdout" 2>&1
  run_status=$?
  running_status=124
  i=0
  while [ "$i" -lt 180 ]; do
    if ctr --address "$ADDR" tasks ls 2>/dev/null \
      | grep -F "$id" | grep -q RUNNING; then
      running_status=0
      break
    fi
    i=$((i + 1))
    sleep 1
  done
  sleep 2

  out=/tmp/kata-memory-rss-$accel.tsv
  printf 'mode\tcomponent\tpid\tvmrss_kib\tname\tcmdline\n' > "$out"
  for proc in /proc/[0-9]*; do
    pid=${proc#/proc/}
    cmdline=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    component=
    case "$cmdline" in
      *qemu-system-x86_64*kata-memory-$accel*) component=qemu ;;
      *virtiofsd*) component=virtiofsd ;;
      *containerd-shim-kata-v2*kata-memory-$accel*) component=shim ;;
    esac
    [ -n "$component" ] || continue
    rss=$(sed -n 's/^VmRSS:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$proc/status")
    name=$(sed -n 's/^Name:[[:space:]]*//p' "$proc/status")
    [ -n "$rss" ] || rss=0
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$accel" "$component" "$pid" "$rss" "$name" "$cmdline" >> "$out"
  done

  qemu_rss=$(awk -F '\t' '$2 == "qemu" { sum += $4 } END { print sum + 0 }' "$out")
  virtiofsd_rss=$(awk -F '\t' '$2 == "virtiofsd" { sum += $4 } END { print sum + 0 }' "$out")
  shim_rss=$(awk -F '\t' '$2 == "shim" { sum += $4 } END { print sum + 0 }' "$out")
  total_rss=$((qemu_rss + virtiofsd_rss + shim_rss))
  printf '%s\tsummary\t-\t%s\tcomponents\tqemu=%s,virtiofsd=%s,shim=%s,run_status=%s,running_status=%s\n' \
    "$accel" "$total_rss" "$qemu_rss" "$virtiofsd_rss" "$shim_rss" \
    "$run_status" "$running_status" >> "$out"

  timeout 10s ctr --address "$ADDR" tasks kill -s SIGKILL "$id" \
    > "/tmp/kata-memory-kill-$accel.stdout" 2>&1 || true
  wait_task_stopped "$id" 30 "/tmp/kata-memory-stop-$accel.stdout"
  cleanup_stop_status=$?
  timeout 30s ctr --address "$ADDR" tasks rm "$id" \
    > "/tmp/kata-memory-task-rm-$accel.stdout" 2>&1
  task_rm_status=$?
  timeout 30s ctr --address "$ADDR" containers rm "$id" \
    > "/tmp/kata-memory-container-rm-$accel.stdout" 2>&1
  container_rm_status=$?
  printf 'mode\trun_status\trunning_status\tcleanup_stop_status\ttask_rm_status\tcontainer_rm_status\n%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$accel" "$run_status" "$running_status" "$cleanup_stop_status" \
    "$task_rm_status" "$container_rm_status" > "/tmp/kata-memory-status-$accel.tsv"
}

[ "$#" -eq 2 ] || usage
benchmark=$1
accel=$2
case "$accel" in
  kvm|tcg) ;;
  *) usage ;;
esac

case "$benchmark" in
  lifecycle) run_lifecycle "$accel" ;;
  memory) run_memory "$accel" ;;
  *) usage ;;
esac
