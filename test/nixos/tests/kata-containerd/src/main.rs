// SPDX-License-Identifier: MPL-2.0

//! The test suite for containerd and Kata runtime bring-up on Asterinas NixOS.

use nixos_test_framework::*;

nixos_test_main!();

fn wait_for_containerd_ready(nixos_shell: &mut Session) -> Result<(), Error> {
    nixos_shell.run_cmd(
        "rm -f /tmp/containerd-ready.state /tmp/containerd-ready.log
         i=0
         ready=0
         while [ \"$i\" -lt 90 ]; do
           state=$(systemctl is-active containerd 2>&1 || true)
           if [ \"$state\" = active ] && [ -S /run/containerd/containerd.sock ]; then
             echo ready > /tmp/containerd-ready.state
             ready=1
             break
           fi
           echo \"$state\" >> /tmp/containerd-ready.log
           i=$((i + 1))
           sleep 2
         done
         if [ \"$ready\" -ne 1 ]; then
           echo unavailable > /tmp/containerd-ready.state
           {
             systemctl show containerd \
               -p ActiveState -p SubState -p ExecMainPID -p MainPID -p Result \
               -p NotifyAccess -p Type -p TimeoutStartUSec 2>&1 || true
             systemctl status containerd --no-pager -l 2>&1 || true
             ls -ld /run/containerd /run/containerd/containerd.sock 2>&1 || true
           } >> /tmp/containerd-ready.log
         fi",
    )?;
    nixos_shell.run_cmd_and_expect(
        "if grep -qx ready /tmp/containerd-ready.state; then
           printf 'CONTAINERD%s\n' '_STATE_READY'
         else
           echo CONTAINERD_STATE_UNAVAILABLE
           tail -120 /tmp/containerd-ready.log 2>/dev/null || true
         fi",
        "CONTAINERD_STATE_READY",
    )
}

#[nixos_test]
fn containerd_service_basic(nixos_shell: &mut Session) -> Result<(), Error> {
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd_and_expect("ctr version", "Server:")?;

    nixos_shell.run_cmd(
        "tr '\\0' ' ' < /proc/$(pidof containerd)/cmdline > /tmp/containerd-cmdline.txt",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/containerd-cmdline.txt",
        "containerd-config-checked.toml",
    )?;

    Ok(())
}

#[nixos_test]
fn kata_runtime_registered(nixos_shell: &mut Session) -> Result<(), Error> {
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd_and_expect(
        "command -v containerd-shim-kata-v2",
        "/run/current-system/sw/bin/containerd-shim-kata-v2",
    )?;
    nixos_shell.run_cmd_and_expect(
        "containerd-shim-kata-v2 --version",
        "containerd shim (Rust)",
    )?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-containers/configuration.toml && echo kata-rs-config-ready",
        "kata-rs-config-ready",
    )?;
    nixos_shell.run_cmd_and_expect("command -v kata-debug-run-all", "kata-debug-run-all")?;
    nixos_shell.run_cmd_and_expect("command -v kata-debug-run-image", "kata-debug-run-image")?;
    nixos_shell.run_cmd_and_expect("command -v kata-debug-pull-image", "kata-debug-pull-image")?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-debug/guest/10-run-all.sh && echo kata-debug-ready",
        "kata-debug-ready",
    )?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-debug/guest/20-run-image-probe.sh && echo kata-image-debug-ready",
        "kata-image-debug-ready",
    )?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-debug/guest/21-pull-image.sh && echo kata-pull-debug-ready",
        "kata-pull-debug-ready",
    )?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-debug/guest/30-run-security-demo.sh && echo kata-security-demo-ready",
        "kata-security-demo-ready",
    )?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-debug/busybox.tar && echo kata-image-tar-ready",
        "kata-image-tar-ready",
    )?;
    nixos_shell.run_cmd_and_expect(
        "test -r /etc/kata-debug/security-demo.tar && echo kata-security-demo-tar-ready",
        "kata-security-demo-tar-ready",
    )?;

    nixos_shell.run_cmd(
        "CONFIG=$(tr '\\0' '\\n' < /proc/$(pidof containerd)/cmdline | grep 'containerd-config-checked.toml'); grep -n 'io.containerd.kata.v2' \"$CONFIG\" > /tmp/kata-runtime-config.txt",
    )?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-runtime-config.txt", "io.containerd.kata.v2")?;

    Ok(())
}

/// Captures the early `containerd.service` startup state without waiting for
/// the image-path test to time out. This is useful when containerd remains in
/// `activating` before `/run/containerd/containerd.sock` appears.
#[nixos_test]
fn containerd_hang_diag(nixos_shell: &mut Session) -> Result<(), Error> {
    nixos_shell.run_cmd(
        r#"cat > /tmp/containerd-hang-diag.sh <<'EOF'
#!/bin/sh
set +e

sample() {
    label=$1
    echo "===DIAG_SAMPLE_${label}==="

    echo "--- service show ---"
    timeout 10s systemctl show containerd \
        -p ActiveState -p SubState -p ExecMainPID -p MainPID -p Result \
        -p NotifyAccess -p Type -p TimeoutStartUSec 2>&1 || true

    echo "--- service status ---"
    timeout 10s systemctl status containerd --no-pager -l 2>&1 | head -80 || true

    echo "--- socket state ---"
    ls -ld /run/containerd /run/containerd/containerd.sock 2>&1 || true
    timeout 5s find /run/containerd -maxdepth 2 -ls 2>&1 || true

    echo "--- process list ---"
    ps -e -o pid,ppid,stat,wchan:28,args 2>/dev/null \
        | grep -iE 'containerd|shim|runc|kata' \
        | grep -v grep || true

    echo "--- proc details ---"
    for d in /proc/[0-9]*; do
        c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
        case "$c" in
            *containerd*|*shim*|*runc*)
            p=${d#/proc/}
            echo "--- PID $p : $c"
            grep -E '^State|^PPid' "$d/status" 2>/dev/null || true
            echo "wchan=$(cat "$d/wchan" 2>/dev/null)"
            echo "syscall=$(cat "$d/syscall" 2>/dev/null)"
            echo children:
            cat "$d/task/$p/children" 2>/dev/null || true
            echo fds:
            ls -l "$d/fd" 2>&1 | head -40 || true
                ;;
        esac
    done

    echo "--- journal containerd ---"
    timeout 10s journalctl -u containerd -b --no-pager -n 80 2>&1 || true

}

last=0
for at in 5 30 60 120; do
    sleep $((at - last))
    sample "$at"
    last=$at
done

echo ===DIAG_END===
exit 0
EOF"#,
    )?;
    nixos_shell.run_cmd("chmod +x /tmp/containerd-hang-diag.sh")?;
    nixos_shell.run_cmd_and_expect("/tmp/containerd-hang-diag.sh", "===DIAG_END===")?;
    Ok(())
}

/// Reset rootfs-probe state and patch
/// `/etc/kata-debug/guest/02-run-rootfs-probe.sh` with the given container
/// shell command (substituted for the upstream `echo ok-from-kata` command).
///
/// On the first call, the upstream script is backed up to `.orig`; subsequent
/// calls restore from that backup before patching, so multiple test cases can
/// run sequentially without worrying about prior sed state.
///
/// `container_cmd` is interpolated into a sed `s|...|...|` replacement. It
/// must contain neither single quotes nor the `|` character — both would
/// terminate the replacement string. Use shell-safe constructs only.
fn prepare_rootfs_probe(nixos_shell: &mut Session, container_cmd: &str) -> Result<(), Error> {
    assert!(
        !container_cmd.contains('\'') && !container_cmd.contains('|'),
        "container_cmd must not contain ' or |: {container_cmd:?}"
    );

    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-debug-run-all.stdout /tmp/kata-run.txt \
         /tmp/kata-rootfs /tmp/containerd-debug.sock /tmp/containerd-root /tmp/containerd-state",
    )?;

    // Snapshot the upstream script on first run, then restore from snapshot
    // on every call so each test patches from a known-clean baseline.
    nixos_shell.run_cmd(
        "if [ ! -f /etc/kata-debug/guest/02-run-rootfs-probe.sh.orig ]; then \
           cp /etc/kata-debug/guest/02-run-rootfs-probe.sh \
              /etc/kata-debug/guest/02-run-rootfs-probe.sh.orig; \
         fi; \
         cp /etc/kata-debug/guest/02-run-rootfs-probe.sh.orig \
            /etc/kata-debug/guest/02-run-rootfs-probe.sh",
    )?;

    // Normalize bare `sh -c '...'` to `/bin/sh -c '...'` (upstream form).
    nixos_shell.run_cmd(
        "sed -i \"s|  sh -c 'echo ok-from-kata'|  /bin/sh -c 'echo ok-from-kata'|\" \
         /etc/kata-debug/guest/02-run-rootfs-probe.sh",
    )?;

    // Replace the upstream `/bin/sh -c 'echo ok-from-kata'` with our command.
    nixos_shell.run_cmd(&format!(
        "sed -i \"s|/bin/sh -c 'echo ok-from-kata'|/bin/sh -c '{container_cmd}'|\" \
         /etc/kata-debug/guest/02-run-rootfs-probe.sh"
    ))?;

    // Copy the bash binary into rootfs (the upstream `ln -sf` symlink doesn't
    // resolve inside the container's mount view).
    nixos_shell.run_cmd(
        "sed -i 's|ln -sf \"$SH\" \"$ROOTFS/bin/sh\"|cp \"$SH\" \"$ROOTFS/bin/sh\"|' \
         /etc/kata-debug/guest/02-run-rootfs-probe.sh",
    )?;
    // `--rm` deletes the container before we can read its task state.
    nixos_shell.run_cmd(
        "sed -i 's|run --rootfs --rm|run --rootfs|' /etc/kata-debug/guest/02-run-rootfs-probe.sh",
    )?;
    nixos_shell.run_cmd(
        "sed -i 's|timeout 120s ctr|timeout 80s ctr|' /etc/kata-debug/guest/02-run-rootfs-probe.sh",
    )?;

    // PT_INTERP fallback: the rootfs ldd-copy puts the dynamic loader at
    // `lib64/`, but NixOS bash embeds an absolute `/nix/store/<glibc>/lib/...`
    // PT_INTERP path. Mirror `ld-linux*` between `lib/` and `lib64/` inside
    // each glibc store dir before the container starts.
    //
    // Skipped if the source script (`guest-scripts/02-run-rootfs-probe.sh`)
    // already includes the same fix (idempotent via grep).
    nixos_shell.run_cmd(
        r##"if ! grep -q PT_INTERP_DETECTED /etc/kata-debug/guest/02-run-rootfs-probe.sh; then cat > /tmp/kata-ptinterp-block <<'EOF'
# PT_INTERP detection (no readelf on minimal NixOS) + lib/lib64 mirror.
INTERP=$(LC_ALL=C tr '\0' '\n' < "$SH" 2>/dev/null | grep -m1 '/ld-linux' || true)
echo "PT_INTERP_DETECTED=$INTERP" | tee -a "$OUT_DIR/probe.log"
if [ -n "$INTERP" ]; then
  mkdir -p "$ROOTFS$(dirname "$INTERP")"
  if [ -e "$INTERP" ] && [ ! -e "$ROOTFS$INTERP" ]; then
    cp "$INTERP" "$ROOTFS$INTERP"
  fi
fi
for d in "$ROOTFS"/nix/store/*-glibc-*/; do
  [ -d "$d" ] || continue
  mkdir -p "$d"lib "$d"lib64
  for f in "$d"lib64/ld-linux* "$d"lib/ld-linux*; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    [ -e "$d"lib/"$base" ] || cp "$f" "$d"lib/"$base" 2>/dev/null || true
    [ -e "$d"lib64/"$base" ] || cp "$f" "$d"lib64/"$base" 2>/dev/null || true
  done
done
echo "rootfs ld-linux:" | tee -a "$OUT_DIR/probe.log"
find "$ROOTFS" -name 'ld-linux*' -ls | tee -a "$OUT_DIR/probe.log"
EOF
awk 'FNR == NR { block = block $0 "\n"; next } /echo "rootfs files:"/ && !done { printf "%s", block; done = 1 } { print }' /tmp/kata-ptinterp-block /etc/kata-debug/guest/02-run-rootfs-probe.sh > /tmp/02-run-rootfs-probe.sh.new && mv /tmp/02-run-rootfs-probe.sh.new /etc/kata-debug/guest/02-run-rootfs-probe.sh; fi"##,
    )?;
    Ok(())
}

fn run_kata_debug(nixos_shell: &mut Session) -> Result<(), Error> {
    nixos_shell.run_cmd(
        "timeout 180s kata-debug-run-all > /tmp/kata-debug-run-all.stdout 2>&1; \
         echo run-all-exit:$? >> /tmp/kata-debug-run-all.stdout",
    )
}

/// Smoke test: the container runs `/bin/sh -c '...; exit 0'`, writes a file
/// inside its rootfs, and exits 0. Validates the full
/// containerd → shim-kata-v2 → runtime-rs → nested QEMU → kata-agent path
/// plus host vhost-vsock control + data plane.
#[nixos_test]
fn kata_rootfs_run_probe(nixos_shell: &mut Session) -> Result<(), Error> {
    prepare_rootfs_probe(nixos_shell, "echo SHRAN > /testout.txt; exit 0")?;
    run_kata_debug(nixos_shell)?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-run.txt", "exit:0")?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-rootfs/testout.txt", "SHRAN")?;
    Ok(())
}

/// Validates that a container exit code other than zero propagates back
/// through ttrpc → kata-agent → runtime-rs → shim → ctr, and that the
/// container's stdout is captured into ctr's output stream. Exercises the
/// vhost-vsock data plane in both directions with a non-trivial payload.
#[nixos_test]
fn kata_rootfs_exit_code_propagation(nixos_shell: &mut Session) -> Result<(), Error> {
    // Multi-step shell using only bash builtins (the rootfs has only bash;
    // `cat` etc. aren't ldd-copied). Writes a file, prints to stdout, exits non-zero.
    prepare_rootfs_probe(
        nixos_shell,
        "echo HI-FROM-CTR > /testout.txt; echo HI-FROM-CTR; exit 7",
    )?;
    run_kata_debug(nixos_shell)?;
    // ctr propagates the container's exit code as its own; the script
    // appends `exit:$STATUS` to /tmp/kata-run.txt.
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-run.txt", "exit:7")?;
    // ctr captures container stdout into the same file.
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-run.txt", "HI-FROM-CTR")?;
    // The file write inside the container's rootfs is also visible on the host.
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-rootfs/testout.txt", "HI-FROM-CTR")?;
    Ok(())
}

/// Validates the real container image path: import an offline busybox image into
/// containerd, let containerd prepare the snapshot rootfs, then run it through
/// the Kata runtime without `--rootfs`.
fn run_kata_image_run_busybox(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    assert!(matches!(accel, "kvm" | "tcg"));
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-debug-run-image.stdout \
         /tmp/kata-debug-run-image.status /tmp/kata-run.txt /tmp/kata-image-run.txt \
         /tmp/kata-image-import.txt /tmp/kata-image-run-status.txt \
         /tmp/kata-image-run.pid /tmp/containerd-debug.sock /tmp/containerd-root \
         /tmp/containerd-state",
    )?;
    nixos_shell.run_cmd(&format!(
        "(KATA_QEMU_ACCEL={accel} KATA_QEMU_BOOT_DIAG=1 \
           KATA_IMAGE_RUN_TIMEOUT=120s KATA_IMAGE_LIVE_INTERVAL=5 \
           KATA_IMAGE_LIVE_SAMPLES=40 \
           KATA_DEBUG_COLLECT_TIMEOUT=45s \
           timeout 300s kata-debug-run-image; \
         echo $? > /tmp/kata-debug-run-image.status) \
         2>&1 | tee /tmp/kata-debug-run-image.stdout; \
         echo run-image-exit:$(cat /tmp/kata-debug-run-image.status) \
           >> /tmp/kata-debug-run-image.stdout"
    ))?;
    nixos_shell.run_cmd_and_expect(
        "grep -E '^run-image-exit:' /tmp/kata-debug-run-image.stdout; \
         ls -la /tmp/kata-debug-out/ 2>/dev/null; \
         echo BUSYBOX_DIAG_END",
        "BUSYBOX_DIAG_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "for log in /tmp/kata-debug-out/qemu-wrapper-*.mode \
                    /tmp/kata-debug-out/qemu-wrapper-*.effective-argv; do \
           test -e \"$log\" || continue; \
           echo --- \"$log\"; cat \"$log\"; \
         done; \
         for log in /tmp/kata-debug-out/qemu-wrapper-*.boot-serial.log; do \
           test -e \"$log\" || continue; \
           echo --- \"$log\"; \
           grep -a -E 'Linux version|EXT4-fs|kata-agent|ttRPC|virtio|panic|BUG:|Call Trace|VFS:' \"$log\" \
             | tail -120 || true; \
         done; \
         for log in /tmp/kata-debug-out/qemu-wrapper-*.qemu-debug.log; do \
           test -e \"$log\" || continue; \
           echo --- \"$log\"; tail -120 \"$log\"; \
         done; \
         echo BUSYBOX_DIAG_FILES_END",
        "BUSYBOX_DIAG_FILES_END",
    )?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-image-run.txt", "HI-IMG")?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-image-run.txt", "exit:7")?;
    Ok(())
}

#[nixos_test]
fn kata_image_run_busybox(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_image_run_busybox(nixos_shell, "kvm")
}

/// TCG-only diagnostic for the same image/runtime path. This is not a final
/// Kata result because the experiment requires inner KVM.
#[nixos_test]
fn kata_image_run_busybox_tcg_diagnostic(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_image_run_busybox(nixos_shell, "tcg")
}

/// Runs the same untrusted Agent task fixture used by the Docker comparison.
/// The fixture records its task identity and probes paths that must not expose
/// the Asterinas Host. It intentionally does not claim a kernel compromise;
/// a real Guest-kernel CVE can later replace the payload inside the image.
#[nixos_test]
fn kata_agent_security_demo_fixture(nixos_shell: &mut Session) -> Result<(), Error> {
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-security-demo.stdout \
         /run/security-demo-host-canary; \
         timeout 420s kata-debug-run-security-demo \
           > /tmp/kata-security-demo.stdout 2>&1; \
         echo security-demo-exit:$? >> /tmp/kata-security-demo.stdout",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-security-demo.stdout; \
         cat /tmp/kata-debug-out/security-demo.log 2>/dev/null || true; \
         echo SECURITY_DEMO_OUTPUT_END",
        "SECURITY_DEMO_OUTPUT_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'SECURITY_DEMO_RESULT=pass' /tmp/kata-debug-out/security-demo.log",
        "SECURITY_DEMO_RESULT=pass",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'TASK_UID=1000' /tmp/kata-debug-out/security-demo.log",
        "TASK_UID=1000",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'ATTACK_FIXTURE=task-boundary-probe-only' \
           /tmp/kata-debug-out/security-demo.log",
        "ATTACK_FIXTURE=task-boundary-probe-only",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'BOUNDARY_BLOCKED path=/run/security-demo-host-canary' \
           /tmp/kata-debug-out/security-demo.log",
        "BOUNDARY_BLOCKED path=/run/security-demo-host-canary",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'SECURITY_DEMO_RECOVERY=pass' /tmp/kata-debug-out/security-demo.log",
        "SECURITY_DEMO_RECOVERY=pass",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'RECOVERY_TASK=normal-agent-task' \
           /tmp/kata-debug-out/security-demo.log",
        "RECOVERY_TASK=normal-agent-task",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -qx 'host-canary-original' /run/security-demo-host-canary && \
           echo SECURITY_DEMO_HOST_CANARY_OK",
        "SECURITY_DEMO_HOST_CANARY_OK",
    )?;
    Ok(())
}

/// Runs Dirty Pipe in a vulnerable Kata Guest kernel and checks that the
/// Asterinas Host boundary and the next Agent task remain healthy.
fn run_kata_agent_security_demo_cve_2022_0847(
    nixos_shell: &mut Session,
    accel: &str,
) -> Result<(), Error> {
    assert!(matches!(accel, "kvm" | "tcg"));
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd(&format!(
        "rm -rf /tmp/kata-debug-out /tmp/containerd-debug.sock \
         /tmp/containerd-root /tmp/containerd-state; \
         KATA_QEMU_ACCEL={accel} KATA_QEMU_BOOT_DIAG=1 \
           kata-debug-start-containerd \
           > /tmp/kata-security-demo-cve-containerd.stdout 2>&1"
    ))?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-security-demo-cve-containerd.stdout; \
         test -S /tmp/containerd-debug.sock && echo SECURITY_DEMO_DEBUG_CONTAINERD_READY",
        "SECURITY_DEMO_DEBUG_CONTAINERD_READY",
    )?;
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-security-demo-cve.stdout \
         /run/security-demo-host-canary; \
         SECURITY_DEMO_ATTACK_MODE=cve-2022-0847 \
           timeout 420s kata-debug-run-security-demo \
           > /tmp/kata-security-demo-cve.stdout 2>&1; \
         echo security-demo-cve-exit:$? >> /tmp/kata-security-demo-cve.stdout",
    )?;
    nixos_shell
        .run_cmd("kata-debug-collect > /tmp/kata-security-demo-cve-collect.stdout 2>&1 || true")?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-security-demo-cve.stdout; \
         cat /tmp/kata-debug-out/security-demo.log 2>/dev/null || true; \
         cat /tmp/kata-security-demo-cve-collect.stdout; \
         echo SECURITY_DEMO_CVE_OUTPUT_END",
        "SECURITY_DEMO_CVE_OUTPUT_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'SECURITY_DEMO_RESULT=pass' \
           /tmp/kata-debug-out/security-demo.log",
        "SECURITY_DEMO_RESULT=pass",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'TASK_KERNEL=Linux 5.15.24' \
           /tmp/kata-debug-out/security-demo.log",
        "TASK_KERNEL=Linux 5.15.24",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'CVE_PAYLOAD_EXIT=0' /tmp/kata-debug-out/security-demo.log",
        "CVE_PAYLOAD_EXIT=0",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'CVE_TARGET_CONTENT=gXest-canary-original ' \
           /tmp/kata-debug-out/security-demo.log",
        "CVE_TARGET_CONTENT=gXest-canary-original ",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'CVE_REPRODUCTION=success' \
           /tmp/kata-debug-out/security-demo.log",
        "CVE_REPRODUCTION=success",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'BOUNDARY_BLOCKED path=/run/security-demo-host-canary' \
           /tmp/kata-debug-out/security-demo.log",
        "BOUNDARY_BLOCKED path=/run/security-demo-host-canary",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'SECURITY_DEMO_RECOVERY=pass' \
           /tmp/kata-debug-out/security-demo.log",
        "SECURITY_DEMO_RECOVERY=pass",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -qx 'host-canary-original' /run/security-demo-host-canary && \
           echo SECURITY_DEMO_HOST_CANARY_OK",
        "SECURITY_DEMO_HOST_CANARY_OK",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-debug-out/qemu-wrapper-*.mode 2>/dev/null || true",
        &format!("KATA_QEMU_ACCEL={accel}"),
    )?;
    Ok(())
}

#[nixos_test]
fn kata_agent_security_demo_cve_2022_0847(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_agent_security_demo_cve_2022_0847(nixos_shell, "kvm")
}

/// Diagnostic variant: bind a read-only file from the Kata Guest root ext4
/// into the same Agent task. The image's /opt/guest-canary is virtiofs-backed,
/// so it is not a valid Dirty Pipe target for proving the Guest ext4 lane.
#[nixos_test]
fn kata_agent_security_demo_cve_guest_ext4_bind(nixos_shell: &mut Session) -> Result<(), Error> {
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/containerd-debug.sock \
         /tmp/containerd-root /tmp/containerd-state; \
         KATA_QEMU_ACCEL=kvm KATA_QEMU_BOOT_DIAG=1 \
           kata-debug-start-containerd \
           > /tmp/kata-security-demo-bind-containerd.stdout 2>&1",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-security-demo-bind-containerd.stdout; \
         test -S /tmp/containerd-debug.sock && echo GUEST_EXT4_BIND_CONTAINERD_READY",
        "GUEST_EXT4_BIND_CONTAINERD_READY",
    )?;
    nixos_shell.run_cmd_and_expect(
        "rm -f /tmp/kata-security-demo-bind.stdout; \
         ctr --address /tmp/containerd-debug.sock image import \
           /etc/kata-debug/security-demo.tar >/tmp/kata-security-demo-bind-import.stdout 2>&1; \
         timeout 300s ctr --debug --address /tmp/containerd-debug.sock run \
           --runtime io.containerd.kata.v2 --user 1000:1000 \
           --mount type=bind,src=/etc/passwd,dst=/opt/guest-canary,options=rbind:ro \
           --env SECURITY_DEMO_ATTACK_MODE=cve-2022-0847 \
           --env CVE_PAYLOAD_TARGET=/opt/guest-canary \
           docker.io/library/agent-security-demo:fixture-v1 \
           agent-security-demo-guest-ext4-bind /opt/agent-task.sh \
           >/tmp/kata-security-demo-bind.stdout 2>&1; \
         echo bind-run-exit:$? >> /tmp/kata-security-demo-bind.stdout; \
         cat /tmp/kata-security-demo-bind.stdout; \
         echo GUEST_EXT4_BIND_OUTPUT_END",
        "GUEST_EXT4_BIND_OUTPUT_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'CVE_PAYLOAD_EXIT=0' /tmp/kata-security-demo-bind.stdout",
        "CVE_PAYLOAD_EXIT=0",
    )?;
    Ok(())
}

#[nixos_test]
fn kata_agent_security_demo_cve_2022_0847_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_agent_security_demo_cve_2022_0847(nixos_shell, "tcg")
}

fn run_kata_image_launch_bench(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    assert!(
        matches!(accel, "kvm" | "tcg"),
        "unsupported Kata QEMU accelerator: {accel}"
    );
    let launch_timeout_secs = 180;

    // The debug launcher discovers its config from the system containerd process.
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-image-launch-* \
         /tmp/containerd-debug.sock /tmp/containerd-root /tmp/containerd-state",
    )?;
    nixos_shell.run_cmd(&format!(
        "KATA_QEMU_ACCEL={accel} KATA_QEMU_BOOT_DIAG=1 \
         kata-debug-start-containerd \
         > /tmp/kata-image-launch-containerd-{accel}.stdout 2>&1"
    ))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-image-launch-containerd-{accel}.stdout; \
             test -S /tmp/containerd-debug.sock && echo KATA_BENCH_CONTAINERD_READY"
        ),
        "KATA_BENCH_CONTAINERD_READY",
    )?;
    nixos_shell.run_cmd(&format!(
        "import_start_ns=$(date +%s%N); \
         timeout 120s ctr --address /tmp/containerd-debug.sock image import \
           /etc/kata-debug/busybox.tar \
           > /tmp/kata-image-launch-import-{accel}.stdout 2>&1; \
         import_status=$?; \
         import_end_ns=$(date +%s%N); \
         image_ref=$(ctr --address /tmp/containerd-debug.sock images ls -q \
           2>/dev/null | grep -m1 busybox || true); \
         test -n \"$image_ref\" || image_ref=busybox:latest; \
         launch_start_ns=$(date +%s%N); \
         ctr --debug --address /tmp/containerd-debug.sock run \
           --detach --runtime io.containerd.kata.v2 \
           \"$image_ref\" kata-image-launch-{accel} \
           /bin/sh -c 'echo ready > /kata-bench-ready' \
           > /tmp/kata-image-launch-run-{accel}.stdout 2>&1 & \
         ctr_pid=$!; \
         launch_status=124; \
         launch_deadline_s=$(( $(date +%s) + {launch_timeout_secs} )); \
         while [ \"$(date +%s)\" -lt \"$launch_deadline_s\" ]; do \
           ready_marker=$(find /tmp/containerd-root \
             -path '*/snapshots/*/fs/kata-bench-ready' \
             -type f -print -quit 2>/dev/null); \
           if [ -n \"$ready_marker\" ]; then \
             launch_status=0; \
             break; \
           fi; \
           if ! kill -0 \"$ctr_pid\" 2>/dev/null; then \
             wait \"$ctr_pid\"; \
             launch_status=$?; \
             break; \
           fi; \
           sleep 0.2; \
         done; \
         launch_end_ns=$(date +%s%N); \
         kill -KILL \"$ctr_pid\" >/dev/null 2>&1 || true; \
         import_ms=$(( (import_end_ns - import_start_ns) / 1000000 )); \
         launch_ms=$(( (launch_end_ns - launch_start_ns) / 1000000 )); \
         printf 'mode\\timport_ms\\timport_status\\tlaunch_ms\\tlaunch_status\\n{accel}\\t%s\\t%s\\t%s\\t%s\\n' \
           \"$import_ms\" \"$import_status\" \"$launch_ms\" \"$launch_status\" \
           > /tmp/kata-image-launch-{accel}.tsv"
    ))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-image-launch-{accel}.tsv; \
             tail -80 /tmp/kata-image-launch-containerd-{accel}.stdout; \
             tail -120 /tmp/kata-image-launch-run-{accel}.stdout; \
             find /tmp/containerd-root -name kata-bench-ready -print \
               2>/dev/null || true; \
             for log in /tmp/kata-debug-out/qemu-wrapper-*.stderr \
                        /tmp/kata-debug-out/qemu-wrapper-*.boot-serial.log; do \
               test -e \"$log\" || continue; \
               echo --- \"$log\"; \
               tail -160 \"$log\"; \
             done; \
             for log in /tmp/kata-debug-out/qemu-wrapper-*.effective-argv \
                        /tmp/kata-debug-out/qemu-wrapper-*.qemu-debug.log; do \
               test -e \"$log\" || continue; \
               echo --- \"$log\"; \
               tail -240 \"$log\"; \
             done; \
             tail -160 /tmp/kata-debug-out/containerd.log 2>/dev/null || true; \
             echo KATA_IMAGE_LAUNCH_BENCH_OUTPUT_END"
        ),
        "KATA_IMAGE_LAUNCH_BENCH_OUTPUT_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "awk -F '\\t' 'NR == 2 && $1 == \"{accel}\" && $3 == 0 && $5 == 0 \
             {{ print \"KATA_IMAGE_LAUNCH_BENCH_OK\" }}' \
             /tmp/kata-image-launch-{accel}.tsv"
        ),
        "KATA_IMAGE_LAUNCH_BENCH_OK",
    )?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-debug-out/qemu-wrapper-*.mode 2>/dev/null || true; \
             cat /tmp/kata-debug-out/qemu-wrapper-*.effective-argv 2>/dev/null || true"
        ),
        &format!("KATA_QEMU_ACCEL={accel}"),
    )?;
    Ok(())
}

#[nixos_test]
fn kata_image_launch_bench_kvm(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_image_launch_bench(nixos_shell, "kvm")
}

#[nixos_test]
fn kata_image_launch_bench_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_image_launch_bench(nixos_shell, "tcg")
}

/// Validates the online registry path: pull busybox through the guest network,
/// then run the pulled image through the Kata runtime without `--rootfs`.
#[nixos_test]
fn kata_image_pull_busybox(nixos_shell: &mut Session) -> Result<(), Error> {
    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd(
        "rm -f /tmp/kata-image-pull.log /tmp/kata-image-pull-run.txt \
         /tmp/kata-image-pull-step.out",
    )?;
    nixos_shell.run_cmd(
        "timeout 1500s kata-debug-pull-image; \
         echo pull-script-exit:$? | tee -a /tmp/kata-image-pull.log",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -E '^=== (pull|run) |^pull_rc:|^PULL_RESULT=|^pull-script-exit:|^HI-PULL$|^exit:' \
           /tmp/kata-image-pull.log || true; \
         echo PULL_DIAG_END",
        "PULL_DIAG_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -E '^PULL_RESULT=ok( |$)' /tmp/kata-image-pull.log",
        "PULL_RESULT=ok",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -x 'pull-script-exit:0' /tmp/kata-image-pull.log",
        "pull-script-exit:0",
    )?;
    nixos_shell.run_cmd_and_expect("grep -x 'HI-PULL' /tmp/kata-image-pull-run.txt", "HI-PULL")?;
    nixos_shell.run_cmd_and_expect("grep -x 'exit:7' /tmp/kata-image-pull-run.txt", "exit:7")?;
    Ok(())
}
