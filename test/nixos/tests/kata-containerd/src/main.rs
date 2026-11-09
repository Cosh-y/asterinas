// SPDX-License-Identifier: MPL-2.0

//! The test suite for containerd and Kata runtime bring-up on Asterinas NixOS.

use std::fmt::Write;

use nixos_test_framework::*;

nixos_test_main!();

fn install_embedded_script(
    nixos_shell: &mut Session,
    contents: &str,
    guest_path: &str,
) -> Result<(), Error> {
    const CONSOLE_CHUNK_BYTES: usize = 512;

    nixos_shell.run_cmd(&format!("rm -f {guest_path}"))?;
    for chunk in contents.as_bytes().chunks(CONSOLE_CHUNK_BYTES) {
        let mut octal = String::with_capacity(chunk.len() * 4);
        for byte in chunk {
            write!(&mut octal, "\\{byte:03o}").unwrap();
        }
        nixos_shell.run_cmd(&format!("printf '%b' '{octal}' >> {guest_path}"))?;
    }
    nixos_shell.run_cmd_and_expect(
        &format!(
            "chmod 0755 {guest_path}; \
             test -x {guest_path} && echo KATA_EMBEDDED_SCRIPT_READY"
        ),
        "KATA_EMBEDDED_SCRIPT_READY",
    )
}

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
        "test -r /etc/kata-debug/busybox.tar && echo kata-image-tar-ready",
        "kata-image-tar-ready",
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

/// Reset rootfs-probe state and patch a writable copy of the guest scripts with
/// the given container shell command (substituted for `echo ok-from-kata`).
/// The originals under `/etc` are Nix store-backed and read-only.
///
/// `container_cmd` is interpolated into a sed `s|...|...|` replacement. It
/// must contain neither single quotes nor the `|` character — both would
/// terminate the replacement string. Ampersands are escaped before interpolation.
fn prepare_rootfs_probe(nixos_shell: &mut Session, container_cmd: &str) -> Result<(), Error> {
    assert!(
        !container_cmd.contains('\'') && !container_cmd.contains('|'),
        "container_cmd must not contain ' or |: {container_cmd:?}"
    );

    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-debug-run-all.stdout /tmp/kata-run.txt \
         /tmp/kata-rootfs /tmp/kata-debug-guest /tmp/containerd-debug.sock \
         /tmp/containerd-root /tmp/containerd-state",
    )?;

    nixos_shell.run_cmd_and_expect(
        "cp -RL /etc/kata-debug/guest /tmp/kata-debug-guest; \
         test -w /tmp/kata-debug-guest/02-run-rootfs-probe.sh && \
         echo KATA_DEBUG_SCRIPTS_WRITABLE",
        "KATA_DEBUG_SCRIPTS_WRITABLE",
    )?;

    // Normalize bare `sh -c '...'` to `/bin/sh -c '...'` (upstream form).
    nixos_shell.run_cmd(
        "sed -i \"s|  sh -c 'echo ok-from-kata'|  /bin/sh -c 'echo ok-from-kata'|\" \
         /tmp/kata-debug-guest/02-run-rootfs-probe.sh",
    )?;

    // Escape sed's replacement metacharacter, then replace the upstream
    // `/bin/sh -c 'echo ok-from-kata'` with our command.
    let sed_container_cmd = container_cmd.replace('&', "\\&");
    nixos_shell.run_cmd(&format!(
        "sed -i \"s|/bin/sh -c 'echo ok-from-kata'|/bin/sh -c '{sed_container_cmd}'|\" \
         /tmp/kata-debug-guest/02-run-rootfs-probe.sh"
    ))?;

    // Copy the bash binary into rootfs (the upstream `ln -sf` symlink doesn't
    // resolve inside the container's mount view).
    nixos_shell.run_cmd(
        "sed -i 's|ln -sf \"$SH\" \"$ROOTFS/bin/sh\"|cp \"$SH\" \"$ROOTFS/bin/sh\"|' \
         /tmp/kata-debug-guest/02-run-rootfs-probe.sh",
    )?;
    // `--rm` deletes the container before we can read its task state.
    nixos_shell.run_cmd(
        "sed -i 's|run --rootfs --rm|run --rootfs|' \
         /tmp/kata-debug-guest/02-run-rootfs-probe.sh",
    )?;
    nixos_shell.run_cmd(
        "sed -i 's|timeout 120s ctr|timeout 180s ctr|' \
         /tmp/kata-debug-guest/02-run-rootfs-probe.sh",
    )?;

    // PT_INTERP fallback: the rootfs ldd-copy puts the dynamic loader at
    // `lib64/`, but NixOS bash embeds an absolute `/nix/store/<glibc>/lib/...`
    // PT_INTERP path. Mirror `ld-linux*` between `lib/` and `lib64/` inside
    // each glibc store dir before the container starts.
    //
    // Skipped if the source script (`guest-scripts/02-run-rootfs-probe.sh`)
    // already includes the same fix (idempotent via grep).
    nixos_shell.run_cmd(
        r##"if ! grep -q PT_INTERP_DETECTED /tmp/kata-debug-guest/02-run-rootfs-probe.sh; then cat > /tmp/kata-ptinterp-block <<'EOF'
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
awk 'FNR == NR { block = block $0 "\n"; next } /echo "rootfs files:"/ && !done { printf "%s", block; done = 1 } { print }' /tmp/kata-ptinterp-block /tmp/kata-debug-guest/02-run-rootfs-probe.sh > /tmp/02-run-rootfs-probe.sh.new && mv /tmp/02-run-rootfs-probe.sh.new /tmp/kata-debug-guest/02-run-rootfs-probe.sh; fi"##,
    )?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "grep -F -- '{container_cmd}' /tmp/kata-debug-guest/02-run-rootfs-probe.sh; \
             grep -F -- 'timeout 180s ctr' /tmp/kata-debug-guest/02-run-rootfs-probe.sh; \
             echo KATA_ROOTFS_PROBE_PREPARED"
        ),
        "KATA_ROOTFS_PROBE_PREPARED",
    )?;
    Ok(())
}

fn run_kata_debug(
    nixos_shell: &mut Session,
    accel: &str,
    workload: &str,
    extra_rootfs_binaries: Option<&str>,
) -> Result<(), Error> {
    assert!(
        matches!(accel, "kvm" | "tcg"),
        "unsupported Kata QEMU accelerator: {accel}"
    );
    assert!(
        matches!(
            workload,
            "write-visibility" | "exit-propagation" | "fio-shared-rootfs"
        ),
        "unsupported Kata rootfs workload: {workload}"
    );
    let extra_rootfs_env = extra_rootfs_binaries
        .map(|binaries| format!("KATA_ROOTFS_EXTRA_BINARIES={binaries} "))
        .unwrap_or_default();

    nixos_shell.run_cmd(&format!(
        "run_start_ns=$(date +%s%N); \
         {extra_rootfs_env}KATA_QEMU_ACCEL={accel} KATA_QEMU_BOOT_DIAG=1 \
           timeout 240s sh /tmp/kata-debug-guest/10-run-all.sh \
           > /tmp/kata-debug-run-all.stdout 2>&1; \
         driver_status=$?; \
         run_end_ns=$(date +%s%N); \
         e2e_ms=$(( (run_end_ns - run_start_ns) / 1000000 )); \
         container_status=$(sed -n 's/^exit://p' /tmp/kata-run.txt \
           2>/dev/null | tail -n1); \
         test -n \"$container_status\" || container_status=missing; \
         printf 'workload\\tmode\\te2e_ms\\tdriver_status\\tcontainer_status\\n{workload}\\t{accel}\\t%s\\t%s\\t%s\\n' \
           \"$e2e_ms\" \"$driver_status\" \"$container_status\" \
           > /tmp/kata-rootfs-e2e-{workload}-{accel}.tsv; \
         echo run-all-exit:$driver_status >> /tmp/kata-debug-run-all.stdout"
    ))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "tail -160 /tmp/kata-debug-run-all.stdout 2>/dev/null || true; \
             cat /tmp/kata-rootfs-e2e-{workload}-{accel}.tsv; \
             cat /tmp/kata-debug-out/qemu-wrapper-*.mode 2>/dev/null || true; \
             cat /tmp/kata-debug-out/qemu-wrapper-*.effective-argv 2>/dev/null || true"
        ),
        &format!("KATA_QEMU_ACCEL={accel}"),
    )?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "echo KATA_WRAPPER_EVIDENCE_BEGIN; \
             cat /tmp/kata-debug-out/qemu-wrapper-*.mode 2>/dev/null || true; \
             cat /tmp/kata-debug-out/qemu-wrapper-*.effective-argv 2>/dev/null || true; \
             for log in /tmp/kata-debug-out/qemu-wrapper-*.stderr; do \
               test -e \"$log\" || continue; echo --- \"$log\"; cat \"$log\"; \
             done; \
             echo KATA_CONTAINERD_LOG_BEGIN; \
             tail -240 /tmp/kata-debug-out/containerd.log 2>/dev/null || true; \
             echo KATA_WRAPPER_EVIDENCE_END"
        ),
        "KATA_WRAPPER_EVIDENCE_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-debug-out/qemu-wrapper-*.mode 2>/dev/null || true",
        &format!("KATA_QEMU_ACCEL={accel}"),
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-debug-out/qemu-wrapper-*.effective-argv 2>/dev/null || true",
        &format!("q35,accel={accel}"),
    )?;
    nixos_shell.run_cmd_and_expect(
        "if grep -F 'Read-only file system' \
           /tmp/kata-debug-out/qemu-wrapper-*.stderr >/dev/null 2>&1; then \
           echo KATA_READ_ONLY_IMAGE_ERROR; \
         else echo KATA_WRAPPER_STDERR_NO_READ_ONLY_IMAGE_ERROR; fi",
        "KATA_WRAPPER_STDERR_NO_READ_ONLY_IMAGE_ERROR",
    )
}

/// Smoke test: the container runs `/bin/sh -c '...; exit 0'`, writes a file
/// inside its rootfs, and exits 0. Validates the full
/// containerd → shim-kata-v2 → runtime-rs → nested QEMU → kata-agent path
/// plus host vhost-vsock control + data plane.
#[nixos_test]
fn kata_rootfs_run_probe(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_rootfs_write_visibility(nixos_shell, "kvm")
}

#[nixos_test]
fn kata_rootfs_run_probe_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_rootfs_write_visibility(nixos_shell, "tcg")
}

fn run_kata_rootfs_write_visibility(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    prepare_rootfs_probe(nixos_shell, "echo SHRAN > /testout.txt; exit 0")?;
    run_kata_debug(nixos_shell, accel, "write-visibility", None)?;
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
    run_kata_rootfs_exit_code_propagation(nixos_shell, "kvm")
}

#[nixos_test]
fn kata_rootfs_exit_code_propagation_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_rootfs_exit_code_propagation(nixos_shell, "tcg")
}

fn run_kata_rootfs_exit_code_propagation(
    nixos_shell: &mut Session,
    accel: &str,
) -> Result<(), Error> {
    // Multi-step shell using only bash builtins (the rootfs has only bash;
    // `cat` etc. aren't ldd-copied). Writes a file, prints to stdout, exits non-zero.
    prepare_rootfs_probe(
        nixos_shell,
        "echo HI-FROM-CTR > /testout.txt; echo HI-FROM-CTR; exit 7",
    )?;
    run_kata_debug(nixos_shell, accel, "exit-propagation", None)?;
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
#[nixos_test]
fn kata_image_run_busybox(nixos_shell: &mut Session) -> Result<(), Error> {
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-debug-run-image.stdout \
         /tmp/kata-debug-run-image.status /tmp/kata-run.txt /tmp/kata-image-run.txt \
         /tmp/kata-image-import.txt /tmp/containerd-debug.sock /tmp/containerd-root \
         /tmp/containerd-state",
    )?;
    nixos_shell.run_cmd(
        "(timeout 1080s kata-debug-run-image; echo $? > /tmp/kata-debug-run-image.status) \
         2>&1 | tee /tmp/kata-debug-run-image.stdout; \
         echo run-image-exit:$(cat /tmp/kata-debug-run-image.status) \
           >> /tmp/kata-debug-run-image.stdout",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -E '^run-image-exit:' /tmp/kata-debug-run-image.stdout; \
         tail -180 /tmp/kata-debug-run-image.stdout; \
         ls -la /tmp/kata-debug-out/ 2>/dev/null; \
         echo BUSYBOX_DIAG_END",
        "BUSYBOX_DIAG_END",
    )?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-image-run.txt", "HI-IMG")?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-image-run.txt", "exit:7")?;
    Ok(())
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

fn prepare_kata_benchmark_image(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    assert!(
        matches!(accel, "kvm" | "tcg"),
        "unsupported Kata QEMU accelerator: {accel}"
    );

    wait_for_containerd_ready(nixos_shell)?;
    nixos_shell.run_cmd(
        "rm -rf /tmp/kata-debug-out /tmp/kata-benchmark-* \
         /tmp/containerd-debug.sock /tmp/containerd-root /tmp/containerd-state",
    )?;
    nixos_shell.run_cmd_and_expect(
        "kata_kernel=$(find /nix/store -path \
           '*/share/kata-containers/vmlinux-6.12.22-151' -type f -print -quit); \
         test -n \"$kata_kernel\"; \
         cp -L /etc/kata-containers/configuration.toml \
           /tmp/kata-benchmark-configuration.toml; \
         chmod u+w /tmp/kata-benchmark-configuration.toml; \
         sed -i \
           -e \"s|^kernel = .*|kernel = \\\"$kata_kernel\\\"|\" \
           -e 's|^default_memory = .*|default_memory = 128|' \
           -e 's|^default_maxmemory = .*|default_maxmemory = 256|' \
           /tmp/kata-benchmark-configuration.toml; \
         grep -E '^(kernel|image|machine_type|default_vcpus|default_maxvcpus|default_memory|default_maxmemory) = ' \
           /tmp/kata-benchmark-configuration.toml; \
         echo KATA_BENCH_CONFIG_READY",
        "KATA_BENCH_CONFIG_READY",
    )?;
    nixos_shell.run_cmd(&format!(
        "containerd_config=$(tr '\\0' '\\n' < /proc/$(pidof containerd)/cmdline \
           | grep -m1 'containerd-config-checked.toml'); \
         test -r \"$containerd_config\"; \
         systemctl stop containerd.socket containerd >/dev/null 2>&1 || true; \
         mkdir -p /tmp/kata-debug-out /tmp/containerd-root /tmp/containerd-state; \
         KATA_CONF_FILE=/tmp/kata-benchmark-configuration.toml \
         KATA_QEMU_ACCEL={accel} KATA_QEMU_BOOT_DIAG=0 \
         containerd --log-level debug --address /tmp/containerd-debug.sock \
           --root /tmp/containerd-root --state /tmp/containerd-state \
           --config \"$containerd_config\" \
           > /tmp/kata-debug-out/containerd.log 2>&1 & \
         echo $! > /tmp/kata-debug-out/containerd.pid; \
         i=0; while [ \"$i\" -lt 50 ] && [ ! -S /tmp/containerd-debug.sock ]; do \
           i=$((i + 1)); sleep 0.2; \
         done; \
         test -S /tmp/containerd-debug.sock; \
         printf 'containerd config: %s\\nkata config: %s\\n' \
           \"$containerd_config\" /tmp/kata-benchmark-configuration.toml \
           > /tmp/kata-benchmark-containerd-{accel}.stdout"
    ))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-benchmark-containerd-{accel}.stdout; \
             test -S /tmp/containerd-debug.sock && echo KATA_BENCH_CONTAINERD_READY"
        ),
        "KATA_BENCH_CONTAINERD_READY",
    )?;
    nixos_shell.run_cmd(&format!(
        "import_start_ns=$(date +%s%N); \
         timeout 120s ctr --address /tmp/containerd-debug.sock image import \
           /etc/kata-debug/busybox.tar \
           > /tmp/kata-benchmark-import-{accel}.stdout 2>&1; \
         import_status=$?; \
         import_end_ns=$(date +%s%N); \
         image_ref=$(ctr --address /tmp/containerd-debug.sock images ls -q \
           2>/dev/null | grep -m1 busybox || true); \
         printf '%s\\n' \"$image_ref\" > /tmp/kata-benchmark-image-ref; \
         printf 'mode\\timport_ms\\timport_status\\timage\\n{accel}\\t%s\\t%s\\t%s\\n' \
           \"$(( (import_end_ns - import_start_ns) / 1000000 ))\" \
           \"$import_status\" \"$image_ref\" \
           > /tmp/kata-benchmark-import-{accel}.tsv"
    ))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-benchmark-import-{accel}.stdout; \
             cat /tmp/kata-benchmark-import-{accel}.tsv; \
             test -s /tmp/kata-benchmark-image-ref && \
             awk -F '\\t' 'NR == 2 && $3 == 0 {{ print \"KATA_BENCH_IMAGE_READY\" }}' \
               /tmp/kata-benchmark-import-{accel}.tsv"
        ),
        "KATA_BENCH_IMAGE_READY",
    )
}

fn run_kata_lifecycle_bench(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    let iterations = 1;
    prepare_kata_benchmark_image(nixos_shell, accel)?;
    install_embedded_script(
        nixos_shell,
        include_str!("../guest-scripts/12-run-metrics-bench.sh"),
        "/tmp/kata-run-metrics-bench.sh",
    )?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "sed -i 's/while \\[ \"$i\" -le 3 \\]; do/while [ \"$i\" -le {iterations} ]; do/' \
               /tmp/kata-run-metrics-bench.sh; \
             grep -F 'while [ \"$i\" -le {iterations} ]; do' \
               /tmp/kata-run-metrics-bench.sh; \
             echo KATA_LIFECYCLE_ITERATIONS_READY"
        ),
        "KATA_LIFECYCLE_ITERATIONS_READY",
    )?;
    nixos_shell.run_cmd(&format!(
        "sh /tmp/kata-run-metrics-bench.sh lifecycle {accel}"
    ))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-lifecycle-{accel}.tsv; \
             for log in /tmp/kata-lifecycle-*.stdout; do \
               test -s \"$log\" || continue; \
               echo \"--- $log ---\"; cat \"$log\"; \
             done; \
             awk -F '\\t' 'NR > 1 {{ rows++; \
               if ($6 != 0 || $7 != 0 || $8 != 0 || $9 != 0 || $10 != 0) bad++ \
             }} END {{ if (rows == {iterations} && bad == 0) \
               print \"KATA_LIFECYCLE_BENCH_OK\" }}' \
               /tmp/kata-lifecycle-{accel}.tsv"
        ),
        "KATA_LIFECYCLE_BENCH_OK",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-debug-out/qemu-wrapper-*.effective-argv 2>/dev/null || true",
        &format!("q35,accel={accel}"),
    )
}

#[nixos_test]
fn kata_lifecycle_bench_kvm(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_lifecycle_bench(nixos_shell, "kvm")
}

#[nixos_test]
fn kata_lifecycle_bench_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_lifecycle_bench(nixos_shell, "tcg")
}

fn run_kata_memory_rss_bench(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    prepare_kata_benchmark_image(nixos_shell, accel)?;
    install_embedded_script(
        nixos_shell,
        include_str!("../guest-scripts/12-run-metrics-bench.sh"),
        "/tmp/kata-run-metrics-bench.sh",
    )?;
    nixos_shell.run_cmd_and_expect(
        "sed -i 's|\"$id\" /bin/sleep 300|\"$id\" /bin/busybox sleep 300|' \
           /tmp/kata-run-metrics-bench.sh; \
         grep -F '\"$id\" /bin/busybox sleep 300' \
           /tmp/kata-run-metrics-bench.sh; \
         echo KATA_MEMORY_COMMAND_READY",
        "KATA_MEMORY_COMMAND_READY",
    )?;
    nixos_shell.run_cmd(&format!("sh /tmp/kata-run-metrics-bench.sh memory {accel}"))?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "cat /tmp/kata-memory-run-{accel}.stdout; \
             cat /tmp/kata-memory-rss-{accel}.tsv; \
             cat /tmp/kata-memory-status-{accel}.tsv; \
             awk -F '\\t' '$2 == \"qemu\" && $4 > 0 {{ qemu = 1 }} \
               $2 == \"virtiofsd\" && $4 > 0 {{ virtiofsd = 1 }} \
               $2 == \"shim\" && $4 > 0 {{ shim = 1 }} \
               $2 == \"summary\" && $4 > 0 {{ summary = 1 }} \
               END {{ if (qemu && virtiofsd && shim && summary) \
                 print \"KATA_MEMORY_COMPONENTS_OK\" }}' \
               /tmp/kata-memory-rss-{accel}.tsv; \
             awk -F '\\t' 'NR == 2 && $2 == 0 && $3 == 0 && $4 == 0 && \
               $5 == 0 && $6 == 0 {{ print \"KATA_MEMORY_STATUS_OK\" }}' \
               /tmp/kata-memory-status-{accel}.tsv; \
             echo KATA_MEMORY_RSS_OUTPUT_END"
        ),
        "KATA_MEMORY_RSS_OUTPUT_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        &format!(
            "components_ok=$(awk -F '\\t' '$2 == \"qemu\" && $4 > 0 {{ qemu = 1 }} \
               $2 == \"virtiofsd\" && $4 > 0 {{ virtiofsd = 1 }} \
               $2 == \"shim\" && $4 > 0 {{ shim = 1 }} \
               $2 == \"summary\" && $4 > 0 {{ summary = 1 }} \
               END {{ if (qemu && virtiofsd && shim && summary) \
                 print \"yes\" }}' /tmp/kata-memory-rss-{accel}.tsv); \
             status_ok=$(awk -F '\\t' 'NR == 2 && $2 == 0 && $3 == 0 && \
               $4 == 0 && $5 == 0 && $6 == 0 {{ print \"yes\" }}' \
               /tmp/kata-memory-status-{accel}.tsv); \
             test \"$components_ok\" = yes && test \"$status_ok\" = yes && \
               echo KATA_MEMORY_RSS_OK"
        ),
        "KATA_MEMORY_RSS_OK",
    )?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-debug-out/qemu-wrapper-*.effective-argv 2>/dev/null || true",
        &format!("q35,accel={accel}"),
    )
}

#[nixos_test]
fn kata_memory_rss_bench_kvm(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_memory_rss_bench(nixos_shell, "kvm")
}

#[nixos_test]
fn kata_memory_rss_bench_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_memory_rss_bench(nixos_shell, "tcg")
}

fn run_kata_fio_bench(nixos_shell: &mut Session, accel: &str) -> Result<(), Error> {
    let fio_command = "/bin/fio --name=seqwrite --filename=/fio-test \
        --rw=write --bs=1M --size=128M --ioengine=sync --direct=1 --time_based \
        --runtime=10 --ramp_time=1 --numjobs=1 --group_reporting \
        --output-format=json --output=/fio-seqwrite.json && \
        /bin/fio --name=randrw --filename=/fio-test \
        --rw=randrw --rwmixread=75 --bs=4k --size=128M --ioengine=sync --iodepth=1 \
        --direct=1 --time_based --runtime=10 --ramp_time=1 --numjobs=1 \
        --group_reporting --output-format=json --output=/fio-randrw.json && \
        echo seqwrite=0 randrw=0 > /fio-status.txt";
    prepare_rootfs_probe(nixos_shell, fio_command)?;
    run_kata_debug(nixos_shell, accel, "fio-shared-rootfs", Some("fio"))?;
    nixos_shell.run_cmd_and_expect("cat /tmp/kata-run.txt", "exit:0")?;
    nixos_shell.run_cmd_and_expect(
        "cat /tmp/kata-rootfs/fio-status.txt; \
         cat /tmp/kata-rootfs/fio-seqwrite.json; \
         cat /tmp/kata-rootfs/fio-randrw.json; \
         echo KATA_FIO_OUTPUT_END",
        "KATA_FIO_OUTPUT_END",
    )?;
    nixos_shell.run_cmd_and_expect(
        "grep -Eq '\"error\"[[:space:]]*:[[:space:]]*0' \
           /tmp/kata-rootfs/fio-seqwrite.json && \
         grep -Eq '\"error\"[[:space:]]*:[[:space:]]*0' \
           /tmp/kata-rootfs/fio-randrw.json && \
         echo KATA_FIO_BENCH_OK",
        "KATA_FIO_BENCH_OK",
    )
}

#[nixos_test]
fn kata_fio_bench_kvm(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_fio_bench(nixos_shell, "kvm")
}

#[nixos_test]
fn kata_fio_bench_tcg(nixos_shell: &mut Session) -> Result<(), Error> {
    run_kata_fio_bench(nixos_shell, "tcg")
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
