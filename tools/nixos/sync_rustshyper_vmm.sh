#!/bin/bash

# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
ASTERINAS_DIR=$(realpath "${SCRIPT_DIR}/../..")
WORKSPACE_DIR=$(realpath "${ASTERINAS_DIR}/..")

ASTERINAS_IMG=${ASTERINAS_IMG:-"${ASTERINAS_DIR}/target/nixos/asterinas.img"}
RUSTSHYPER_VMM_DIR=${RUSTSHYPER_VMM_DIR:-"${WORKSPACE_DIR}/rustshyper-vmm"}
RUSTSHYPER_VMM_TARGET=${RUSTSHYPER_VMM_TARGET:-x86_64-unknown-linux-musl}
RUSTSHYPER_VMM_BIN=${RUSTSHYPER_VMM_BIN:-"${RUSTSHYPER_VMM_DIR}/target/${RUSTSHYPER_VMM_TARGET}/release/rustshyper-vmm"}
BZIMAGE_PATH=${BZIMAGE_PATH:-"${WORKSPACE_DIR}/bzImage"}
INITRAMFS_PATH=${INITRAMFS_PATH:-"${WORKSPACE_DIR}/initramfs.cpio.gz"}
MOUNT_DIR=${MOUNT_DIR:-/mnt/asterinas-root}

build_rustshyper_vmm() {
    echo "Building rustshyper-vmm from ${RUSTSHYPER_VMM_DIR} for ${RUSTSHYPER_VMM_TARGET}..."

    if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
        local sudo_home
        sudo_home=$(getent passwd "${SUDO_USER}" | cut -d: -f6)
        sudo -H -u "${SUDO_USER}" env \
            PATH="${sudo_home}/.cargo/bin:${PATH}" \
            bash -lc "cd \"${RUSTSHYPER_VMM_DIR}\" && cargo build --release --target \"${RUSTSHYPER_VMM_TARGET}\""
    else
        (
            cd "${RUSTSHYPER_VMM_DIR}"
            cargo build --release --target "${RUSTSHYPER_VMM_TARGET}"
        )
    fi
}

usage() {
    cat <<EOF
Usage: $0

Environment variables:
  ASTERINAS_IMG         Path to asterinas.img
  RUSTSHYPER_VMM_DIR    Path to rustshyper-vmm source tree
  RUSTSHYPER_VMM_TARGET Cargo target used to build rustshyper-vmm
  RUSTSHYPER_VMM_BIN    Path to a prebuilt rustshyper-vmm binary
  RUSTSHYPER_VMM_SKIP_BUILD
                         Set to 1 to skip the default incremental VMM build
  BZIMAGE_PATH          Path to the Linux bzImage to sync into /root/bzImage
  INITRAMFS_PATH        Path to the Linux initramfs to sync into /root/initramfs.cpio.gz
  MOUNT_DIR             Temporary mount point for the root partition
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ ! -f "${ASTERINAS_IMG}" ]; then
    echo "Error: Asterinas image not found at ${ASTERINAS_IMG}" >&2
    exit 1
fi

if [ "${RUSTSHYPER_VMM_SKIP_BUILD:-0}" != "1" ]; then
    if [ ! -f "${RUSTSHYPER_VMM_DIR}/Cargo.toml" ]; then
        echo "Error: rustshyper-vmm source tree not found at ${RUSTSHYPER_VMM_DIR}" >&2
        exit 1
    fi

    build_rustshyper_vmm
fi

if [ ! -f "${RUSTSHYPER_VMM_BIN}" ]; then
    echo "Error: rustshyper-vmm binary not found at ${RUSTSHYPER_VMM_BIN}" >&2
    exit 1
fi

if [ ! -f "${BZIMAGE_PATH}" ]; then
    echo "Error: bzImage not found at ${BZIMAGE_PATH}" >&2
    exit 1
fi

if [ ! -f "${INITRAMFS_PATH}" ]; then
    echo "Error: initramfs not found at ${INITRAMFS_PATH}" >&2
    exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

LOOP_DEV=""
ROOT_PART=""
MOUNTED=0

cleanup() {
    if [ "${MOUNTED}" -eq 1 ]; then
        ${SUDO} umount "${MOUNT_DIR}" 2>/dev/null || true
    fi
    if [ -n "${LOOP_DEV}" ]; then
        ${SUDO} losetup -d "${LOOP_DEV}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM ERR

echo "Attaching ${ASTERINAS_IMG}..."
LOOP_DEV=$(${SUDO} losetup -fP --show "${ASTERINAS_IMG}")
ROOT_PART="${LOOP_DEV}p2"

for _ in $(seq 1 20); do
    if [ -b "${ROOT_PART}" ]; then
        break
    fi
    sleep 0.1
done

if [ ! -b "${ROOT_PART}" ]; then
    echo "Error: root partition ${ROOT_PART} not found" >&2
    exit 1
fi

${SUDO} mkdir -p "${MOUNT_DIR}"
echo "Mounting ${ROOT_PART} at ${MOUNT_DIR}..."
${SUDO} mount "${ROOT_PART}" "${MOUNT_DIR}"
MOUNTED=1

${SUDO} mkdir -p "${MOUNT_DIR}/root" "${MOUNT_DIR}/usr/local/bin"
${SUDO} cp "${RUSTSHYPER_VMM_BIN}" "${MOUNT_DIR}/root/rustshyper-vmm"
${SUDO} cp "${RUSTSHYPER_VMM_BIN}" "${MOUNT_DIR}/usr/local/bin/rustshyper-vmm"
${SUDO} cp "${BZIMAGE_PATH}" "${MOUNT_DIR}/root/bzImage"
${SUDO} cp "${INITRAMFS_PATH}" "${MOUNT_DIR}/root/initramfs.cpio.gz"
${SUDO} chmod 755 "${MOUNT_DIR}/root/rustshyper-vmm" "${MOUNT_DIR}/usr/local/bin/rustshyper-vmm"
${SUDO} sync

echo "Synced rustshyper-vmm and guest boot assets into ${ASTERINAS_IMG}:"
echo "  /root/rustshyper-vmm"
echo "  /usr/local/bin/rustshyper-vmm"
echo "  /root/bzImage"
echo "  /root/initramfs.cpio.gz"
