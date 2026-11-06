// SPDX-License-Identifier: MPL-2.0

use ostd::arch::cpu::context::FpuContext;

use super::ioctl::KvmFpu;
use crate::prelude::*;

pub(super) struct VcpuFpu {
    pub guest: FpuContext,
    pub host: FpuContext,
}

impl VcpuFpu {
    pub(super) fn new() -> Self {
        Self {
            guest: FpuContext::new(),
            host: FpuContext::new(),
        }
    }

    pub(super) fn get(&self) -> KvmFpu {
        let registers = FxSaveRegisters::from_first_bytes(self.guest.as_bytes());
        KvmFpu {
            fpr: registers.fpr,
            fcw: registers.fcw,
            fsw: registers.fsw,
            ftwx: registers.ftwx,
            last_opcode: registers.last_opcode,
            last_ip: registers.last_ip,
            last_dp: registers.last_dp,
            xmm: registers.xmm,
            // Linux leaves `mxcsr` zeroed in `KVM_GET_FPU`.
            // See <https://elixir.bootlin.com/linux/v6.8/source/arch/x86/kvm/x86.c#L11244>.
            ..KvmFpu::default()
        }
    }

    pub(super) fn set(&mut self, fpu: &KvmFpu) {
        let bytes = self.guest.as_bytes_mut();
        let registers = FxSaveRegisters {
            fcw: fpu.fcw,
            fsw: fpu.fsw,
            ftwx: fpu.ftwx,
            last_opcode: fpu.last_opcode,
            last_ip: fpu.last_ip,
            last_dp: fpu.last_dp,
            fpr: fpu.fpr,
            xmm: fpu.xmm,
            // `KVM_SET_FPU` also ignores `mxcsr`; preserve it and `mxcsr_mask`.
            ..FxSaveRegisters::from_first_bytes(bytes)
        };
        bytes[..size_of::<FxSaveRegisters>()].copy_from_slice(registers.as_bytes());

        const XSAVE_HEADER_OFFSET: usize = 512;
        const XFEATURE_X87: u64 = 1 << 0;
        const XFEATURE_SSE: u64 = 1 << 1;

        if bytes.len() > XSAVE_HEADER_OFFSET {
            // Otherwise `XRSTOR` may initialize these components instead of loading them.
            let xstate_bv =
                u64::from_first_bytes(&bytes[XSAVE_HEADER_OFFSET..]) | XFEATURE_X87 | XFEATURE_SSE;
            bytes[XSAVE_HEADER_OFFSET..XSAVE_HEADER_OFFSET + size_of::<u64>()]
                .copy_from_slice(xstate_bv.as_bytes());
        }
    }
}

/// The register prefix shared by the `FXSAVE64` and `XSAVE64` formats.
#[repr(C)]
#[derive(Clone, Copy, Pod)]
struct FxSaveRegisters {
    fcw: u16,
    fsw: u16,
    ftwx: u8,
    reserved: u8,
    last_opcode: u16,
    last_ip: u64,
    last_dp: u64,
    mxcsr: u32,
    mxcsr_mask: u32,
    fpr: [[u8; 16]; 8],
    xmm: [[u8; 16]; 16],
}
