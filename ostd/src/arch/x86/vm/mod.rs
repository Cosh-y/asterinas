// SPDX-License-Identifier: MPL-2.0

//! Provides Intel VMX-based guest virtualization support.
//!
//! This module contains the x86-specific guest CPU model, VMX/VMCS helpers,
//! EPT support, VM-exit decoding, and low-level instruction emulation used by
//! OSTD's guest virtualization layer.
//!
//! Public exports are limited to the vCPU state and exit types that the
//! kernel-side KVM-compatible device needs. VMX implementation details remain
//! crate-private.

pub(crate) mod context;
pub(crate) mod control_regs;
mod cpuid;
mod emulate;
pub(crate) mod ept;
pub(crate) mod exit;
pub(crate) mod interrupt;
mod types;
pub(crate) mod vmcs;
pub(crate) mod vmx;
pub(crate) mod x86;

#[cfg(ktest)]
mod tests;

pub use self::{
    context::GuestContext,
    cpuid::{GuestCpuidEntry, default_cpuid_entries},
    exit::GuestExitInfo,
    types::{VcpuDtable, VcpuRegs, VcpuSegment, VcpuSregs},
    vmx::VmxExitReason,
};
