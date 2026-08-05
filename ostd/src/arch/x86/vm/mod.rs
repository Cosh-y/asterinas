// SPDX-License-Identifier: MPL-2.0

//! Provides Intel VMX-based guest virtualization support.
//!
//! This module contains the x86-specific guest CPU model, VMX/VMCS helpers,
//! EPT support, and VM-exit decoding used by OSTD's guest virtualization layer.
//!
//! Public exports are limited to the vCPU state and exit types that the
//! kernel-side KVM-compatible device needs. VMX implementation details remain
//! crate-private.

pub(crate) mod context;
pub(crate) mod control_regs;
pub(crate) mod ept;
pub(crate) mod exit;
pub(crate) mod guest_mode;
mod host_context;
pub(crate) mod interrupt;
mod types;
pub(crate) mod vmcs;
mod vmcs_state;
pub(crate) mod vmx;
pub(crate) mod x86;

#[cfg(ktest)]
mod tests;

pub use self::{
    context::{GuestContext, VcpuRunState},
    exit::GuestExitInfo,
    types::{
        GuestInterrupt, GuestTimerInstant, VcpuDtable, VcpuRegs, VcpuSegment, VcpuSregs,
        X86GprIndex,
    },
    vmx::VmxExitReason,
};
