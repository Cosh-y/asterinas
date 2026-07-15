// SPDX-License-Identifier: MPL-2.0

//! Emulates guest control-register accesses in the kernel hypervisor layer.

use ostd::arch::vm::{GuestExitInfo, VcpuSregs, X86GprIndex};

use super::vcpu::Vcpu;
use crate::prelude::*;

const CR0_PAGING: u64 = 1 << 31;
const EFER_LONG_MODE_ENABLE: u64 = 1 << 8;
const EFER_LONG_MODE_ACTIVE: u64 = 1 << 10;

/// Emulates a VM-exit caused by a guest control-register access.
pub(super) fn emulate_cr_access(vcpu: &Vcpu, exit_info: &GuestExitInfo) -> Result<()> {
    let qualification = exit_info.exit_qualification;
    let cr_index = (qualification & 0xF) as u8;
    let access = ((qualification >> 4) & 0b11) as u8;
    let gpr_encoding = ((qualification >> 8) & 0xF) as u8;
    let gpr = X86GprIndex::from_x86_reg_encoding(gpr_encoding)?;

    match access {
        0 => emulate_cr_write(vcpu, cr_index, gpr),
        1 => emulate_cr_read(vcpu, cr_index, gpr),
        other => {
            warn!("hypervisor: unsupported CR access type {}", other);
        }
    }

    vcpu.guest_context()
        .advance_rip(u64::from(exit_info.instruction_len));
    Ok(())
}

fn emulate_cr_write(vcpu: &Vcpu, cr_index: u8, gpr: X86GprIndex) {
    let mut context = vcpu.guest_context();
    let value = context.gpr(gpr);
    let mut sregs = context.sregs();

    match cr_index {
        0 => sregs.cr0 = value,
        2 => sregs.cr2 = value,
        3 => sregs.cr3 = value,
        4 => sregs.cr4 = value,
        other => {
            warn!("hypervisor: ignoring guest write to CR{}", other);
            return;
        }
    }

    sync_efer_lma(&mut sregs);
    context.set_sregs(sregs);
}

fn emulate_cr_read(vcpu: &Vcpu, cr_index: u8, gpr: X86GprIndex) {
    let mut context = vcpu.guest_context();
    let sregs = context.sregs();
    let value = match cr_index {
        0 => sregs.cr0,
        2 => sregs.cr2,
        3 => sregs.cr3,
        4 => sregs.cr4,
        other => {
            warn!("hypervisor: ignoring guest read from CR{}", other);
            0
        }
    };
    context.set_gpr(gpr, 8, value);
}

/// Synchronizes EFER.LMA from EFER.LME and CR0.PG.
pub(super) fn sync_efer_lma(sregs: &mut VcpuSregs) {
    if sregs.efer & EFER_LONG_MODE_ENABLE != 0 && sregs.cr0 & CR0_PAGING != 0 {
        sregs.efer |= EFER_LONG_MODE_ACTIVE;
    } else {
        sregs.efer &= !EFER_LONG_MODE_ACTIVE;
    }
}
