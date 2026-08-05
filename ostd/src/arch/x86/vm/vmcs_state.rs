// SPDX-License-Identifier: MPL-2.0

use x86::msr::*;

use super::{
    context::GuestContext,
    control_regs::{VcpuControlRegister, VcpuControlRegisters},
    types::{VcpuDtable, VcpuSegment, X86GprIndex},
    vmcs::segment_access_rights,
    vmx::{
        VmcsControl32, VmcsControlNW, VmcsGuest16, VmcsGuest32, VmcsGuest64, VmcsGuestNW,
        VmxExitInfo, VmxExitReason,
    },
};
use crate::prelude::*;

pub(super) fn load_guest_run_msrs(context: &GuestContext) -> Result<()> {
    context.vmcs.write_guest_run_msrs([
        context.arch().msr(IA32_STAR),
        context.arch().msr(IA32_LSTAR),
        context.arch().msr(IA32_CSTAR),
        context.arch().msr(IA32_FMASK),
        context.arch().msr(IA32_KERNEL_GSBASE),
    ])
}

pub(super) fn apply_dirty_vmcs_state(context: &mut GuestContext) -> Result<()> {
    let dirty = context.vmcs_dirty();
    if dirty.rip {
        VmcsGuestNW::RIP.write(context.arch().rip() as usize)?;
    }
    if dirty.rsp {
        VmcsGuestNW::RSP.write(context.arch().gpr(X86GprIndex::Rsp) as usize)?;
    }
    if dirty.rflags {
        // Architectural bit 1 is fixed to one.
        VmcsGuestNW::RFLAGS.write((context.arch().rflags() | 0x2) as usize)?;
    }
    if dirty.sregs {
        write_sregs_to_vmcs(context)?;
    } else {
        if dirty.efer {
            write_guest_efer_to_vmcs(context)?;
        }
        if dirty.pat {
            VmcsGuest64::IA32_PAT.write(context.arch().msr(IA32_PAT))?;
        }
        if dirty.fs_base {
            VmcsGuestNW::FS_BASE.write(context.arch().msr(IA32_FS_BASE) as usize)?;
        }
        if dirty.gs_base {
            VmcsGuestNW::GS_BASE.write(context.arch().msr(IA32_GS_BASE) as usize)?;
        }
        if dirty.sysenter_cs {
            VmcsGuest32::IA32_SYSENTER_CS.write(context.arch().msr(IA32_SYSENTER_CS) as u32)?;
        }
        if dirty.sysenter_esp {
            VmcsGuestNW::IA32_SYSENTER_ESP.write(context.arch().msr(IA32_SYSENTER_ESP) as usize)?;
        }
        if dirty.sysenter_eip {
            VmcsGuestNW::IA32_SYSENTER_EIP.write(context.arch().msr(IA32_SYSENTER_EIP) as usize)?;
        }
        if dirty.run_msrs {
            load_guest_run_msrs(context)?;
        }
    }
    context.clear_vmcs_dirty();
    Ok(())
}

pub(super) fn save_guest_context(
    context: &mut GuestContext,
    exit_info: &VmxExitInfo,
) -> Result<()> {
    context.arch_mut().save_fpu();
    save_guest_run_msrs(context)?;
    context
        .arch_mut()
        .set_fs_base(VmcsGuestNW::FS_BASE.read()? as u64);
    context
        .arch_mut()
        .set_gs_base(VmcsGuestNW::GS_BASE.read()? as u64);

    use x86_64::registers::control::Cr2;
    context.arch_mut().set_cr2(Cr2::read_raw());

    // VM-exit information already contains RIP, so do not issue a second
    // VMREAD for the hottest field.
    context.arch_mut().set_rip(exit_info.guest_rip);

    match VmxExitReason::try_from(exit_info.exit_reason) {
        Ok(VmxExitReason::CR_ACCESS) => save_full_sregs_from_vmcs(context)?,
        Ok(VmxExitReason::EPT_VIOLATION) => {
            save_address_translation_state_from_vmcs(context, true)?;
            context
                .arch_mut()
                .set_gpr(X86GprIndex::Rsp, 8, VmcsGuestNW::RSP.read()? as u64);
        }
        Ok(VmxExitReason::IO_INSTRUCTION) => {
            let is_input = exit_info.exit_qualification & (1 << 3) != 0;
            let is_string = exit_info.exit_qualification & (1 << 4) != 0;
            let is_dword = exit_info.exit_qualification & 0b111 == 3;
            if is_string {
                save_pio_string_state_from_vmcs(context)?;
            } else if is_input && is_dword {
                save_address_translation_state_from_vmcs(context, true)?;
            }
        }
        Ok(VmxExitReason::MSR_WRITE)
            if context.arch().gpr(X86GprIndex::Rcx) as u32 == IA32_EFER =>
        {
            save_full_sregs_from_vmcs(context)?;
        }
        _ => {}
    }

    Ok(())
}

pub(super) fn synchronize_vmcs_state(context: &mut GuestContext) -> Result<()> {
    context.arch_mut().set_rip(VmcsGuestNW::RIP.read()?);
    context
        .arch_mut()
        .set_gpr(X86GprIndex::Rsp, 8, VmcsGuestNW::RSP.read()? as u64);
    context
        .arch_mut()
        .set_rflags(VmcsGuestNW::RFLAGS.read()? as u64);

    // Reading all special registers also materializes FS.base, GS.base, and
    // EFER into the MSR cache, so do not read those VMCS fields again below.
    save_full_sregs_from_vmcs(context)?;
    save_guest_run_msrs(context)?;
    context.arch_mut().set_msr(
        IA32_SYSENTER_CS,
        u64::from(VmcsGuest32::IA32_SYSENTER_CS.read()?),
    );
    context.arch_mut().set_msr(
        IA32_SYSENTER_ESP,
        VmcsGuestNW::IA32_SYSENTER_ESP.read()? as u64,
    );
    context.arch_mut().set_msr(
        IA32_SYSENTER_EIP,
        VmcsGuestNW::IA32_SYSENTER_EIP.read()? as u64,
    );
    Ok(())
}

fn save_guest_run_msrs(context: &mut GuestContext) -> Result<()> {
    let [star, lstar, cstar, syscall_mask, kernel_gs_base] = context.vmcs.read_guest_run_msrs()?;

    context.arch_mut().set_msr(IA32_STAR, star);
    context.arch_mut().set_msr(IA32_LSTAR, lstar);
    context.arch_mut().set_msr(IA32_CSTAR, cstar);
    context.arch_mut().set_msr(IA32_FMASK, syscall_mask);
    context
        .arch_mut()
        .set_msr(IA32_KERNEL_GSBASE, kernel_gs_base);
    Ok(())
}

fn read_dtable_from_vmcs(base_field: VmcsGuestNW, limit_field: VmcsGuest32) -> Result<VcpuDtable> {
    Ok(VcpuDtable {
        base: base_field.read()? as u64,
        limit: limit_field.read()? as u16,
        padding: [0; 3],
    })
}

fn write_sregs_to_vmcs(context: &GuestContext) -> Result<()> {
    let arch = context.arch();
    let sregs = arch.sregs();

    write_control_registers_to_vmcs(arch.control_regs())?;
    VmcsGuestNW::CR3.write(arch.cr3() as usize)?;
    write_guest_efer_to_vmcs(context)?;

    VmcsGuest64::IA32_PAT.write(arch.msr(IA32_PAT))?;
    VmcsGuest32::IA32_SYSENTER_CS.write(arch.msr(IA32_SYSENTER_CS) as u32)?;
    VmcsGuestNW::IA32_SYSENTER_ESP.write(arch.msr(IA32_SYSENTER_ESP) as usize)?;
    VmcsGuestNW::IA32_SYSENTER_EIP.write(arch.msr(IA32_SYSENTER_EIP) as usize)?;

    write_segment_to_vmcs(
        &sregs.cs,
        VmcsGuest16::CS_SELECTOR,
        VmcsGuestNW::CS_BASE,
        VmcsGuest32::CS_LIMIT,
        VmcsGuest32::CS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.ds,
        VmcsGuest16::DS_SELECTOR,
        VmcsGuestNW::DS_BASE,
        VmcsGuest32::DS_LIMIT,
        VmcsGuest32::DS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.es,
        VmcsGuest16::ES_SELECTOR,
        VmcsGuestNW::ES_BASE,
        VmcsGuest32::ES_LIMIT,
        VmcsGuest32::ES_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.fs,
        VmcsGuest16::FS_SELECTOR,
        VmcsGuestNW::FS_BASE,
        VmcsGuest32::FS_LIMIT,
        VmcsGuest32::FS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.gs,
        VmcsGuest16::GS_SELECTOR,
        VmcsGuestNW::GS_BASE,
        VmcsGuest32::GS_LIMIT,
        VmcsGuest32::GS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.ss,
        VmcsGuest16::SS_SELECTOR,
        VmcsGuestNW::SS_BASE,
        VmcsGuest32::SS_LIMIT,
        VmcsGuest32::SS_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.tr,
        VmcsGuest16::TR_SELECTOR,
        VmcsGuestNW::TR_BASE,
        VmcsGuest32::TR_LIMIT,
        VmcsGuest32::TR_ACCESS_RIGHTS,
    )?;
    write_segment_to_vmcs(
        &sregs.ldt,
        VmcsGuest16::LDTR_SELECTOR,
        VmcsGuestNW::LDTR_BASE,
        VmcsGuest32::LDTR_LIMIT,
        VmcsGuest32::LDTR_ACCESS_RIGHTS,
    )?;
    VmcsGuestNW::GDTR_BASE.write(sregs.gdt.base as usize)?;
    VmcsGuest32::GDTR_LIMIT.write(u32::from(sregs.gdt.limit))?;
    VmcsGuestNW::IDTR_BASE.write(sregs.idt.base as usize)?;
    VmcsGuest32::IDTR_LIMIT.write(u32::from(sregs.idt.limit))?;
    Ok(())
}

fn write_guest_efer_to_vmcs(context: &GuestContext) -> Result<()> {
    use x86::vmx::vmcs::control::EntryControls;
    use x86_64::registers::model_specific::EferFlags;

    let guest_efer = context.arch().msr(IA32_EFER);
    VmcsGuest64::IA32_EFER.write(guest_efer)?;
    let mut entry = VmcsControl32::VMENTRY_CONTROLS.read()?;
    if guest_efer & EferFlags::LONG_MODE_ACTIVE.bits() != 0 {
        entry |= EntryControls::IA32E_MODE_GUEST.bits();
    } else {
        entry &= !EntryControls::IA32E_MODE_GUEST.bits();
    }
    VmcsControl32::VMENTRY_CONTROLS.write(entry)
}

fn write_segment_to_vmcs(
    segment: &VcpuSegment,
    selector_field: VmcsGuest16,
    base_field: VmcsGuestNW,
    limit_field: VmcsGuest32,
    rights_field: VmcsGuest32,
) -> Result<()> {
    selector_field.write(segment.selector)?;
    base_field.write(segment.base as usize)?;
    limit_field.write(segment.limit)?;
    rights_field.write(segment_access_rights(segment))
}

fn save_address_translation_state_from_vmcs(
    context: &mut GuestContext,
    save_cs: bool,
) -> Result<()> {
    context.arch_mut().set_cr3(VmcsGuestNW::CR3.read()? as u64);
    context
        .arch_mut()
        .set_control_regs_from_vmcs(read_control_registers_from_vmcs()?);
    context
        .arch_mut()
        .set_msr(IA32_EFER, VmcsGuest64::IA32_EFER.read()?);
    if save_cs {
        context.arch_mut().set_cs(read_segment_from_vmcs(
            VmcsGuest16::CS_SELECTOR,
            VmcsGuestNW::CS_BASE,
            VmcsGuest32::CS_LIMIT,
            VmcsGuest32::CS_ACCESS_RIGHTS,
        )?);
    }
    Ok(())
}

fn save_pio_string_state_from_vmcs(context: &mut GuestContext) -> Result<()> {
    save_address_translation_state_from_vmcs(context, true)?;
    context
        .arch_mut()
        .set_rflags(VmcsGuestNW::RFLAGS.read()? as u64);
    save_data_segments_from_vmcs(context)
}

fn save_data_segments_from_vmcs(context: &mut GuestContext) -> Result<()> {
    context.arch_mut().set_ds(read_segment_from_vmcs(
        VmcsGuest16::DS_SELECTOR,
        VmcsGuestNW::DS_BASE,
        VmcsGuest32::DS_LIMIT,
        VmcsGuest32::DS_ACCESS_RIGHTS,
    )?);
    context.arch_mut().set_es(read_segment_from_vmcs(
        VmcsGuest16::ES_SELECTOR,
        VmcsGuestNW::ES_BASE,
        VmcsGuest32::ES_LIMIT,
        VmcsGuest32::ES_ACCESS_RIGHTS,
    )?);
    context.arch_mut().set_fs(read_segment_from_vmcs(
        VmcsGuest16::FS_SELECTOR,
        VmcsGuestNW::FS_BASE,
        VmcsGuest32::FS_LIMIT,
        VmcsGuest32::FS_ACCESS_RIGHTS,
    )?);
    context.arch_mut().set_gs(read_segment_from_vmcs(
        VmcsGuest16::GS_SELECTOR,
        VmcsGuestNW::GS_BASE,
        VmcsGuest32::GS_LIMIT,
        VmcsGuest32::GS_ACCESS_RIGHTS,
    )?);
    context.arch_mut().set_ss(read_segment_from_vmcs(
        VmcsGuest16::SS_SELECTOR,
        VmcsGuestNW::SS_BASE,
        VmcsGuest32::SS_LIMIT,
        VmcsGuest32::SS_ACCESS_RIGHTS,
    )?);
    Ok(())
}

fn save_full_sregs_from_vmcs(context: &mut GuestContext) -> Result<()> {
    save_address_translation_state_from_vmcs(context, true)?;
    save_data_segments_from_vmcs(context)?;
    context.arch_mut().set_tr(read_segment_from_vmcs(
        VmcsGuest16::TR_SELECTOR,
        VmcsGuestNW::TR_BASE,
        VmcsGuest32::TR_LIMIT,
        VmcsGuest32::TR_ACCESS_RIGHTS,
    )?);
    context.arch_mut().set_ldt(read_segment_from_vmcs(
        VmcsGuest16::LDTR_SELECTOR,
        VmcsGuestNW::LDTR_BASE,
        VmcsGuest32::LDTR_LIMIT,
        VmcsGuest32::LDTR_ACCESS_RIGHTS,
    )?);
    context.arch_mut().set_gdt(read_dtable_from_vmcs(
        VmcsGuestNW::GDTR_BASE,
        VmcsGuest32::GDTR_LIMIT,
    )?);
    context.arch_mut().set_idt(read_dtable_from_vmcs(
        VmcsGuestNW::IDTR_BASE,
        VmcsGuest32::IDTR_LIMIT,
    )?);
    Ok(())
}

fn write_control_registers_to_vmcs(control_regs: VcpuControlRegisters) -> Result<()> {
    let cr0 = control_regs.cr0();
    VmcsGuestNW::CR0.write(cr0.real() as usize)?;
    VmcsControlNW::CR0_GUEST_HOST_MASK.write(cr0.host_mask() as usize)?;
    VmcsControlNW::CR0_READ_SHADOW.write(cr0.read_shadow() as usize)?;

    let cr4 = control_regs.cr4();
    VmcsGuestNW::CR4.write(cr4.real() as usize)?;
    VmcsControlNW::CR4_GUEST_HOST_MASK.write(cr4.host_mask() as usize)?;
    VmcsControlNW::CR4_READ_SHADOW.write(cr4.read_shadow() as usize)
}

fn read_control_registers_from_vmcs() -> Result<VcpuControlRegisters> {
    let cr0 = VcpuControlRegister::from_vmcs(
        VmcsControlNW::CR0_GUEST_HOST_MASK.read()? as u64,
        VmcsControlNW::CR0_READ_SHADOW.read()? as u64,
        VmcsGuestNW::CR0.read()? as u64,
    );
    let cr4 = VcpuControlRegister::from_vmcs(
        VmcsControlNW::CR4_GUEST_HOST_MASK.read()? as u64,
        VmcsControlNW::CR4_READ_SHADOW.read()? as u64,
        VmcsGuestNW::CR4.read()? as u64,
    );
    Ok(VcpuControlRegisters::from_vmcs(cr0, cr4))
}

fn read_segment_from_vmcs(
    selector_field: VmcsGuest16,
    base_field: VmcsGuestNW,
    limit_field: VmcsGuest32,
    rights_field: VmcsGuest32,
) -> Result<VcpuSegment> {
    let rights = rights_field.read()?;
    Ok(VcpuSegment {
        base: base_field.read()? as u64,
        limit: limit_field.read()?,
        selector: selector_field.read()?,
        type_: (rights & 0x0f) as u8,
        present: ((rights >> 7) & 0x1) as u8,
        dpl: ((rights >> 5) & 0x3) as u8,
        db: ((rights >> 14) & 0x1) as u8,
        s: ((rights >> 4) & 0x1) as u8,
        l: ((rights >> 13) & 0x1) as u8,
        g: ((rights >> 15) & 0x1) as u8,
        avl: ((rights >> 12) & 0x1) as u8,
        unusable: ((rights >> 16) & 0x1) as u8,
        padding: 0,
    })
}
