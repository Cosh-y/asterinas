use x86_64::registers::control::{Cr0Flags, Cr4Flags, EferFlags};

use crate::{
    Error,
    arch::vm::{
        context::GuestContext,
        vmx::{Msr, VmcsReadOnlyNW},
    },
    prelude::*,
    sync::Mutex,
};

pub(crate) fn emulate_cr_access(context: &Mutex<GuestContext>) -> Result<()> {
    let qualification = VmcsReadOnlyNW::EXIT_QUALIFICATION
        .read()
        .map_err(Error::from)?;
    let cr_index = (qualification & 0xF) as u8;
    let access = ((qualification >> 4) & 0b11) as u8;
    let gpr_index = ((qualification >> 8) & 0xF) as u8;
    let gpr_index = map_exit_qualification_gpr_index_to_common_gpr_index(gpr_index);

    match access {
        // write
        0 => {
            let value = context.lock().arch().gpr(gpr_index);
            match cr_index {
                // A value different from the shadow was written to the masked bit.
                0 => emulate_cr0_write(context, value)?,
                2 => context.lock().arch_mut().set_cr2(value),
                3 => context.lock().arch_mut().set_cr3(value),
                4 => emulate_cr4_write(context, value)?,
                other => warn!("rustshyper: ignoring guest write to CR{}", other),
            }
        }
        // read
        1 => {
            let value = match cr_index {
                0 => {
                    error!("rustshyper: guest read CR0 causes VM-exit, which should not happen");
                    0
                }
                2 => context.lock().arch().cr2(),
                3 => context.lock().arch().cr3(),
                4 => {
                    error!("rustshyper: guest read CR4 causes VM-exit, which should not happen");
                    0
                }
                other => {
                    warn!("rustshyper: ignoring guest read from CR{}", other);
                    0
                }
            };
            context.lock().arch_mut().set_gpr(gpr_index, 8, value);
        }
        other => {
            warn!("rustshyper: unsupported CR access type {}", other);
        }
    }

    Ok(())
}

fn emulate_cr0_write(context: &Mutex<GuestContext>, value: u64) -> Result<()> {
    let sanitized_value = sanitize_guest_cr0(value);
    let shadow_value = sanitized_value;
    let actual_value = sanitized_value;

    let mut guest_efer = context.lock().arch().msr(x86::msr::IA32_EFER);
    if (guest_efer & EferFlags::LONG_MODE_ENABLE.bits()) != 0
        && (actual_value & Cr0Flags::PAGING.bits()) != 0
    {
        guest_efer |= EferFlags::LONG_MODE_ACTIVE.bits();
    } else {
        guest_efer &= !EferFlags::LONG_MODE_ACTIVE.bits();
    }

    context
        .lock()
        .arch_mut()
        .set_msr(x86::msr::IA32_EFER, guest_efer);
    context.lock().arch_mut().set_cr0(actual_value);
    Ok(())
}

pub(crate) fn sanitize_guest_cr0(cr0: u64) -> u64 {
    let fixed0 = Msr::IA32_VMX_CR0_FIXED0.read();
    let fixed1 = Msr::IA32_VMX_CR0_FIXED1.read();

    // The PE/PG bit cannot be forcibly set to 1 based on fixed0. I can not explain.
    let valid_cr0 = (cr0
        | (fixed0 & !Cr0Flags::PROTECTED_MODE_ENABLE.bits() & !Cr0Flags::PAGING.bits()))
        & fixed1;

    if cr0 != valid_cr0 {
        error!(
            "rustshyper: guest CR0 sanitized from {} to {}",
            format_binary64_grouped(cr0),
            format_binary64_grouped(valid_cr0)
        );
    }

    valid_cr0
}

pub(crate) fn emulate_cr4_write(context: &Mutex<GuestContext>, value: u64) -> Result<()> {
    let actual_value = sanitize_guest_cr4(value);
    let shadow_value = actual_value;
    error!(
        "guest write cr4: original value {:x}, sanitized value {:x}",
        value, actual_value
    );
    context.lock().arch_mut().set_cr4(actual_value);

    Ok(())
}

pub(crate) fn sanitize_guest_cr4(value: u64) -> u64 {
    let fixed0 = Msr::IA32_VMX_CR4_FIXED0.read();
    let fixed1 = Msr::IA32_VMX_CR4_FIXED1.read();

    // actual hardware CR4 in VMCS must satisfy fixed bits
    (value | fixed0 | Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS.bits())
        & (fixed1 & !Cr4Flags::FSGSBASE.bits())
}

fn map_exit_qualification_gpr_index_to_common_gpr_index(index: u8) -> u8 {
    // The exit qualification field in the VM-Exit Information Fields of the VMCS
    // uses a different numbering method for gpr than that used in this project.
    // Therefore, a mapping layer is needed.
    //
    // The encoding method of the "exit qualification" field:
    // Intel® 64 and IA-32 Architectures Software Developer’s Manual
    // Volume 3, 29.2.1 Basic VM-Exit Information
    match index {
        0 => 0,
        1 => 2,
        2 => 3,
        3 => 1,
        4 => 7,
        5 => 6,
        6 => 4,
        7 => 5,
        other => other,
    }
}

use alloc::{format, string::String, vec::Vec};
use core::str;

pub fn format_binary64_grouped(val: u64) -> String {
    let bits = format!("{:064b}", val);

    let grouped = bits
        .as_bytes()
        .chunks(4)
        .map(|chunk| str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join("_");

    format!("0b{}", grouped)
}
