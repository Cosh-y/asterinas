// SPDX-License-Identifier: MPL-2.0

use crate::{error::Error, prelude::*};

type PhysAddr = Paddr;

/// Enters VMX operation using the supplied VMXON region.
///
/// # Safety
///
/// The physical address must reference a page-aligned VMXON region whose first
/// 31 bits contain the current VMCS revision identifier, and the current CPU
/// must have `CR4.VMXE` set.
#[inline]
pub(crate) unsafe fn vmxon(vmxon_region: PhysAddr) -> Result<()> {
    let mut rflags: u64;
    unsafe {
        core::arch::asm!(
            "vmxon [{}]",
            "pushfq",
            "pop {}",
            in(reg) &vmxon_region,
            out(reg) rflags,
            options(nostack)
        );
    }

    // Check CF (bit 0) and ZF (bit 6)
    if (rflags & 1) != 0 {
        return Err(Error::InvalidArgs); // VMfailInvalid
    }
    if (rflags & (1 << 6)) != 0 {
        return Err(Error::InvalidArgs); // VMfailValid
    }

    Ok(())
}

/// Leaves VMX operation on the current CPU.
#[inline]
pub(crate) fn vmxoff() -> Result<()> {
    let mut rflags: u64;
    // SAFETY: `VMXOFF` only changes VMX operation state on the current CPU.
    unsafe {
        core::arch::asm!(
            "vmxoff",
            "pushfq",
            "pop {}",
            out(reg) rflags,
            options(nostack)
        );
    }

    if (rflags & 1) != 0 {
        return Err(Error::InvalidArgs);
    }
    if (rflags & (1 << 6)) != 0 {
        return Err(Error::InvalidArgs);
    }

    Ok(())
}

/// Clears a VMCS region from any CPU that currently owns it.
#[inline]
pub(crate) fn vmclear(vmcs: u64) -> Result<()> {
    let mut rflags: u64;
    unsafe {
        core::arch::asm!(
            "vmclear [{}]",
            "pushfq",
            "pop {}",
            in(reg) &vmcs,
            out(reg) rflags,
            options(nostack)
        );
    }

    if (rflags & 1) != 0 {
        return Err(Error::InvalidArgs);
    }
    if (rflags & (1 << 6)) != 0 {
        return Err(Error::InvalidArgs);
    }

    Ok(())
}

/// Makes a VMCS region current on this CPU.
#[inline]
pub(crate) fn vmptrld(vmcs: u64) -> Result<()> {
    let mut rflags: u64;
    unsafe {
        core::arch::asm!(
            "vmptrld [{}]",
            "pushfq",
            "pop {}",
            in(reg) &vmcs,
            out(reg) rflags,
            options(nostack)
        );
    }

    if (rflags & 1) != 0 {
        return Err(Error::InvalidArgs);
    }
    if (rflags & (1 << 6)) != 0 {
        return Err(Error::InvalidArgs);
    }

    Ok(())
}
