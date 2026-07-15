// SPDX-License-Identifier: MPL-2.0

/*
 * This file contains code derived from the RVM-Tutorial project.
 * Source: https://github.com/equation314/RVM-Tutorial
 */
#[expect(
    non_camel_case_types,
    reason = "VMX names follow Intel SDM terminology, and the catalog includes MSRs reserved for future VMX paths."
)]
#[cfg_attr(
    not(ktest),
    expect(
        dead_code,
        reason = "VMX names follow Intel SDM terminology, and the catalog includes MSRs reserved for future VMX paths."
    )
)]
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Msr {
    IA32_FEATURE_CONTROL = 0x3a,

    IA32_PAT = 0x277,

    IA32_VMX_BASIC = 0x480,
    IA32_VMX_PINBASED_CTLS = 0x481,
    IA32_VMX_PROCBASED_CTLS = 0x482,
    IA32_VMX_EXIT_CTLS = 0x483,
    IA32_VMX_ENTRY_CTLS = 0x484,
    IA32_VMX_MISC = 0x485,
    IA32_VMX_CR0_FIXED0 = 0x486,
    IA32_VMX_CR0_FIXED1 = 0x487,
    IA32_VMX_CR4_FIXED0 = 0x488,
    IA32_VMX_CR4_FIXED1 = 0x489,
    IA32_VMX_PROCBASED_CTLS2 = 0x48b,
    IA32_VMX_EPT_VPID_CAP = 0x48c,
    IA32_VMX_TRUE_PINBASED_CTLS = 0x48d,
    IA32_VMX_TRUE_PROCBASED_CTLS = 0x48e,
    IA32_VMX_TRUE_EXIT_CTLS = 0x48f,
    IA32_VMX_TRUE_ENTRY_CTLS = 0x490,

    IA32_EFER = 0xc000_0080,
    IA32_STAR = 0xc000_0081,
    IA32_LSTAR = 0xc000_0082,
    IA32_CSTAR = 0xc000_0083,
    IA32_FMASK = 0xc000_0084,

    IA32_FS_BASE = 0xc000_0100,
    IA32_GS_BASE = 0xc000_0101,
    IA32_KERNEL_GSBASE = 0xc000_0102,
}

#[inline]
unsafe fn rdmsr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") low,
            out("edx") high,
            options(nomem, nostack)
        );
    }
    ((high as u64) << 32) | (low as u64)
}

#[inline]
unsafe fn wrmsr(msr: u32, value: u64) {
    let low = value as u32;
    let high = (value >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") low,
            in("edx") high,
            options(nostack)
        );
    }
}

impl Msr {
    pub(crate) fn read(&self) -> u64 {
        unsafe { rdmsr(*self as u32) }
    }

    pub(crate) fn write(&self, value: u64) {
        unsafe { wrmsr(*self as u32, value) }
    }
}
