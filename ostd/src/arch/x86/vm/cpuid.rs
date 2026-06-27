// SPDX-License-Identifier: MPL-2.0

//! Provides the CPUID model visible to an x86 guest vCPU.

use core::arch::x86_64::CpuidResult;

use crate::{
    arch::{cpu::cpuid::cpuid, tsc_freq},
    prelude::*,
};

/// The CPUID entry matches the `ECX` subleaf.
const GUEST_CPUID_FLAG_SIGNIFICANT_INDEX: u32 = 1 << 0;
const DEFAULT_CPUID_VCPU_COUNT: u32 = 1;
const DEFAULT_CPUID_APIC_ID: u32 = 0;

/// A CPUID result entry visible to a guest vCPU.
#[derive(Clone, Copy, Debug)]
pub struct GuestCpuidEntry {
    /// CPUID function, i.e., input `EAX`.
    pub function: u32,
    /// CPUID index/subleaf, i.e., input `ECX`.
    pub index: u32,
    /// KVM-compatible flags describing how the entry should be matched.
    pub flags: u32,
    /// Output `EAX`.
    pub eax: u32,
    /// Output `EBX`.
    pub ebx: u32,
    /// Output `ECX`.
    pub ecx: u32,
    /// Output `EDX`.
    pub edx: u32,
}

impl GuestCpuidEntry {
    fn new(function: u32, index: u32, flags: u32, result: CpuidResult) -> Self {
        Self {
            function,
            index,
            flags,
            eax: result.eax,
            ebx: result.ebx,
            ecx: result.ecx,
            edx: result.edx,
        }
    }

    pub(crate) fn matches(self, function: u32, index: u32) -> bool {
        if self.function != function {
            return false;
        }
        self.flags & GUEST_CPUID_FLAG_SIGNIFICANT_INDEX == 0 || self.index == index
    }
}

impl Default for GuestCpuidEntry {
    fn default() -> Self {
        Self::new(
            0,
            0,
            0,
            CpuidResult {
                eax: 0,
                ebx: 0,
                ecx: 0,
                edx: 0,
            },
        )
    }
}

/// Returns the default CPUID entries supported by the hypervisor.
///
/// The returned entries are a feature template with a neutral uniprocessor
/// topology. Userspace remains responsible for supplying the final per-vCPU
/// CPUID set through `KVM_SET_CPUID2`.
pub fn default_cpuid_entries() -> Vec<GuestCpuidEntry> {
    const MAX_BASIC_CPUID: u32 = 0x16;
    const MAX_EXTENDED_CPUID: u32 = 0x8000_0008;

    let max_basic = cpuid(0, 0)
        .map(|result| result.eax)
        .unwrap_or(0)
        .max(MAX_BASIC_CPUID);
    let mut entries = Vec::new();

    for function in 0..=max_basic {
        match function {
            4 => push_cache_cpuid_entries(&mut entries),
            7 => push_indexed_cpuid_entry(&mut entries, function, 0),
            0x0b | 0x1f => push_default_topology_cpuid_entries(&mut entries, function),
            0x0d => push_indexed_cpuid_entry(&mut entries, function, 0),
            _ => entries.push(default_cpuid_entry(function, 0, 0)),
        }
    }

    let max_extended = cpuid(0x8000_0000, 0)
        .map(|result| result.eax)
        .unwrap_or(0x8000_0000)
        .min(MAX_EXTENDED_CPUID);
    for function in 0x8000_0000..=max_extended {
        entries.push(default_cpuid_entry(function, 0, 0));
    }

    entries
}

fn push_cache_cpuid_entries(entries: &mut Vec<GuestCpuidEntry>) {
    const MAX_CACHE_SUBLEAVES: u32 = 16;

    for index in 0..MAX_CACHE_SUBLEAVES {
        let entry = default_cpuid_entry(4, index, GUEST_CPUID_FLAG_SIGNIFICANT_INDEX);
        let cache_type = entry.eax & 0x1f;
        entries.push(entry);
        if cache_type == 0 {
            break;
        }
    }
}

fn push_default_topology_cpuid_entries(entries: &mut Vec<GuestCpuidEntry>, function: u32) {
    for index in 0..=2 {
        entries.push(default_cpuid_entry(
            function,
            index,
            GUEST_CPUID_FLAG_SIGNIFICANT_INDEX,
        ));
    }
}

fn push_indexed_cpuid_entry(entries: &mut Vec<GuestCpuidEntry>, function: u32, index: u32) {
    entries.push(default_cpuid_entry(
        function,
        index,
        GUEST_CPUID_FLAG_SIGNIFICANT_INDEX,
    ));
}

fn default_cpuid_entry(function: u32, index: u32, flags: u32) -> GuestCpuidEntry {
    let result = cpuid(function, index).unwrap_or(CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 0,
        edx: 0,
    });
    let result = sanitize_cpuid_result(function, index, result);

    GuestCpuidEntry::new(function, index, flags, result)
}

fn sanitize_cpuid_result(function: u32, index: u32, result: CpuidResult) -> CpuidResult {
    const CPUID_1_ECX_VMX: u32 = 1 << 5;
    const CPUID_1_ECX_FMA: u32 = 1 << 12;
    const CPUID_1_ECX_PCID: u32 = 1 << 17;
    const CPUID_1_ECX_X2APIC: u32 = 1 << 21;
    const CPUID_1_ECX_TSC_DEADLINE: u32 = 1 << 24;
    const CPUID_1_ECX_XSAVE: u32 = 1 << 26;
    const CPUID_1_ECX_OSXSAVE: u32 = 1 << 27;
    const CPUID_1_ECX_AVX: u32 = 1 << 28;
    const CPUID_1_EDX_APIC: u32 = 1 << 9;
    const CPUID_1_EDX_HTT: u32 = 1 << 28;
    const CPUID_7_EBX_FSGSBASE: u32 = 1 << 0;
    const CPUID_7_EBX_HLE: u32 = 1 << 4;
    const CPUID_7_EBX_AVX2: u32 = 1 << 5;
    const CPUID_7_EBX_INVPCID: u32 = 1 << 10;
    const CPUID_7_EBX_RTM: u32 = 1 << 11;
    const CPUID_7_EBX_AVX512F: u32 = 1 << 16;
    const CPUID_7_EBX_AVX512DQ: u32 = 1 << 17;
    const CPUID_7_EBX_AVX512CD: u32 = 1 << 28;
    const CPUID_7_EBX_AVX512BW: u32 = 1 << 30;
    const CPUID_7_EBX_AVX512VL: u32 = 1 << 31;
    const CPUID_7_ECX_AVX512VBMI: u32 = 1 << 1;
    const CPUID_7_ECX_VAES: u32 = 1 << 9;
    const CPUID_7_ECX_VPCLMULQDQ: u32 = 1 << 10;
    const CPUID_7_ECX_AVX512VNNI: u32 = 1 << 11;
    const CPUID_7_ECX_AVX512BITALG: u32 = 1 << 12;
    const CPUID_7_ECX_AVX512VPOPCNTDQ: u32 = 1 << 14;
    const CPUID_TSC_CRYSTAL_HZ: u32 = 1_000_000;

    let CpuidResult {
        mut eax,
        mut ebx,
        mut ecx,
        mut edx,
    } = result;

    match function {
        0 => {
            eax = eax.max(0x16);
        }
        1 => {
            ecx &= !(CPUID_1_ECX_VMX
                | CPUID_1_ECX_FMA
                | CPUID_1_ECX_X2APIC
                | CPUID_1_ECX_TSC_DEADLINE
                | CPUID_1_ECX_PCID
                | CPUID_1_ECX_XSAVE
                | CPUID_1_ECX_OSXSAVE
                | CPUID_1_ECX_AVX);
            ebx = (ebx & 0x0000_ffff)
                | ((DEFAULT_CPUID_VCPU_COUNT & 0xff) << 16)
                | ((DEFAULT_CPUID_APIC_ID & 0xff) << 24);
            edx |= CPUID_1_EDX_APIC;
            edx &= !CPUID_1_EDX_HTT;
        }
        4 if (eax & 0x1f) != 0 => {
            let cores_per_package_minus_one = DEFAULT_CPUID_VCPU_COUNT.saturating_sub(1).min(0x3f);
            eax = (eax & !(0x3f << 26)) | (cores_per_package_minus_one << 26);
        }
        7 if index == 0 => {
            ebx &= !(CPUID_7_EBX_FSGSBASE
                | CPUID_7_EBX_HLE
                | CPUID_7_EBX_AVX2
                | CPUID_7_EBX_RTM
                | CPUID_7_EBX_INVPCID
                | CPUID_7_EBX_AVX512F
                | CPUID_7_EBX_AVX512DQ
                | CPUID_7_EBX_AVX512CD
                | CPUID_7_EBX_AVX512BW
                | CPUID_7_EBX_AVX512VL);
            ecx &= !(CPUID_7_ECX_AVX512VBMI
                | CPUID_7_ECX_VAES
                | CPUID_7_ECX_VPCLMULQDQ
                | CPUID_7_ECX_AVX512VNNI
                | CPUID_7_ECX_AVX512BITALG
                | CPUID_7_ECX_AVX512VPOPCNTDQ);
        }
        0x0d => {
            eax = 0;
            ebx = 0;
            ecx = 0;
            edx = 0;
        }
        0x0b | 0x1f => {
            let topology = default_topology_cpuid(index);
            eax = topology.eax;
            ebx = topology.ebx;
            ecx = topology.ecx;
            edx = topology.edx;
        }
        0x15 => {
            if let Some(tsc_mhz) = virtual_tsc_mhz() {
                eax = 1;
                ebx = tsc_mhz;
                ecx = CPUID_TSC_CRYSTAL_HZ;
                edx = 0;
            }
        }
        0x16 => {
            if let Some(tsc_mhz) = virtual_tsc_mhz() {
                eax = tsc_mhz;
                ebx = tsc_mhz;
                ecx = 0;
                edx = 0;
            }
        }
        _ => {}
    }

    CpuidResult { eax, ebx, ecx, edx }
}

fn default_topology_cpuid(subleaf: u32) -> CpuidResult {
    CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: subleaf,
        edx: DEFAULT_CPUID_APIC_ID,
    }
}

fn virtual_tsc_mhz() -> Option<u32> {
    let mhz = (tsc_freq().saturating_add(500_000)) / 1_000_000;
    u32::try_from(mhz).ok().filter(|&mhz| mhz != 0)
}
