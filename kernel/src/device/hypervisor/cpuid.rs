// SPDX-License-Identifier: MPL-2.0

//! Provides guest CPUID policy and CPUID-exit emulation.

use core::arch::x86_64::CpuidResult;

use ostd::arch::{
    cpu::cpuid::cpuid,
    tsc_freq,
    vm::{GuestExitInfo, X86GprIndex},
};

use super::{ioctl::VcpuCpuidEntry2, vcpu::Vcpu};
use crate::prelude::*;

// CPUID leaves
const CPUID_LEAF_MAX_INPUT_FOR_BASIC_CPUID: u32 = 0x0;
const CPUID_LEAF_VERSION_INFO: u32 = 0x1;
const CPUID_LEAF_CACHE_PARAMS: u32 = 0x4;
const CPUID_LEAF_THERMAL_AND_POWER_MANAGEMENT: u32 = 0x6;
const CPUID_LEAF_STRUCTURED_EXTENDED_FEATURES: u32 = 0x7;
const CPUID_LEAF_EXTENDED_TOPOLOGY: u32 = 0x0b;
const CPUID_LEAF_PROCESSOR_EXTENDED_STATE: u32 = 0x0d;
const CPUID_LEAF_TSC_AND_CRYSTAL_CLOCK_INFO: u32 = 0x15;
const CPUID_LEAF_PROCESSOR_FREQUENCY_INFO: u32 = 0x16;
const CPUID_LEAF_V2_EXTENDED_TOPOLOGY: u32 = 0x1f;
// CPUID leaves (KVM specific)
// https://www.kernel.org/doc/html/latest/virt/kvm/x86/cpuid.html
const CPUID_LEAF_KVM_SIGNATURE: u32 = 0x4000_0000;
const CPUID_LEAF_KVM_FEATURES: u32 = 0x4000_0001;
// CPUID leaves (extended)
const CPUID_LEAF_MAX_INPUT_FOR_EXTENDED_CPUID: u32 = 0x8000_0000;
const CPUID_LEAF_ADDRESS_SIZE_INFO: u32 = 0x8000_0008;

// CPUID feature flags
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
const CPUID_6_EAX_ARAT: u32 = 1 << 2;
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

const KVM_FEATURE_CLOCKSOURCE: u32 = 1 << 0;
const KVM_FEATURE_CLOCKSOURCE2: u32 = 1 << 3;

const DEFAULT_CPUID_VCPU_COUNT: u32 = 1;
const DEFAULT_CPUID_APIC_ID: u32 = 0;

/// The CPUID entry matches the `ECX` subleaf.
const GUEST_CPUID_FLAG_SIGNIFICANT_INDEX: u32 = 1 << 0;

pub(super) const VIRTUAL_TSC_CRYSTAL_HZ: u64 = 24_000_000;

/// Emulates a guest CPUID instruction using this vCPU's configured entries.
pub(super) fn emulate_cpuid(vcpu: &Vcpu, exit_info: &GuestExitInfo) -> Result<()> {
    let (function, index) = {
        let context = vcpu.guest_context();
        (
            context.gpr(X86GprIndex::Rax) as u32,
            context.gpr(X86GprIndex::Rcx) as u32,
        )
    };
    let entry = vcpu.cpuid_result(function, index);

    let mut context = vcpu.guest_context();
    context.set_gpr(X86GprIndex::Rax, 8, u64::from(entry.eax));
    context.set_gpr(X86GprIndex::Rbx, 8, u64::from(entry.ebx));
    context.set_gpr(X86GprIndex::Rcx, 8, u64::from(entry.ecx));
    context.set_gpr(X86GprIndex::Rdx, 8, u64::from(entry.edx));
    context.advance_rip(u64::from(exit_info.instruction_len));
    Ok(())
}

/// Returns the default CPUID entries supported by the hypervisor.
///
/// The returned entries are a feature template with a neutral uniprocessor
/// topology. Userspace remains responsible for supplying the final per-vCPU
/// CPUID set through `KVM_SET_CPUID2`.
pub(super) fn default_cpuid_entries() -> Vec<VcpuCpuidEntry2> {
    const MAX_BASIC_CPUID: u32 = CPUID_LEAF_PROCESSOR_FREQUENCY_INFO;
    const MAX_EXTENDED_CPUID: u32 = CPUID_LEAF_ADDRESS_SIZE_INFO;

    let max_basic = cpuid(CPUID_LEAF_MAX_INPUT_FOR_BASIC_CPUID, 0)
        .map(|result| result.eax)
        .unwrap_or(0)
        .max(MAX_BASIC_CPUID);
    let mut entries = Vec::new();
    for function in 0..=max_basic {
        let CpuidResult {
            mut eax,
            mut ebx,
            mut ecx,
            mut edx,
        } = cpuid(function, 0).unwrap_or(CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        });
        let mut flags = 0;
        match function {
            CPUID_LEAF_MAX_INPUT_FOR_BASIC_CPUID => {
                eax = eax.max(MAX_BASIC_CPUID);
            }
            CPUID_LEAF_VERSION_INFO => {
                ecx &= !(CPUID_1_ECX_VMX
                    | CPUID_1_ECX_FMA
                    | CPUID_1_ECX_X2APIC
                    | CPUID_1_ECX_PCID
                    | CPUID_1_ECX_XSAVE
                    | CPUID_1_ECX_OSXSAVE
                    | CPUID_1_ECX_AVX);
                ecx |= CPUID_1_ECX_TSC_DEADLINE;
                ebx = (ebx & 0x0000_ffff)
                    | ((DEFAULT_CPUID_VCPU_COUNT & 0xff) << 16)
                    | ((DEFAULT_CPUID_APIC_ID & 0xff) << 24);
                edx |= CPUID_1_EDX_APIC;
                edx &= !CPUID_1_EDX_HTT;
            }
            CPUID_LEAF_CACHE_PARAMS => {
                const MAX_CACHE_SUBLEAVES: u32 = 16;
                for index in 0..MAX_CACHE_SUBLEAVES {
                    let result = cpuid(function, index).unwrap_or(CpuidResult {
                        eax: 0,
                        ebx: 0,
                        ecx: 0,
                        edx: 0,
                    });
                    let cores_per_package_minus_one =
                        DEFAULT_CPUID_VCPU_COUNT.saturating_sub(1).min(0x3f);
                    let eax = (result.eax & !(0x3f << 26)) | (cores_per_package_minus_one << 26);
                    let entry = VcpuCpuidEntry2 {
                        function,
                        index,
                        flags: GUEST_CPUID_FLAG_SIGNIFICANT_INDEX,
                        eax,
                        ebx: result.ebx,
                        ecx: result.ecx,
                        edx: result.edx,
                        padding: [0; 3],
                    };
                    let cache_type = entry.eax & 0x1f;
                    entries.push(entry);
                    if cache_type == 0 {
                        break;
                    }
                }
                continue;
            }
            CPUID_LEAF_THERMAL_AND_POWER_MANAGEMENT => {
                eax |= CPUID_6_EAX_ARAT;
            }
            CPUID_LEAF_STRUCTURED_EXTENDED_FEATURES => {
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
            CPUID_LEAF_EXTENDED_TOPOLOGY | CPUID_LEAF_V2_EXTENDED_TOPOLOGY => {
                for index in 0..=2 {
                    let entry = VcpuCpuidEntry2 {
                        function,
                        index,
                        flags: GUEST_CPUID_FLAG_SIGNIFICANT_INDEX,
                        eax: 0,
                        ebx: 0,
                        ecx: index,
                        edx: DEFAULT_CPUID_APIC_ID,
                        padding: [0; 3],
                    };
                    entries.push(entry);
                }
                continue;
            }
            CPUID_LEAF_PROCESSOR_EXTENDED_STATE => {
                eax = 0;
                ebx = 0;
                ecx = 0;
                edx = 0;
                flags = GUEST_CPUID_FLAG_SIGNIFICANT_INDEX;
            }
            CPUID_LEAF_TSC_AND_CRYSTAL_CLOCK_INFO => {
                if let Some((denominator, numerator, crystal_hz)) = virtual_tsc_ratio() {
                    eax = denominator;
                    ebx = numerator;
                    ecx = crystal_hz;
                    edx = 0;
                }
            }
            CPUID_LEAF_PROCESSOR_FREQUENCY_INFO => {
                if let Some(tsc_mhz) = virtual_tsc_mhz() {
                    eax = tsc_mhz;
                    ebx = tsc_mhz;
                    ecx = 0;
                    edx = 0;
                }
            }
            _ => {}
        }
        let entry = VcpuCpuidEntry2 {
            function,
            index: 0,
            flags,
            eax,
            ebx,
            ecx,
            edx,
            padding: [0; 3],
        };
        entries.push(entry);
    }

    let max_extended = cpuid(CPUID_LEAF_MAX_INPUT_FOR_EXTENDED_CPUID, 0)
        .map(|result| result.eax)
        .unwrap_or(CPUID_LEAF_MAX_INPUT_FOR_EXTENDED_CPUID)
        .min(MAX_EXTENDED_CPUID);
    for function in CPUID_LEAF_MAX_INPUT_FOR_EXTENDED_CPUID..=max_extended {
        let CpuidResult { eax, ebx, ecx, edx } = cpuid(function, 0).unwrap_or(CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        });
        let entry = VcpuCpuidEntry2 {
            function,
            index: 0,
            flags: 0,
            eax,
            ebx,
            ecx,
            edx,
            padding: [0; 3],
        };
        entries.push(entry);
    }

    entries.push(VcpuCpuidEntry2 {
        function: CPUID_LEAF_KVM_SIGNATURE,
        index: 0,
        flags: 0,
        eax: CPUID_LEAF_KVM_FEATURES,
        ebx: u32::from_le_bytes(*b"KVMK"),
        ecx: u32::from_le_bytes(*b"VMKV"),
        edx: u32::from_le_bytes(*b"M\0\0\0"),
        padding: [0; 3],
    });
    entries.push(VcpuCpuidEntry2 {
        function: CPUID_LEAF_KVM_FEATURES,
        index: 0,
        flags: 0,
        eax: KVM_FEATURE_CLOCKSOURCE | KVM_FEATURE_CLOCKSOURCE2,
        ebx: 0,
        ecx: 0,
        edx: 0,
        padding: [0; 3],
    });

    entries
}

pub(super) fn cpuid_entry(
    entries: &[VcpuCpuidEntry2],
    function: u32,
    index: u32,
) -> VcpuCpuidEntry2 {
    entries
        .iter()
        .copied()
        .find(|entry| entry_matches(*entry, function, index))
        .unwrap_or_default()
}

fn entry_matches(entry: VcpuCpuidEntry2, function: u32, index: u32) -> bool {
    if entry.function != function {
        return false;
    }
    entry.flags & GUEST_CPUID_FLAG_SIGNIFICANT_INDEX == 0 || entry.index == index
}

fn virtual_tsc_mhz() -> Option<u32> {
    let mhz = (tsc_freq().saturating_add(500_000)) / 1_000_000;
    u32::try_from(mhz).ok().filter(|&mhz| mhz != 0)
}

fn virtual_tsc_ratio() -> Option<(u32, u32, u32)> {
    let tsc_hz = tsc_freq();
    let divisor = gcd(tsc_hz, VIRTUAL_TSC_CRYSTAL_HZ);
    let denominator = VIRTUAL_TSC_CRYSTAL_HZ / divisor;
    let numerator = tsc_hz / divisor;
    Some((
        u32::try_from(denominator)
            .ok()
            .filter(|&value| value != 0)?,
        u32::try_from(numerator).ok().filter(|&value| value != 0)?,
        u32::try_from(VIRTUAL_TSC_CRYSTAL_HZ).ok()?,
    ))
}

fn gcd(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}
