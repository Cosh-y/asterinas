use super::{
    ioctl::{KVM_CPUID_FLAG_SIGNIFCANT_INDEX, KvmCpuidEntry2},
    msr::APIC_ENABLE,
    vcpu::VcpuState,
};
use crate::{kvm::vcpu::Vcpu, prelude::*};

pub(super) struct VcpuCpuid {
    pub entries: Vec<KvmCpuidEntry2>,
}

impl VcpuCpuid {
    pub(super) fn find(&self, function: u32, index: u32) -> Option<&KvmCpuidEntry2> {
        self.entries.iter().find(|entry| {
            entry.function == function
                && (entry.flags & KVM_CPUID_FLAG_SIGNIFCANT_INDEX == 0 || entry.index == index)
        })
    }

    pub(super) fn lookup(&self, function: u32, index: u32) -> Option<&KvmCpuidEntry2> {
        if let Some(entry) = self.find(function, index) {
            return Some(entry);
        }

        let basic = self.find(0, 0)?;

        let class = match function {
            // Basic class.
            0x0000_0000..=0x3fff_ffff => 0,
            // Hypervisor sub-classes, each spanning 0x100 function numbers.
            0x4000_0000..=0x4fff_ffff => function & 0xffff_ff00,
            // Basic class.
            0x5000_0000..=0x7fff_ffff => 0,
            // Extended class.
            0x8000_0000..=0xbfff_ffff => 0x8000_0000,
            // All functions >= 0xc000_0000 under the Centaur base.
            0xc000_0000..=0xffff_ffff => 0xc000_0000,
        };

        if self
            .find(class, 0)
            .is_some_and(|entry| function <= entry.eax)
        {
            return None;
        }

        self.find(basic.eax, index)
    }
}

// `CPUID.01H:EDX`.
const CPUID_FPU: u32 = 1 << 0;
const CPUID_PSE: u32 = 1 << 3;
const CPUID_TSC: u32 = 1 << 4;
const CPUID_MSR: u32 = 1 << 5;
const CPUID_PAE: u32 = 1 << 6;
const CPUID_CX8: u32 = 1 << 8;
const CPUID_APIC: u32 = 1 << 9;
const CPUID_SYSENTER_SYSEXIT: u32 = 1 << 11;
const CPUID_PGE: u32 = 1 << 13;
const CPUID_CMOV: u32 = 1 << 15;
const CPUID_PAT: u32 = 1 << 16;
const CPUID_PSE36: u32 = 1 << 17;
const CPUID_MMX: u32 = 1 << 23;
const CPUID_FXSR: u32 = 1 << 24;
const CPUID_SSE: u32 = 1 << 25;
const CPUID_SSE2: u32 = 1 << 26;

// `CPUID.01H:ECX`.
const CPUID_X2APIC: u32 = 1 << 21;
const CPUID_HYPERVISOR: u32 = 1 << 31;

// `CPUID.80000001H:ECX`.
const CPUID_LAHF_SAHF_64: u32 = 1 << 0;

// `CPUID.80000001H:EDX`.
const CPUID_SYSCALL_SYSRET: u32 = 1 << 11;
pub(super) const CPUID_NX: u32 = 1 << 20;
pub(super) const CPUID_LM: u32 = 1 << 29;

pub(super) fn supported_cpuid() -> [KvmCpuidEntry2; 8] {
    const BASIC_EDX: u32 = CPUID_FPU
        | CPUID_PSE
        | CPUID_TSC
        | CPUID_MSR
        | CPUID_PAE
        | CPUID_CX8
        | CPUID_APIC
        | CPUID_SYSENTER_SYSEXIT
        | CPUID_PGE
        | CPUID_CMOV
        | CPUID_PAT
        | CPUID_PSE36
        | CPUID_MMX
        | CPUID_FXSR
        | CPUID_SSE
        | CPUID_SSE2;

    const EXTENDED_EDX: u32 = CPUID_SYSCALL_SYSRET | CPUID_NX | CPUID_LM;

    let vendor = cpuid::cpuid(0, 0).unwrap();
    let basic = cpuid::cpuid(1, 0).unwrap();
    let extended = cpuid::cpuid(0x8000_0001, 0).unwrap();

    [
        // Maximum basic CPUID leaf and CPU vendor ID.
        KvmCpuidEntry2 {
            function: 0,
            eax: 7,
            ebx: vendor.ebx,
            ecx: vendor.ecx,
            edx: vendor.edx,
            ..Default::default()
        },
        // Processor signature, initial APIC ID, and basic feature bits.
        KvmCpuidEntry2 {
            function: 1,
            eax: basic.eax,
            // One logical processor; userspace supplies the guest APIC ID.
            ebx: 1 << 16,
            ecx: CPUID_X2APIC | CPUID_HYPERVISOR,
            edx: basic.edx & BASIC_EDX,
            ..Default::default()
        },
        // Structured extended feature bits and maximum subleaf index.
        KvmCpuidEntry2 {
            function: 7,
            flags: KVM_CPUID_FLAG_SIGNIFCANT_INDEX,
            ..Default::default()
        },
        // KVM hypervisor signature and maximum hypervisor CPUID leaf.
        KvmCpuidEntry2 {
            function: 0x4000_0000,
            eax: 0x4000_0001,
            ebx: u32::from_le_bytes(*b"KVMK"),
            ecx: u32::from_le_bytes(*b"VMKV"),
            edx: u32::from_le_bytes(*b"M\0\0\0"),
            ..Default::default()
        },
        // KVM paravirtual feature bits; none are exposed.
        KvmCpuidEntry2 {
            function: 0x4000_0001,
            ..Default::default()
        },
        // Maximum extended CPUID leaf.
        KvmCpuidEntry2 {
            function: 0x8000_0000,
            eax: 0x8000_0008,
            ..Default::default()
        },
        // Extended processor feature bits, including NX and long mode.
        KvmCpuidEntry2 {
            function: 0x8000_0001,
            ecx: extended.ecx & CPUID_LAHF_SAHF_64,
            edx: extended.edx & EXTENDED_EDX,
            ..Default::default()
        },
        // Physical and linear address widths.
        KvmCpuidEntry2 {
            function: 0x8000_0008,
            eax: 36 | (48 << 8),
            ..Default::default()
        },
    ]
}

impl Vcpu {
    pub(super) fn update_cpuid_runtime(&self, state: &mut VcpuState) {
        let apic_base = state.guest_context.sregs().apic_base;
        if let Some(entry) = state
            .cpuid
            .entries
            .iter_mut()
            .find(|entry| entry.function == 1)
        {
            entry.edx &= !CPUID_APIC;
            if apic_base & APIC_ENABLE != 0 {
                entry.edx |= CPUID_APIC;
            }
        }
    }
}
