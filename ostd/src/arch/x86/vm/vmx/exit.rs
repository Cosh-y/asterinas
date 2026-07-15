// SPDX-License-Identifier: MPL-2.0

use super::vmcs::{VmcsGuestNW, VmcsReadOnly32, VmcsReadOnly64, VmcsReadOnlyNW};
use crate::prelude::*;

type GuestPhysAddr = Gpaddr;

macro_rules! def_exit_reasons {
    (
        $( #[$meta:meta] )*
        pub enum $name:ident {
            $( $variant:ident = $val:expr ),* $(,)?
        }
    ) => {
        $( #[$meta] )*
        #[repr(u32)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $( $variant = $val ),*
        }

        impl core::convert::TryFrom<u32> for $name {
            type Error = u32;

            fn try_from(value: u32) -> core::result::Result<Self, Self::Error> {
                match value {
                    $( $val => Ok($name::$variant), )*
                    _ => Err(value),
                }
            }
        }
    };
}

def_exit_reasons! {
    #[expect(non_camel_case_types, reason = "VMX names follow Intel SDM terminology.")]
    /// VMX basic exit reasons. (SDM Vol. 3D, Appendix C)
    pub enum VmxExitReason {
        EXCEPTION_NMI = 0,
        EXTERNAL_INTERRUPT = 1,
        TRIPLE_FAULT = 2,
        INIT = 3,
        SIPI = 4,
        SMI = 5,
        OTHER_SMI = 6,
        INTERRUPT_WINDOW = 7,
        NMI_WINDOW = 8,
        TASK_SWITCH = 9,
        CPUID = 10,
        GETSEC = 11,
        HLT = 12,
        INVD = 13,
        INVLPG = 14,
        RDPMC = 15,
        RDTSC = 16,
        RSM = 17,
        VMCALL = 18,
        VMCLEAR = 19,
        VMLAUNCH = 20,
        VMPTRLD = 21,
        VMPTRST = 22,
        VMREAD = 23,
        VMRESUME = 24,
        VMWRITE = 25,
        VMOFF = 26,
        VMON = 27,
        CR_ACCESS = 28,
        DR_ACCESS = 29,
        IO_INSTRUCTION = 30,
        MSR_READ = 31,
        MSR_WRITE = 32,
        INVALID_GUEST_STATE = 33,
        MSR_LOAD_FAIL = 34,
        MWAIT_INSTRUCTION = 36,
        MONITOR_TRAP_FLAG = 37,
        MONITOR_INSTRUCTION = 39,
        PAUSE_INSTRUCTION = 40,
        MCE_DURING_VMENTRY = 41,
        TPR_BELOW_THRESHOLD = 43,
        APIC_ACCESS = 44,
        VIRTUALIZED_EOI = 45,
        GDTR_IDTR = 46,
        LDTR_TR = 47,
        EPT_VIOLATION = 48,
        EPT_MISCONFIG = 49,
        INVEPT = 50,
        RDTSCP = 51,
        PREEMPTION_TIMER = 52,
        INVVPID = 53,
        WBINVD = 54,
        XSETBV = 55,
        APIC_WRITE = 56,
        RDRAND = 57,
        INVPCID = 58,
        VMFUNC = 59,
        ENCLS = 60,
        RDSEED = 61,
        PML_FULL = 62,
        XSAVES = 63,
        XRSTORS = 64,
        PCONFIG = 65,
        SPP_EVENT = 66,
        UMWAIT = 67,
        TPAUSE = 68,
        LOADIWKEY = 69,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VmxExitInfo {
    pub(crate) entry_failure: bool,
    pub(crate) exit_reason: u32,
    pub(crate) instruction_len: u32,
    pub(crate) exit_qualification: u64,
    pub(crate) guest_phys_addr: GuestPhysAddr,
    pub(crate) guest_rip: GuestPhysAddr,
}

/// Reads the VM-exit information fields from the current VMCS.
pub(crate) fn exit_info() -> Result<VmxExitInfo> {
    let reason_raw = VmcsReadOnly32::EXIT_REASON.read()?;
    let entry_failure = (reason_raw & (1 << 31)) != 0;
    let exit_reason = reason_raw & 0x7FFF_FFFF;
    let instruction_len = VmcsReadOnly32::VMEXIT_INSTRUCTION_LEN.read().unwrap_or(0);
    let exit_qualification = VmcsReadOnlyNW::EXIT_QUALIFICATION.read()? as _;
    let guest_phys_addr = VmcsReadOnly64::GUEST_PHYSICAL_ADDR.read()?;
    let guest_rip = VmcsGuestNW::RIP.read()?;
    Ok(VmxExitInfo {
        entry_failure,
        exit_reason,
        instruction_len,
        exit_qualification,
        guest_phys_addr: guest_phys_addr as _,
        guest_rip: guest_rip as _,
    })
}
