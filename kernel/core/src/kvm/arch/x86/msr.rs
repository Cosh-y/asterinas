use x86_64::registers::{control::Cr0Flags, model_specific::EferFlags};

use super::{
    apic::LapicEvent::Eoi2Ioapic,
    cpuid::{CPUID_LM, CPUID_NX},
    vcpu::VcpuState,
};
use crate::{kvm::vcpu::Vcpu, prelude::*};

pub(super) enum MsrAccess {
    Userspace,
    Guest,
}

const IA32_TSC: u32 = 0x10;
const IA32_APIC_BASE: u32 = 0x1b;
const IA32_SYSENTER_CS: u32 = 0x174;
const IA32_SYSENTER_ESP: u32 = 0x175;
const IA32_SYSENTER_EIP: u32 = 0x176;
const IA32_MISC_ENABLE: u32 = 0x1a0;
const IA32_PAT: u32 = 0x277;

const EFER: u32 = 0xc000_0080;
const STAR: u32 = 0xc000_0081;
const LSTAR: u32 = 0xc000_0082;
const CSTAR: u32 = 0xc000_0083;
const FMASK: u32 = 0xc000_0084;
const FS_BASE: u32 = 0xc000_0100;
const GS_BASE: u32 = 0xc000_0101;
const KERNEL_GS_BASE: u32 = 0xc000_0102;

pub(super) const APIC_ENABLE: u64 = 1 << 11;
const X2APIC_ENABLE: u64 = 1 << 10;

impl Vcpu {
    pub(super) fn read_msr(&self, state: &VcpuState, index: u32) -> Result<u64> {
        let context = &state.guest_context;
        let msrs = context.msrs();
        let sregs = context.sregs();

        let value = match index {
            IA32_APIC_BASE => sregs.apic_base,
            IA32_TSC => self.guest_tsc(context.get_tsc_offset()),
            EFER => sregs.efer,
            FS_BASE => sregs.fs.base,
            GS_BASE => sregs.gs.base,
            IA32_SYSENTER_CS => msrs.sysenter_cs,
            IA32_SYSENTER_ESP => msrs.sysenter_esp,
            IA32_SYSENTER_EIP => msrs.sysenter_eip,
            IA32_PAT => msrs.pat,
            STAR => msrs.star,
            LSTAR => msrs.lstar,
            CSTAR => 0,
            FMASK => msrs.syscall_mask,
            KERNEL_GS_BASE => msrs.kernel_gs_base,
            IA32_MISC_ENABLE => msrs.misc_enable,
            0x800..=0x8ff => {
                if sregs.apic_base & (APIC_ENABLE | X2APIC_ENABLE) != APIC_ENABLE | X2APIC_ENABLE {
                    return_errno!(Errno::EINVAL);
                }

                return self
                    .arch()
                    .lapic()
                    .emulate_msr_read(
                        index,
                        self.guest_tsc(state.guest_context.get_tsc_offset())
                    );
            }
            _ => return_errno!(Errno::EINVAL),
        };

        Ok(value)
    }

    pub(super) fn write_msr(
        &self,
        state: &mut VcpuState,
        index: u32,
        value: u64,
        access: MsrAccess,
    ) -> Result<()> {
        match index {
            IA32_APIC_BASE => self.set_apic_base(state, value, access)?,
            IA32_TSC => self.set_tsc(state, value, access)?,
            EFER => Self::set_efer(state, value, access)?,
            IA32_SYSENTER_CS => state.guest_context.msrs_mut().sysenter_cs = value,
            IA32_SYSENTER_ESP => state.guest_context.msrs_mut().sysenter_esp = value,
            IA32_SYSENTER_EIP => state.guest_context.msrs_mut().sysenter_eip = value,
            IA32_PAT
                if value
                    .to_le_bytes()
                    .iter()
                    .all(|entry| matches!(*entry, 0 | 1 | 4..=7)) =>
            {
                state.guest_context.msrs_mut().pat = value
            }
            STAR => state.guest_context.msrs_mut().star = value,
            LSTAR => state.guest_context.msrs_mut().lstar = value,
            CSTAR => {},
            FMASK => state.guest_context.msrs_mut().syscall_mask = value,
            KERNEL_GS_BASE => state.guest_context.msrs_mut().kernel_gs_base = value,
            FS_BASE => state.guest_context.sregs_mut().fs.base = value,
            GS_BASE => state.guest_context.sregs_mut().gs.base = value,
            IA32_MISC_ENABLE => state.guest_context.msrs_mut().misc_enable = value,
            0x800..=0x8ff => {
                if state.guest_context.sregs().apic_base & (APIC_ENABLE | X2APIC_ENABLE)
                    != APIC_ENABLE | X2APIC_ENABLE
                {
                    return_errno!(Errno::EINVAL);
                }

                let event =
                    self.arch()
                        .lapic()
                        .emulate_msr_write(
                            index,
                            value,
                            self.guest_tsc(state.guest_context.get_tsc_offset())
                        )?;

                if let Some(Eoi2Ioapic { vector }) = event {
                    self.vm().arch().ioapic().receive_eoi(vector)?;
                }
            }
            _ => return_errno!(Errno::EINVAL),
        }

        Ok(())
    }

    fn set_apic_base(&self, state: &mut VcpuState, value: u64, access: MsrAccess) -> Result<()> {
        let mode_mask = APIC_ENABLE | X2APIC_ENABLE;
        let new_mode = value & mode_mask;

        // Reject EN=0, EXTD=1 for both access paths.
        if new_mode == X2APIC_ENABLE {
            return_errno!(Errno::EINVAL);
        }

        if let MsrAccess::Guest = access {
            let old_mode = state.guest_context.sregs().apic_base & mode_mask;

            // WRMSR cannot transition directly from x2APIC to xAPIC.
            if old_mode == mode_mask && new_mode == APIC_ENABLE {
                return_errno!(Errno::EINVAL);
            }

            // WRMSR cannot transition directly from disabled to x2APIC.
            if old_mode == 0 && new_mode == mode_mask {
                return_errno!(Errno::EINVAL);
            }
        }

        state.guest_context.sregs_mut().apic_base = value;
        self.arch().lapic().set_apic_base(value);

        self.update_cpuid_runtime(state);

        Ok(())
    }

    fn set_tsc(&self, state: &mut VcpuState, value: u64, access: MsrAccess) -> Result<()> {
        // guest_tsc = host_tsc.wrapping_add(tsc_offset);
        // tsc_offset = requested_tsc.wrapping_sub(host_tsc_now);
        let tsc_offset = value.wrapping_sub(ostd::arch::read_tsc());
        let old_tsc_offset = state.guest_context.get_tsc_offset();
        state.guest_context.set_tsc_offset(tsc_offset);

        self.arch().lapic().timer_mut().shift_deadline_tsc(
            tsc_offset.wrapping_sub(old_tsc_offset)
        );

        Ok(())
    }

    fn set_efer(state: &mut VcpuState, value: u64, access: MsrAccess) -> Result<()> {
        let lme = EferFlags::LONG_MODE_ENABLE.bits();
        let lma = EferFlags::LONG_MODE_ACTIVE.bits();
        let nxe = EferFlags::NO_EXECUTE_ENABLE.bits();
        if value & !(EferFlags::SYSTEM_CALL_EXTENSIONS.bits() | lme | lma | nxe) != 0 {
            return_errno!(Errno::EINVAL);
        }

        let sregs = state.guest_context.sregs();
        if let MsrAccess::Guest = access {
            let cpuid_edx = state
                .cpuid_entries()
                .iter()
                .find(|entry| entry.function == 0x8000_0001)
                .map_or(0, |entry| entry.edx);
            if value & (lme | lma) != 0 && cpuid_edx & CPUID_LM == 0 {
                return_errno!(Errno::EINVAL);
            }
            if value & nxe != 0 && cpuid_edx & CPUID_NX == 0 {
                return_errno!(Errno::EINVAL);
            }
            if sregs.cr0 & Cr0Flags::PAGING.bits() != 0 && (sregs.efer ^ value) & lme != 0 {
                return_errno!(Errno::EINVAL);
            }
        }

        // Both `WRMSR` and `KVM_SET_MSRS` preserve `LMA`.
        let value = (value & !lma) | (sregs.efer & lma);
        state.guest_context.sregs_mut().efer = value;

        Ok(())
    }
}
