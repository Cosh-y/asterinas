// SPDX-License-Identifier: MPL-2.0

use core::time::Duration;

use ostd::{
    arch::vm::{
        GuestContext, GuestExitInfo, GuestInterrupt, GuestTimerInstant, VcpuArchState, VcpuMsrs, VcpuRegs, VcpuSregs, VmxExitReason,
    }, irq::DisabledLocalIrqGuard, vm::GuestModeHooks,
};

use super::{
    ioctl::{GetRegs, GetSregs, KvmRegs, KvmSregs, SetRegs, SetSregs},
    apic::Lapic,
};
use crate::{
    device::kvm::ioctl, dispatch_ioctl, kvm::{
        arch::apic::LapicEvent::Eoi2Ioapic, run_page::KvmRunPage, vcpu::{Vcpu, VcpuState},
    }, prelude::*, process::{posix_thread::AsPosixThread, signal::HandlePendingSignal}, util::ioctl::RawIoctl,
};

pub(in crate::kvm) struct VcpuArch {
    state: Mutex<VcpuState>,

    // lapic, 内部处理同步访问
    lapic: SpinLock<Lapic>,

    // TODO:
    // tsc_offset: u64,
}

impl VcpuArch {
    pub(in crate::kvm) fn new(id: u32) -> Result<Self> {
        Ok(Self {
            state: Mutex::new(VcpuState {
                guest_context: Vcpu::new_guest_context()?,
                pending_io: None,
                halted: false,
            }),
            lapic: SpinLock::new(Lapic::new(id)),
        })
    }

    pub(in crate::kvm) fn state(&self) -> MutexGuard<'_, VcpuState> {
        self.state.lock()
    }

    pub(super) fn lapic(&self) -> SpinLockGuard<'_, Lapic> {
        self.lapic.lock()
    }
}

struct VcpuState {
    pub guest_context: GuestContext,
    pub pending_io: Option<PendingIo>,
    pub halted: bool,
}

pub(in crate::kvm) struct PendingIo {
    rip: usize,
    instruction_len: u32,
}

impl VcpuState {
    fn skip_emulated_instruction(&mut self, instruction_len: u32) -> Result<()> {
        let rip = self
            .guest_context
            .regs()
            .rip
            .checked_add(instruction_len as usize)
            .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
        self.guest_context.regs_mut().rip = rip;
        self.guest_context.clear_interrupt_block();
        Ok(())
    }
}

macro_rules! write_run_fields {
    ($run_page:ident; $( [$offset:expr, $value:expr] ),+ $(,)?) => {
        $( $run_page.write_run_val($offset, &$value)?; )+
    };
}

impl Vcpu {
    pub(crate) fn ioctl(&self, raw_ioctl: RawIoctl) -> Result<i32> {
        dispatch_ioctl!(match raw_ioctl {
            cmd @ GetRegs => {
                let regs = KvmRegs::from(*self.state().guest_context.regs());
                cmd.write(&regs)?;
                Ok(0)
            }
            cmd @ SetRegs => {
                let regs = cmd.read()?;
                *self.state().guest_context.regs_mut() = regs.into();
                Ok(0)
            }
            cmd @ GetSregs => {
                let sregs = KvmSregs::from(*self.state().guest_context.sregs());
                cmd.write(&sregs)?;
                Ok(0)
            }
            cmd @ SetSregs => {
                let sregs = cmd.read()?;
                *self.state().guest_context.sregs_mut() = sregs.into();
                Ok(0)
            }
            
            _ => {
                return_errno_with_message!(Errno::ENOTTY, "unsupported vCPU ioctl")
            }
        })
    }

    pub(in crate::kvm) fn new_guest_context() -> Result<GuestContext> {
        Ok(GuestContext::new(
            VcpuRegs::new(),
            VcpuSregs::new_real_mode(),
            VcpuMsrs::default(),
        )?)
    }

    /// Handles a VM exit and updates the guest state and run page.
    ///
    /// Returns `Ok(true)` to return to userspace, `Ok(false)` to resume the guest,
    /// or an error if the exit cannot be handled.
    pub(in crate::kvm) fn handle_exit(
        &self,
        state: &mut VcpuState,
        exit_info: GuestExitInfo,
        run_page: &KvmRunPage,
    ) -> Result<bool> {
        match VmxExitReason::try_from(exit_info.exit_reason) {
            Ok(VmxExitReason::IO_INSTRUCTION) => {
                const IO_IN: u64 = 1 << 3;
                const IO_STRING: u64 = 1 << 4;
                const IO_REP: u64 = 1 << 5;
                const PIO_DATA_OFFSET: usize = PAGE_SIZE;

                if exit_info.exit_qualification & (IO_IN | IO_STRING | IO_REP) != 0 {
                    return_errno_with_message!(
                        Errno::EOPNOTSUPP,
                        "only scalar port output is supported currently"
                    );
                }

                let size = ((exit_info.exit_qualification & 7) + 1) as u8;
                run_page.write_run_bytes(
                    PIO_DATA_OFFSET,
                    &state.guest_context.regs().rax.to_le_bytes()[..usize::from(size)],
                )?;
                write_run_fields!(run_page;
                    [ioctl::KVM_RUN_IO_DIRECTION_OFFSET, ioctl::KVM_EXIT_IO_OUT],
                    [ioctl::KVM_RUN_IO_SIZE_OFFSET, size],
                    [ioctl::KVM_RUN_IO_PORT_OFFSET, (exit_info.exit_qualification >> 16) as u16],
                    [ioctl::KVM_RUN_IO_COUNT_OFFSET, 1u32],
                    [ioctl::KVM_RUN_IO_DATA_OFFSET_OFFSET, PIO_DATA_OFFSET as u64],
                    [ioctl::KVM_RUN_EXIT_REASON_OFFSET, ioctl::KVM_EXIT_IO],
                );

                // Complete the instruction when userspace re-enters `KVM_RUN`.
                state.pending_io = Some(PendingIo {
                    rip: state.guest_context.regs().rip,
                    instruction_len: exit_info.instruction_len,
                });
                Ok(true)
            }
            Ok(VmxExitReason::EPT_VIOLATION) => {
                // MMIO 处理
                self.vm()
                    .memory()
                    .handle_page_fault(exit_info.guest_phys_addr)?;
                Ok(false)
            }
            Ok(VmxExitReason::HLT) => {
                state.skip_emulated_instruction(exit_info.instruction_len)?;
                state.halted = true;

                self.wait_runnable(state)?;

                Ok(false)
            }
            Ok(VmxExitReason::MSR_READ) => {
                let regs = state.guest_context.regs_mut();
                let msr = regs.rcx as u32;
                let value = match msr {
                    // IA32_APIC_BASE
                    0x1B => ,
                    0x800..=0x8FF => {
                        self.arch().lapic().emulate_msr_read(msr, self.guest_tsc())?
                    }
                };
                regs.rax = value as u32 as usize;
                regs.rdx = (value >> 32) as u32 as usize;
                Ok(false)
            }
            Ok(VmxExitReason::MSR_WRITE) => {
                let regs = state.guest_context.regs();
                let msr = regs.rcx as u32;
                let value = ((regs.rdx as u32 as u64) << 32)
                    | (regs.rax as u32 as u64);
                match msr {
                    // IA32_APIC_BASE
                    0x1B => ,
                    0x800..=0x8FF => {
                        let lapic_event = self.arch().lapic().emulate_msr_write(msr, value, self.guest_tsc())?;
                        if let Some(lapic_event) = lapic_event {
                            match lapic_event {
                                Eoi2Ioapic { vector } => {
                                    self.vm().arch().ioapic().receive_eoi(vector);
                                }
                            }
                        }
                    }
                }
                Ok(false)
            }
            _ => return_errno_with_message!(Errno::EOPNOTSUPP, "unsupported VM exit"),
        }
    }

    /// Continues a pending operation after userspace has handled the previous exit.
    ///
    /// Returns `Ok(true)` to continue processing `KVM_RUN`, or `Ok(false)` if a new
    /// exit has been prepared in `run_page` for userspace to handle.
    pub(in crate::kvm) fn finish_pending(&self, state: &mut VcpuState, _run_page: &KvmRunPage) -> Result<bool> {
        if let Some(pending_io) = state.pending_io.as_ref() {
            let rip = state.guest_context.regs().rip;
            if rip == pending_io.rip {
                state.skip_emulated_instruction(pending_io.instruction_len)?;
            }
            state.pending_io = None;
        }

        Ok(true)
    }

    pub(in crate::kvm) fn wait_runnable(&self, state: &mut VcpuState) -> Result<()> {
        if state.halted {
            let interrupts_enabled =
                state.guest_context.regs().rflags & (1 << 9) != 0;
            loop {
                let deadline_tsc = self.arch().lapic().timer().quiry_deadline_tsc();
                let cur_guest_tsc = self.guest_tsc();
                let mut duration = None;
                if let Some(deadline_tsc) = deadline_tsc {
                    if deadline_tsc > cur_guest_tsc {
                        duration = Some(Self::tsc_ticks_to_duration(deadline_tsc - cur_guest_tsc));
                    } else {
                        self.arch().lapic().process_timer_expire(cur_guest_tsc);
                    }
                }

                // Conditions for waking up:
                // 1. A signal is sent to this vCPU thread, ensured by `pause_until`.
                // 2. This vCPU received a guest interrupt.
                let result = self.wait_queue().pause_until_or_timeout(|| {
                    if interrupts_enabled {
                        self.arch().lapic().quiry_irq()
                    } else {
                        // Normal interrupts are masked.
                        // Other interrupts (NMI...) are currently not implemented.
                        None
                    }
                }, duration.as_ref());

                match result {
                    Ok(_) => break,
                    Err(err) if err.error() == Errno::ETIME => continue,
                    Err(err) => return Err(err),
                }
            }

            state.halted = false;
        }

        Ok(())
    }

    pub(super) fn guest_tsc(&self) -> u64 {
        ostd::arch::read_tsc()
    }

    fn tsc_ticks_to_duration(ticks: u64) -> Duration {
        let freq = ostd::arch::tsc_freq();

        let secs = ticks / freq;
        let nanos = ((ticks % freq) as u128 * 1_000_000_000).div_ceil(freq as u128);

        Duration::from_secs(secs) + Duration::from_nanos(nanos as u64)
    }
}

impl GuestModeHooks for Vcpu {
    fn pre_guest_run(&self, state: &mut VcpuArchState, guard: &DisabledLocalIrqGuard) {
        *self.cpu() = Some(guard.current_cpu());
        // Prepare VMX preempt timer.
        self.arch().lapic().process_timer_expire(self.guest_tsc());

        state.timer_deadline = self.arch().lapic().timer()
            .quiry_deadline_tsc()
            .map(|tsc| GuestTimerInstant { tsc });

        // Prepare interrupt injection.
        state.request_interrupt_window = false;

        if state.pending_interrupt.is_some() {
            state.request_interrupt_window = !state.is_interruptable;
            return;
        }
        
        if state.is_interruptable {
            let vector = self.arch().lapic().inject_irq();
            if let Some(vector) = vector {
                state.pending_interrupt = Some(GuestInterrupt { vector });
            }
        } else {
            state.request_interrupt_window = self.arch().lapic().quiry_irq().is_some();
        }
    }

    fn post_guest_run(&self, _state: &mut VcpuArchState, _guard: &DisabledLocalIrqGuard) {
        *self.cpu() = None;
    }

    fn has_kernel_event(&self) -> bool {
        current_thread!().as_posix_thread().unwrap().has_pending()
    }
}
