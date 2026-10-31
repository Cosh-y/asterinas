// SPDX-License-Identifier: MPL-2.0

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
    device::hypervisor::ioctl,
    dispatch_ioctl,
    kvm::{
        run_page::KvmRunPage,
        vcpu::{Vcpu, VcpuState},
    },
    prelude::*,
    process::{posix_thread::AsPosixThread, signal::HandlePendingSignal},
    util::ioctl::RawIoctl,
};

pub(in crate::kvm) struct VcpuArch {
    state: Mutex<VcpuState>,

    // lapic, 内部处理同步访问
    lapic: Lapic,

    tsc_offset: u64,
}

impl VcpuArch {
    pub(in crate::kvm) fn new() -> Self {
        Self {

        }
    }

    pub(super) fn lapic(&self) -> &Lapic {
        &self.lapic
    }
}

struct VcpuState {
    pub guest_context: GuestContext,
    pub pending_io: Option<PendingIo>,
}

pub(in crate::kvm) struct PendingIo {
    rip: usize,
    instruction_len: u32,
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
                let rip = state
                    .guest_context
                    .regs()
                    .rip
                    .checked_add(exit_info.instruction_len as usize)
                    .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
                // run_page.write_run_val(ioctl::KVM_RUN_EXIT_REASON_OFFSET, &ioctl::KVM_EXIT_HLT)?;
                state.guest_context.regs_mut().rip = rip;

                // 这里应该判断 guest 是否设置了 lapic timer
                // 如果是应该设置定时器，到期时向 Lapic 注入时钟中断并调用 vcpu.notify

                // 唤醒的条件
                // 1. signal: pause_until 自动保证
                // 2. guest interrupt: 新的中断进入 LAPIC IRR 并且能进入 ISR
                self.wait_queue().pause_until(|| {
                    self.arch().lapic.quiry_irq()
                })?;

                // Ok(true)
                Ok(false)
            }
            Ok(VmxExitReason::MSR_READ) => {
                let regs = state.guest_context.regs_mut();
                let msr = regs.rcx as u32;
                let value = match msr {
                    // IA32_APIC_BASE
                    0x1B => ,
                    0x800..=0x8FF => {
                        self.arch().lapic.emulate_msr_read(msr, self.guest_tsc())?
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
                        self.arch().lapic.emulate_msr_write(msr, value, self.guest_tsc())?;
                    }
                }
                Ok(false)
            }
            Ok(VmxExitReason::PREEMPTION_TIMER) => {
                // 判断定时器是否到时
                // 设置时钟中断

                Ok(false)
            },
            _ => return_errno_with_message!(Errno::EOPNOTSUPP, "unsupported VM exit"),
        }
    }

    pub(in crate::kvm) fn finish_pending(&self, state: &mut VcpuState) {
        if let Some(pending_io) = state.pending_io.take() {
            let rip = state.guest_context.regs().rip;
            if rip == pending_io.rip {
                state.guest_context.regs_mut().rip += pending_io.instruction_len as usize;
            }
        }
    }

    pub(super) fn guest_tsc(&self) -> u64 {
        ostd::arch::read_tsc() + self.arch().tsc_offset
    }
}

impl GuestModeHooks for Vcpu {
    fn pre_guest_run(&self, state: &mut VcpuArchState, guard: &DisabledLocalIrqGuard) {
        *self.cpu() = Some(guard.current_cpu());
        // Prepare VMX preempt timer.
        let cur_guest_tsc = self.guest_tsc();
        self.arch().lapic.process_timer_expire(cur_guest_tsc);

        state.timer_deadline = self.arch().lapic.timer()
            .quiry_deadline_tsc()
            .map(|tsc| GuestTimerInstant { tsc });

        // Prepare interrupt injection.
        state.request_interrupt_window = false;

        if state.pending_interrupt.is_some() {
            state.request_interrupt_window = !state.is_interruptable;
            return;
        }
        
        if state.is_interruptable {
            let vector = self.arch().lapic.inject_irq();
            if let Some(vector) = vector {
                state.pending_interrupt = Some(GuestInterrupt { vector });
            }
        } else {
            state.request_interrupt_window = self.arch().lapic.quiry_irq().is_some();
        }
    }

    fn post_guest_run(&self, _state: &mut VcpuArchState, _guard: &DisabledLocalIrqGuard) {
        *self.cpu() = None;

        // // 检查是否要注入时钟中断，如果是，通过 lapic 注入
        // let cur_guest_tsc = self.guest_tsc();
        // self.arch().lapic.process_timer_expire(cur_guest_tsc);
    }

    fn has_kernel_event(&self) -> bool {
        current_thread!().as_posix_thread().unwrap().has_pending()
    }
}
