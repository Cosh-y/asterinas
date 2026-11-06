// SPDX-License-Identifier: MPL-2.0

use core::{arch, time::Duration};

use ostd::{
    arch::vm::{
        self, GuestContext, GuestExitInfo, GuestInterrupt, GuestTimerInstant, VcpuArchState, VcpuMsrs,
        VcpuRegs, VcpuSregs, VmxExitReason,
    },
    irq::DisabledLocalIrqGuard, vm::GuestModeHooks,
};

use super::{
    apic::{DeliveryMode, Lapic},
    cpuid::VcpuCpuid,
    fpu::VcpuFpu,
    mmio_decode::{MmioAccess, MmioOp},
    msr::MsrAccess,
    ioctl::{
        GetFpu, GetRegs, GetSregs, GetMsrs, KvmCpuid2, KvmCpuidEntry2, KvmRegs, KvmSregs, SetCpuid2,
        SetFpu, SetMsrs, SetRegs, SetSregs, KvmMsrs, KvmMsrEntry, read_trailing_array, write_trailing_array,
        PIO_DATA_OFFSET,
    },
    vm::IrqchipMode,
};
use crate::{
    device::kvm::ioctl::{self, KVM_RUN_MMIO_DATA_OFFSET}, dispatch_ioctl, kvm::{arch::apic::LapicEvent::Eoi2Ioapic, run_page::KvmRunPage, vcpu::Vcpu, vm::Vm}, prelude::*, process::{posix_thread::AsPosixThread, signal::HandlePendingSignal}, util::ioctl::RawIoctl,
};

pub(in crate::kvm) struct VcpuArch {
    state: Mutex<VcpuState>,

    // lapic, 内部处理同步访问
    lapic: SpinLock<Option<Lapic>>,

    // TODO:
    // fpu 不能像 lapic 那样在 vcpu run 期间被异步访问
    // 理想情况下 fpu 应放在 VcpuState 中，但为了 pre post hook 能访问 fpu 值
    // 目前只能把他放在这里
    // Lock order: `state` -> `fpu`.
    fpu: SpinLock<VcpuFpu>,
}

impl VcpuArch {
    pub(in crate::kvm) fn new(id: u32, vm: &Vm) -> Result<Self> {
        let lapic = if vm.arch().irqchip_mode() != IrqchipMode::None {
            Some(Lapic::new(id))
        } else {
            None
        };
        
        Ok(Self {
            state: Mutex::new(VcpuState {
                guest_context: Vcpu::new_guest_context()?,
                cpuid: VcpuCpuid::default(),
                pending_io: None,
                pending_mmio: None,
                halted: false,
            }),
            lapic: SpinLock::new(lapic),
            fpu: SpinLock::new(VcpuFpu::new()),
        })
    }

    pub(in crate::kvm) fn state(&self) -> MutexGuard<'_, VcpuState> {
        self.state.lock()
    }

    pub(super) fn lapic(&self) -> SpinLockGuard<'_, Lapic> {
        self.lapic.lock()
    }
}

pub(super) struct VcpuState {
    pub guest_context: GuestContext,
    pub cpuid: VcpuCpuid,

    pub pending_io: Option<PendingIo>,
    pub pending_mmio: Option<(usize, MmioOp)>,

    pub halted: bool,
}

pub(in crate::kvm) struct PendingIo {
    rip: usize,
    instruction_len: u32,
    is_in: bool,
    size: u8,
}

impl VcpuState {
    pub(super) fn cpuid_entries(&self) -> &[KvmCpuidEntry2] {
        &self.cpuid.entries
    }

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
                let regs = KvmRegs::from(*self.arch().state().guest_context.regs());
                cmd.write(&regs)?;
                Ok(0)
            }
            cmd @ SetRegs => {
                let regs = cmd.read()?;
                *self.arch().state().guest_context.regs_mut() = regs.into();
                Ok(0)
            }
            cmd @ GetSregs => {
                let sregs = KvmSregs::from(*self.arch().state().guest_context.sregs());
                cmd.write(&sregs)?;
                Ok(0)
            }
            cmd @ SetSregs => {
                let sregs = cmd.read()?;
                let mut state = self.arch().state();
                *state.guest_context.sregs_mut() = sregs.into();
                self.arch().lapic().set_apic_base(state.guest_context.sregs().apic_base);
                self.update_cpuid_runtime(&mut state);
                Ok(0)
            }
            cmd @ GetMsrs => {
                let mut entries: Vec<KvmMsrEntry> =
                    read_trailing_array(raw_ioctl, size_of::<KvmMsrs>(), cmd.read()?.nmsrs as usize)?;
                let mut state = self.arch().state();
                let mut cnt = 0;
                for entry in &mut entries {
                    let Ok(value) = self.read_msr(&state, entry.index) else {
                        break;
                    }
                    entry.data = value;
                    cnt += 1;
                }

                write_trailing_array(raw_ioctl, size_of::<KvmMsrs>(), &entries)?;

                Ok(cnt)
            }
            cmd @ SetMsrs => {
                let entries: Vec<KvmMsrEntry> =
                    read_trailing_array(raw_ioctl, size_of::<KvmMsrs>(), cmd.read()?.nmsrs as usize)?;
                let mut state = self.arch().state();
                let mut cnt = 0;

                for entry in entries {
                    if self
                        .write_msr(
                            &mut state,
                            entry.index,
                            entry.data,
                            MsrAccess::Userspace,
                        )
                        .is_err()
                    {
                        break;
                    }
                    cnt += 1;
                }

                Ok(cnt)
            }
            cmd @ GetFpu => {
                let _state = self.arch().state();
                let value = self.arch().fpu.lock().get();
                cmd.write(&value)?;
                Ok(0)
            }
            cmd @ SetFpu => {
                let _state = self.arch().state();
                let value = cmd.read()?;
                self.arch().fpu.lock().set(&value);
                Ok(0)
            }
            cmd @ SetCpuid2 => {
                let mut state = self.arch().state();
                state.cpuid.entries = 
                    read_trailing_array(raw_ioctl, size_of::<KvmCpuid2>(), cmd.read()?.nent)?;
                self.update_cpuid_runtime(&mut state);
                Ok(0)
            }
            _ => return_errno_with_message!(Errno::ENOTTY, "unsupported vCPU ioctl"),
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
            Ok(VmxExitReason::CPUID) => {
                let regs = state.guest_context.regs();
                let function = regs.rax as u32;
                let index = regs.rcx as u32;

                let mut entry = state
                    .cpuid
                    .lookup(function, index)
                    .copied()
                    .unwrap_or_default();

                let regs = state.guest_context.regs_mut();
                regs.rax = entry.eax as usize;
                regs.rbx = entry.ebx as usize;
                regs.rcx = entry.ecx as usize;
                regs.rdx = entry.edx as usize;

                state.skip_emulated_instruction(exit_info.instruction_len)?;
                Ok(false)
            }
            Ok(VmxExitReason::IO_INSTRUCTION) => {
                const IO_IN: u64 = 1 << 3;
                const IO_STRING: u64 = 1 << 4;
                const IO_REP: u64 = 1 << 5;

                if exit_info.exit_qualification & (IO_STRING | IO_REP) != 0 {
                    return_errno_with_message!(
                        Errno::EOPNOTSUPP,
                        "only scalar port output is supported currently"
                    );
                }

                let size = ((exit_info.exit_qualification & 7) + 1) as u8;
                let port = (exit_info.exit_qualification >> 16) as u16;

                // Port IO access kernel device.
                match port {
                    0x20..=0x21 | 0xa0..=0xa1 | 0x4d0..=0x4d1 => {
                        debug_assert!(size == 1);

                        let mut pic = self.vm().arch().pic();
                        if exit_info.exit_qualification & IO_IN == 0 {
                            pic.emulate_io_write(port, state.guest_context.regs().rax as u8);
                        } else {
                            let value = pic.emulate_io_read(port);
                            let rax = state.guest_context.regs().rax;
                            state.guest_context.regs_mut().rax = (rax & !0xff) | value as usize;
                        }

                        state.skip_emulated_instruction(exit_info.instruction_len)?;
                        return Ok(false);
                    }
                    0x40..=0x43 | 0x61 => {
                        debug_assert!(size == 1);

                        let mut pit = self.vm().arch().pit();
                        if exit_info.exit_qualification & IO_IN == 0 {
                            pit.emulate_io_write(port, state.guest_context.regs().rax as u8);
                        } else {
                            let value = pit.emulate_io_read(port);
                            let rax = state.guest_context.regs().rax;
                            state.guest_context.regs_mut().rax = (rax & !0xff) | value as usize;
                        }

                        state.skip_emulated_instruction(exit_info.instruction_len)?;
                        return Ok(false);
                    }
                }
                
                // Port IO should be handled in userspace.
                if exit_info.exit_qualification & IO_IN == 0 {
                    run_page.write_run_bytes(
                        PIO_DATA_OFFSET,
                        &state.guest_context.regs().rax.to_le_bytes()[..usize::from(size)],
                    )?;
                    write_run_fields!(run_page;
                        [ioctl::KVM_RUN_IO_DIRECTION_OFFSET, ioctl::KVM_EXIT_IO_OUT],
                        [ioctl::KVM_RUN_IO_SIZE_OFFSET, size],
                        [ioctl::KVM_RUN_IO_PORT_OFFSET, port],
                        [ioctl::KVM_RUN_IO_COUNT_OFFSET, 1u32],
                        [ioctl::KVM_RUN_IO_DATA_OFFSET_OFFSET, PIO_DATA_OFFSET as u64],
                        [ioctl::KVM_RUN_EXIT_REASON_OFFSET, ioctl::KVM_EXIT_IO],
                    );

                    state.pending_io = Some(PendingIo {
                        rip: state.guest_context.regs().rip,
                        instruction_len: exit_info.instruction_len,
                        is_in: false,
                        size,
                    });
                } else {
                    write_run_fields!(run_page;
                        [ioctl::KVM_RUN_IO_DIRECTION_OFFSET, ioctl::KVM_EXIT_IO_IN],
                        [ioctl::KVM_RUN_IO_SIZE_OFFSET, size],
                        [ioctl::KVM_RUN_IO_PORT_OFFSET, port],
                        [ioctl::KVM_RUN_IO_COUNT_OFFSET, 1u32],
                        [ioctl::KVM_RUN_IO_DATA_OFFSET_OFFSET, PIO_DATA_OFFSET as u64],
                        [ioctl::KVM_RUN_EXIT_REASON_OFFSET, ioctl::KVM_EXIT_IO],
                    );

                    state.pending_io = Some(PendingIo {
                        rip: state.guest_context.regs().rip,
                        instruction_len: exit_info.instruction_len,
                        is_in: true,
                        size,
                    });
                }
                Ok(true)
            }
            Ok(VmxExitReason::EPT_VIOLATION) => {
                // MMIO access kernel device.
                let ioapic_range = self.vm().arch().ioapic().mmio_range();
                if ioapic_range.contains(&exit_info.guest_phys_addr) {
                    let vm = self.vm();
                    let mut ioapic = vm.arch().ioapic();
                    let op = self.decode_mmio_op(state)?;
                    match &op.access {
                        MmioAccess::Read { reg } => {
                            let value = ioapic.emulate_mmio_read(exit_info.guest_phys_addr)?;
                            reg.write(state.guest_context.regs_mut(), value as usize, op.len);
                        }
                        MmioAccess::Write { data } => {
                            let value = u32::from_le_bytes(data[..4].try_into().unwrap());
                            ioapic.emulate_mmio_write(exit_info.guest_phys_addr, value)?;
                        }
                    }
                    state.skip_emulated_instruction((op.next_rip - state.guest_context.regs().rip) as u32)?;
                    return Ok(false);
                }

                // MMIO access RAM, handle page fault.
                if self
                    .vm()
                    .memory()
                    .handle_page_fault(exit_info.guest_phys_addr)?
                {
                    return Ok(false);
                }

                // MMIO should be handled in userspace.
                let op = self.decode_mmio_op(state)?;
                write_run_fields!(run_page;
                    [ioctl::KVM_RUN_MMIO_PHYS_ADDR_OFFSET, exit_info.guest_phys_addr],
                    [ioctl::KVM_RUN_MMIO_LEN_OFFSET, op.len as u32],
                    [ioctl::KVM_RUN_MMIO_IS_WRITE_OFFSET, false as u8],
                    [ioctl::KVM_RUN_EXIT_REASON_OFFSET, ioctl::KVM_EXIT_MMIO],
                );
                if let MmioAccess::Write { data } = &op.access {
                    write_run_fields!(run_page;
                        [ioctl::KVM_RUN_MMIO_DATA_OFFSET, *data],
                        [ioctl::KVM_RUN_MMIO_IS_WRITE_OFFSET, true as u8],
                    );
                }
                state.pending_mmio = Some((state.guest_context.regs().rip, op));
                
                Ok(true)
            }
            Ok(VmxExitReason::HLT) => {
                state.skip_emulated_instruction(exit_info.instruction_len)?;

                if self.vm().arch().irqchip_mode() == IrqchipMode::None {
                    run_page.write_run_val(
                        ioctl::KVM_RUN_EXIT_REASON_OFFSET,
                        &ioctl::KVM_EXIT_HLT,
                    )?;
                    return Ok(true);
                }

                state.halted = true;
                self.wait_runnable(state)?;

                Ok(false)
            }
            Ok(VmxExitReason::MSR_READ) => {
                let msr = state.guest_context.regs().rcx as u32;
                match self.read_msr(state, msr) {
                    Ok(value) => {
                        let regs = state.guest_context.regs_mut();
                        regs.rax = value as u32 as usize;
                        regs.rdx = (value >> 32) as u32 as usize;

                        state.skip_emulated_instruction(exit_info.instruction_len)?;
                    }
                    // TODO: 根据 read_msr 返回的 `Err(_)` 判断是否要注入 #GP.
                    Err(error) if error.error() == Errno::EINVAL => {
                        //
                    }
                    Err(error) => return Err(error),
                }

                Ok(false)
            }
            Ok(VmxExitReason::MSR_WRITE) => {
                let regs = state.guest_context.regs();
                let msr = regs.rcx as u32;
                let value = ((regs.rdx as u32 as u64) << 32)
                    | (regs.rax as u32 as u64);
                match self.write_msr(state, msr, value, MsrAccess::Guest) {
                    Ok(()) => {
                        state.skip_emulated_instruction(exit_info.instruction_len)?;
                    }
                    Err(error) if error.error() == Errno::EINVAL => {
                        //
                    }
                    Err(error) => return Err(error),
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
    pub(in crate::kvm) fn finish_pending(&self, state: &mut VcpuState, run_page: &KvmRunPage) -> Result<bool> {
        if let Some(pending_io) = state.pending_io.as_ref() {
            let rip = state.guest_context.regs().rip;
            if rip == pending_io.rip {
                state.skip_emulated_instruction(pending_io.instruction_len)?;
                if pending_io.is_in {
                    let rax = &mut state.guest_context.regs_mut().rax;
                    *rax = match pending_io.size {
                        1 => {
                            let value = run_page.read_run_val::<u8>(PIO_DATA_OFFSET)?;
                            (*rax & !0xff) | usize::from(value)
                        }
                        2 => {
                            let value = run_page.read_run_val::<u16>(PIO_DATA_OFFSET)?;
                            (*rax & !0xffff) | usize::from(value)
                        }
                        4 => {
                            let value = run_page.read_run_val::<u32>(PIO_DATA_OFFSET)?;
                            value as usize
                        }
                        _ => return_errno_with_message!(Errno::EINVAL, "invalid PIO size"),
                    };
                }
            }
            state.pending_io = None;
        }

        if let Some((pending_rip, pending_op)) = state.pending_mmio.as_ref() {
            let rip = state.guest_context.regs().rip;
            if rip == *pending_rip {
                if let MmioAccess::Read { reg } = pending_op.access {
                    let value = run_page.read_run_val::<u64>(KVM_RUN_MMIO_DATA_OFFSET)?;
                    reg.write(state.guest_context.regs_mut(), value as usize, pending_op.len);
                }
                state.skip_emulated_instruction((pending_op.next_rip - rip) as u32)?;
            }
            state.pending_mmio = None;
        }

        Ok(true)
    }

    pub(in crate::kvm) fn wait_runnable(&self, state: &mut VcpuState) -> Result<()> {
        if state.halted {
            let interrupts_enabled =
                state.guest_context.regs().rflags & (1 << 9) != 0;
            loop {
                let deadline_tsc = self.arch().lapic().timer().quiry_deadline_tsc();
                let cur_guest_tsc = self.guest_tsc(state.guest_context.get_tsc_offset());
                let mut duration_lapic = None;
                if let Some(deadline_tsc) = deadline_tsc {
                    let remaining = tsc_delta(deadline_tsc, cur_guest_tsc).max(0) as u64;
                    if remaining > 0 {
                        duration_lapic = Some(Self::tsc_ticks_to_duration(remaining));
                    } else {
                        self.arch().lapic().process_timer_expire(cur_guest_tsc);
                    }
                }

                let deadline_tsc = self.vm().arch().pit().quiry_deadline_tsc();
                let cur_tsc = ostd::arch::read_tsc();
                let mut duration_pit = None;
                if let Some(deadline_tsc) = deadline_tsc {
                    let remaining = tsc_delta(deadline_tsc, cur_tsc).max(0) as u64;
                    if remaining > 0 {
                        duration_pit = Some(Self::tsc_ticks_to_duration(remaining));
                    } else {
                        self.vm().arch().pit().process_timer_expire();
                    }
                }

                let duration = duration_lapic
                    .into_iter()
                    .chain(duration_pit)
                    .min();

                // Conditions for waking up:
                // 1. A signal is sent to this vCPU thread, ensured by `pause_until`.
                // 2. This vCPU received a guest interrupt.
                let result = self.wait_queue().pause_until_or_timeout(|| {
                    if !interrupts_enabled {
                        return None;
                    }

                    self.arch().lapic().can_inject_external_interrupt()
                        .then(|| self.vm().arch().pic().quiry_irq())
                        .flatten()
                        .or_else(|| self.arch().lapic().quiry_irq())
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

    pub(super) fn guest_tsc(&self, tsc_offset: u64) -> u64 {
        ostd::arch::read_tsc().wrapping_add(tsc_offset)
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

        {
            let mut fpu = self.arch().fpu.lock();
            fpu.host.save();
            fpu.guest.load();
        }

        // Prepare VMX preempt timer.
        self.arch().lapic().process_timer_expire(self.guest_tsc(state.tsc_offset));
        self.vm().arch().pit().process_timer_expire();

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
            if self.arch().lapic().can_inject_external_interrupt()
                && let Some(vector) = self.vm().arch().pic().inject_irq() {
                state.pending_interrupt = Some(GuestInterrupt { vector });
            } else if let Some(vector) = self.arch().lapic().inject_irq() {
                state.pending_interrupt = Some(GuestInterrupt { vector });
            }
        } else {
            state.request_interrupt_window = self.arch().lapic().quiry_irq().is_some()
                || (self.arch().lapic().can_inject_external_interrupt()
                    && self.vm().arch().pic().quiry_irq().is_some());
        }
    }

    fn post_guest_run(&self, _state: &mut VcpuArchState, _guard: &DisabledLocalIrqGuard) {
        *self.cpu() = None;

        {
            let mut fpu = self.arch().fpu.lock();
            fpu.guest.save();
            fpu.host.load();
        }
    }

    fn has_kernel_event(&self) -> bool {
        current_thread!().as_posix_thread().unwrap().has_pending()
    }
}
