use ostd::{
    arch::vm::{GuestContext, GuestExitInfo, VcpuRunState, VmxExitReason},
    task::Task,
    vm::{GuestMode, GuestRunResult},
};

use super::{
    apic::{Lapic, LapicPort, emulate_apic_mmio},
    cpuid, cr,
    ioctl::{LapicState, MpState, VcpuCpuidEntry2, VcpuMsrEntry, VcpuRegs, VcpuSregs},
    ioeventfd::IoEventAddressSpace,
    kvmclock::KvmClock,
    mmio::{MmioDirection, MmioInstruction, decode_current_mmio_instruction},
    msr::{self, MsrAccess},
    pio::{PioDirection, PioOperation},
    vm::Vm,
};
use crate::prelude::*;

const HLT_WAKEUP_WAIT_TSC_DIVISOR: u64 = 10_000;
const HLT_WAKEUP_WAIT_FALLBACK_TICKS: u64 = 250_000;
const MAX_PIO_IOEVENTFD_BATCH_BYTES: usize = PAGE_SIZE;

#[derive(Clone, Copy, Debug)]
pub(super) struct PendingPioOperation {
    pub operation: PioOperation,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PendingMmioOperation {
    pub instruction: MmioInstruction,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PendingOperation {
    Pio(PendingPioOperation),
    Mmio(PendingMmioOperation),
}

pub struct Vcpu {
    id: u32,
    pub(super) vm: Weak<Vm>,
    pub(super) guest_context: Mutex<GuestContext>,
    guest_mode: GuestMode,
    run_lock: Mutex<()>,
    pub(super) lapic: LapicPort,
    kvmclock: Mutex<KvmClock>,
    cpuid_entries: Mutex<Vec<VcpuCpuidEntry2>>,
}

impl Vcpu {
    pub(super) fn new(id: u32, vm: &Arc<Vm>, lapic: Lapic) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            id,
            vm: Arc::downgrade(vm),
            guest_context: Mutex::new(GuestContext::new(id)?),
            guest_mode: GuestMode::new()?,
            run_lock: Mutex::new(()),
            lapic: LapicPort::new(lapic),
            kvmclock: Mutex::new(KvmClock::default()),
            cpuid_entries: Mutex::new(cpuid::default_cpuid_entries()),
        }))
    }

    pub fn lapic(&self) -> SpinLockGuard<'_, Lapic, ostd::sync::PreemptDisabled> {
        self.lapic.lock()
    }

    pub fn guest_context(&self) -> MutexGuard<'_, GuestContext> {
        self.guest_context.lock()
    }

    pub fn vm(&self) -> Result<Arc<Vm>> {
        self.vm
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::ENOENT, "vm not found"))
    }

    pub(super) fn run(self: &Arc<Self>) -> Result<GuestExitInfo> {
        let vm = self.vm()?;
        let _run_guard = self.run_lock.lock();

        loop {
            let run_result = {
                let mut context = self.guest_context();
                match self.guest_mode.execute(
                    &mut context,
                    vm.memory().guest_mem(),
                    &self.lapic,
                    &self.lapic,
                ) {
                    Ok(exit_info) => exit_info,
                    Err(err) => {
                        error!("hypervisor: GuestMode::execute failed: {:?}", err);
                        return Err(err.into());
                    }
                }
            };
            let GuestRunResult::VmExit(exit_info) = run_result else {
                loop {
                    if self.wait_for_sipi_wakeup() {
                        break;
                    }

                    // TODO: Use a more efficient wait mechanism instead of busy-waiting.
                    Task::yield_now();
                }
                continue;
            };
            let exit_info = match VmxExitReason::try_from(exit_info.exit_reason) {
                Ok(VmxExitReason::IO_INSTRUCTION) => {
                    if self.handle_pio_ioeventfd(&vm, &exit_info)? {
                        None
                    } else {
                        Some(exit_info)
                    }
                }
                Ok(VmxExitReason::EPT_VIOLATION) => {
                    let handled = emulate_apic_mmio(self.clone(), exit_info.guest_phys_addr as u64)
                        .inspect_err(|err| {
                            error!(
                                "hypervisor: APIC MMIO handling failed: reason={:#x}, len={}, \
                             rip={:#x}, gpa={:#x}, qualification={:#x}, err={:?}",
                                exit_info.exit_reason,
                                exit_info.instruction_len,
                                exit_info.guest_rip,
                                exit_info.guest_phys_addr,
                                exit_info.exit_qualification,
                                err
                            );
                        })?;
                    if handled || self.handle_mmio_ioeventfd(&vm, &exit_info)? {
                        None
                    } else {
                        Some(exit_info)
                    }
                }
                Ok(VmxExitReason::PREEMPTION_TIMER) => None,
                Ok(VmxExitReason::CPUID) => {
                    cpuid::emulate_cpuid(self, &exit_info)?;
                    None
                }
                Ok(VmxExitReason::CR_ACCESS) => {
                    cr::emulate_cr_access(self, &exit_info)?;
                    None
                }
                Ok(VmxExitReason::HLT) => {
                    self.guest_context()
                        .advance_rip(exit_info.instruction_len as _);
                    loop {
                        if self.wait_for_hlt_wakeup() {
                            break None;
                        }

                        // TODO: Use a more efficient wait mechanism instead of busy-waiting.
                        Task::yield_now();
                    }
                }
                Ok(VmxExitReason::PAUSE_INSTRUCTION) => None,
                Ok(VmxExitReason::MSR_READ) => {
                    msr::emulate_msr(self, &exit_info, MsrAccess::Read)?;
                    None
                }
                Ok(VmxExitReason::MSR_WRITE) => {
                    msr::emulate_msr(self, &exit_info, MsrAccess::Write)?;
                    None
                }
                Ok(_) => Some(exit_info),
                Err(_) => Some(exit_info),
            };

            if let Some(exit_info) = exit_info {
                return Ok(exit_info);
            }
        }
    }

    pub(super) fn complete_pio_operation(
        &self,
        operation: PendingPioOperation,
        input_data: Option<&[u8]>,
    ) -> Result<()> {
        let vm = self.vm()?;
        operation.operation.complete(
            &mut self.guest_context(),
            vm.memory(),
            operation.count,
            input_data,
        )
    }

    pub(super) fn complete_mmio_operation(
        &self,
        operation: PendingMmioOperation,
        read_value: Option<u64>,
    ) -> Result<()> {
        let instruction = operation.instruction;
        let mut context = self.guest_context();
        if let Some(value) = read_value {
            instruction.complete_read(&mut context, value)?;
        }

        context.advance_rip(u64::from(instruction.len()));
        Ok(())
    }

    /// Handles a PIO exit by checking for an ioeventfd and signaling it if present.
    ///
    /// Returns `Ok(true)` if the PIO exit was handled by signaling an ioeventfd,
    ///         `Ok(false)` otherwise.
    fn handle_pio_ioeventfd(&self, vm: &Vm, exit_info: &GuestExitInfo) -> Result<bool> {
        let context = self.guest_context();
        let Some(operation) = PioOperation::decode(&context, vm.memory(), exit_info)? else {
            return Ok(false);
        };
        let count = operation.batch_count(&context, MAX_PIO_IOEVENTFD_BATCH_BYTES);
        if count == 0 {
            drop(context);
            let input_data = (operation.direction() == PioDirection::In).then_some(&[] as &[u8]);
            operation.complete(&mut self.guest_context(), vm.memory(), 0, input_data)?;
            return Ok(true);
        }

        // Pio ioeventfd is only supported for output operations.
        if operation.direction() != PioDirection::Out {
            return Ok(false);
        }
        if !vm.has_ioeventfd(
            IoEventAddressSpace::Pio,
            u64::from(operation.port()),
            u32::from(operation.size()),
        ) {
            return Ok(false);
        }
        let data = operation.output_data(&context, vm.memory(), count)?;
        let values = operation.output_values(&data)?;
        drop(context);

        if !vm.signal_ioeventfd_batch(
            IoEventAddressSpace::Pio,
            u64::from(operation.port()),
            u32::from(operation.size()),
            &values,
        ) {
            return Ok(false);
        }

        operation.complete(&mut self.guest_context(), vm.memory(), count, None)?;
        Ok(true)
    }

    fn handle_mmio_ioeventfd(&self, vm: &Vm, exit_info: &GuestExitInfo) -> Result<bool> {
        if exit_info.exit_qualification & 0b111 != 0b010 {
            return Ok(false);
        }
        let context = self.guest_context();
        let instruction = decode_current_mmio_instruction(&context, vm.memory())?;
        let Some(instruction) = instruction else {
            return Ok(false);
        };
        if instruction.direction() != MmioDirection::Write {
            return Ok(false);
        }
        let Some(value) = instruction.write_value(&context) else {
            return Ok(false);
        };
        drop(context);
        if !vm.signal_ioeventfd(
            IoEventAddressSpace::Mmio,
            exit_info.guest_phys_addr as u64,
            u32::from(instruction.size()),
            value,
        ) {
            return Ok(false);
        }

        self.guest_context()
            .advance_rip(u64::from(instruction.len()));
        Ok(true)
    }

    pub fn get_regs(&self) -> Result<VcpuRegs> {
        let context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot get regs while vCPU is running");
        }
        Ok(context.regs().into())
    }

    pub fn set_regs(&self, regs: VcpuRegs) -> Result<()> {
        let mut context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot set regs while vCPU is running");
        }
        context.set_regs(regs.into());
        Ok(())
    }

    pub fn get_sregs(&self) -> Result<VcpuSregs> {
        let context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot get sregs while vCPU is running");
        }
        Ok(context.sregs().into())
    }

    pub fn set_sregs(&self, sregs: VcpuSregs) -> Result<()> {
        let mut context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot set sregs while vCPU is running");
        }
        let mut sregs = sregs.into();
        cr::sync_efer_lma(&mut sregs);
        context.set_sregs(sregs);
        Ok(())
    }

    pub fn set_cpuid_entries(&self, entries: Vec<VcpuCpuidEntry2>) -> Result<()> {
        let context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot set CPUID while vCPU is running");
        }
        drop(context);

        *self.cpuid_entries.lock() = entries;
        Ok(())
    }

    pub(super) fn cpuid_result(&self, function: u32, index: u32) -> VcpuCpuidEntry2 {
        let entries = self.cpuid_entries.lock();
        cpuid::cpuid_entry(entries.as_slice(), function, index)
    }

    pub fn get_msrs(&self, entries: &mut [VcpuMsrEntry]) -> Result<i32> {
        let context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot get MSRs while vCPU is running");
        }
        drop(context);

        let mut handled_count = 0;
        for entry in entries {
            entry.data = msr::read_msr(self, entry.index);
            handled_count += 1;
        }

        Ok(handled_count)
    }

    pub fn set_msrs(&self, entries: &[VcpuMsrEntry]) -> Result<i32> {
        {
            let context = self.guest_context.lock();
            if context.is_running() {
                return_errno_with_message!(Errno::EBUSY, "cannot set MSRs while vCPU is running");
            }
        }

        let mut handled_count = 0;
        for entry in entries {
            msr::write_msr(self, entry.index, entry.data)?;
            handled_count += 1;
        }

        Ok(handled_count)
    }

    pub(super) fn read_kvmclock_msr(&self, index: u32) -> u64 {
        self.kvmclock.lock().read_msr(index)
    }

    pub(super) fn write_kvmclock_msr(&self, index: u32, value: u64) -> Result<()> {
        let vm = self.vm()?;
        let guest_tsc = self.guest_context.lock().guest_tsc();
        self.kvmclock.lock().write_msr(index, value, &vm, guest_tsc)
    }

    pub(super) fn update_kvmclock(&self) -> Result<()> {
        let vm = self.vm()?;
        let guest_tsc = self.guest_context.lock().guest_tsc();
        self.kvmclock.lock().update_system_time(&vm, guest_tsc)
    }

    pub fn get_tsc_khz(&self) -> Result<i32> {
        Ok(i32::try_from(current_tsc_khz()?)?)
    }

    pub fn set_tsc_khz(&self, khz: u64) -> Result<()> {
        if khz == current_tsc_khz()? {
            return Ok(());
        }

        return_errno_with_message!(Errno::EINVAL, "TSC frequency scaling is not supported");
    }

    pub fn get_lapic(&self) -> Result<LapicState> {
        {
            let context = self.guest_context.lock();
            if context.is_running() {
                return_errno_with_message!(Errno::EBUSY, "cannot get LAPIC while vCPU is running");
            }
        }

        Ok(self.lapic().to_kvm_state())
    }

    pub fn set_lapic(&self, state: &LapicState) -> Result<()> {
        {
            let context = self.guest_context.lock();
            if context.is_running() {
                return_errno_with_message!(Errno::EBUSY, "cannot set LAPIC while vCPU is running");
            }
        }

        self.lapic().set_from_kvm_state(state);
        Ok(())
    }

    pub fn get_mp_state(&self) -> Result<MpState> {
        Ok(self.guest_context.lock().run_state().into())
    }

    pub fn set_mp_state(&self, state: MpState) -> Result<()> {
        let state = state.try_into()?;
        self.guest_context.lock().set_run_state(state);
        Ok(())
    }

    pub fn receive_sipi(&self, vector: u8) {
        self.guest_context.lock().receive_sipi(vector);
    }

    pub(super) fn wait_for_sipi_wakeup(&self) -> bool {
        self.guest_context.lock().run_state() != VcpuRunState::WaitForSipi
    }

    pub(super) fn wait_for_hlt_wakeup(&self) -> bool {
        use ostd::arch::{read_tsc, tsc_freq};
        let wait_max_ticks = match tsc_freq() {
            0 => HLT_WAKEUP_WAIT_FALLBACK_TICKS,
            freq => (freq / HLT_WAKEUP_WAIT_TSC_DIVISOR).max(1),
        };
        let start_tsc = read_tsc();
        loop {
            let guest_tsc = self.guest_context().guest_tsc();
            self.lapic.lock().poll_deadline(guest_tsc);

            if self.lapic.lock().check_pending_interrupt().is_some() {
                return true;
            }

            let tsc = read_tsc();
            if tsc.saturating_sub(start_tsc) >= wait_max_ticks {
                return false;
            }

            core::hint::spin_loop();
        }
    }
}

impl Drop for Vcpu {
    fn drop(&mut self) {
        error!("hypervisor: release VCPU {}.", self.id);
    }
}

fn current_tsc_khz() -> Result<u64> {
    let khz = ostd::arch::tsc_freq() / 1_000;
    if khz == 0 {
        return_errno_with_message!(Errno::EINVAL, "TSC frequency is not available");
    }
    Ok(khz)
}
