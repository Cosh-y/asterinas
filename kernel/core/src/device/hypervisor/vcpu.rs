use ostd::{
    arch::vm::{GuestContext, VcpuRunState},
    vm::GuestMode,
};

use super::{
    apic::{Lapic, LapicPort},
    cpuid, cr,
    ioctl::{
        KVM_RUN_EXIT_REASON_OFFSET, KVM_RUN_MMAP_SIZE, KVM_RUN_STRUCT_SIZE, LapicState, MpState,
        VcpuCpuidEntry2, VcpuMsrEntry, VcpuRegs, VcpuSregs,
    },
    kvmclock::KvmClock,
    mmio::MmioInstruction,
    msr,
    pio::PioOperation,
    vm::Vm,
};
use crate::{
    prelude::*,
    vm::page_cache::{Vmo, VmoOptions},
};

const HLT_WAKEUP_WAIT_TSC_DIVISOR: u64 = 10_000;
const HLT_WAKEUP_WAIT_FALLBACK_TICKS: u64 = 250_000;

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
    guest_mode: Mutex<GuestMode>,
    pub(super) lapic: LapicPort,
    run_page: Arc<Vmo>,
    pending_operation: Mutex<Option<PendingOperation>>,
    kvmclock: Mutex<KvmClock>,
    cpuid_entries: Mutex<Vec<VcpuCpuidEntry2>>,
}

impl Vcpu {
    pub(super) fn new(id: u32, vm: &Arc<Vm>, lapic: Lapic) -> Result<Arc<Self>> {
        let run_page = VmoOptions::new(KVM_RUN_MMAP_SIZE).alloc()?;
        Ok(Arc::new(Self {
            id,
            vm: Arc::downgrade(vm),
            guest_context: Mutex::new(GuestContext::new(id)?),
            guest_mode: Mutex::new(GuestMode::new()?),
            lapic: LapicPort::new(lapic),
            run_page,
            pending_operation: Mutex::new(None),
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

    pub fn guest_mode(&self) -> MutexGuard<'_, GuestMode> {
        self.guest_mode.lock()
    }

    pub fn vm(&self) -> Result<Arc<Vm>> {
        self.vm
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::ENOENT, "vm not found"))
    }

    pub(super) fn run_page(&self) -> Arc<Vmo> {
        self.run_page.clone()
    }

    pub(super) fn read_run_val<T: Pod>(&self, offset: usize) -> Result<T> {
        let mut value = T::new_zeroed();
        let mut writer = VmWriter::from(value.as_mut_bytes()).to_fallible();
        self.run_page.read(offset, &mut writer)?;
        Ok(value)
    }

    pub(super) fn write_run_val<T: Pod>(&self, offset: usize, value: &T) -> Result<()> {
        let mut reader = VmReader::from(value.as_bytes()).to_fallible();
        self.run_page.write(offset, &mut reader)
    }

    pub(super) fn read_run_bytes(&self, offset: usize, buffer: &mut [u8]) -> Result<()> {
        let mut writer = VmWriter::from(buffer).to_fallible();
        self.run_page.read(offset, &mut writer)
    }

    pub(super) fn write_run_bytes(&self, offset: usize, buffer: &[u8]) -> Result<()> {
        let mut reader = VmReader::from(buffer).to_fallible();
        self.run_page.write(offset, &mut reader)
    }

    pub(super) fn clear_run_output(&self) -> Result<()> {
        static ZERO_PAGE: [u8; PAGE_SIZE] = [0; PAGE_SIZE];

        let mut offset = KVM_RUN_EXIT_REASON_OFFSET;
        while offset < KVM_RUN_STRUCT_SIZE {
            let len = (KVM_RUN_STRUCT_SIZE - offset).min(PAGE_SIZE);
            let mut reader = VmReader::from(&ZERO_PAGE[..len]).to_fallible();
            self.run_page.write(offset, &mut reader)?;
            offset += len;
        }
        Ok(())
    }

    pub(super) fn set_pending_operation(&self, operation: PendingOperation) {
        *self.pending_operation.lock() = Some(operation);
    }

    pub(super) fn take_pending_operation(&self) -> Option<PendingOperation> {
        self.pending_operation.lock().take()
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
