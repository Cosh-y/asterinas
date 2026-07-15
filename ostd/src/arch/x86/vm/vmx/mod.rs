// SPDX-License-Identifier: MPL-2.0

//! VMX (Virtual Machine Extensions) support for Intel VT-x.
//!
//! This module coordinates VMX lifecycle management and exposes the VMX
//! building blocks used by the rest of the x86 virtualization implementation.
#![allow(missing_docs)]

/*
 * This module contains code derived from the RVM-Tutorial project.
 * Source: https://github.com/equation314/RVM-Tutorial
 */

mod context_switch;
mod exit;
mod instructions;
mod invept;
mod msr;
mod vmcs;

use alloc::collections::{BTreeMap, VecDeque};
use core::{
    cell::RefCell,
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
};

pub use self::exit::VmxExitReason;
#[expect(
    unused_imports,
    reason = "These fields remain part of the crate-visible VMCS field catalog."
)]
pub(crate) use self::vmcs::{VmcsControl16, VmcsReadOnly64, VmcsReadOnlyNW};
pub(crate) use self::{
    context_switch::vcpu_run,
    exit::{VmxExitInfo, exit_info},
    invept::{check_ept_support, flush_ept_all_contexts_sync},
    msr::Msr,
    vmcs::{
        VmcsControl32, VmcsControl64, VmcsControlNW, VmcsGuest16, VmcsGuest32, VmcsGuest64,
        VmcsGuestNW, VmcsReadOnly32,
    },
};
pub(super) use self::{
    context_switch::vm_exit_handler_virtaddr,
    instructions::{vmclear, vmptrld},
    vmcs::{VmcsHost16, VmcsHost32, VmcsHost64, VmcsHostNW, alloc_vmcs, set_control},
};
use crate::{
    cpu::{CpuId, PinCurrentCpu},
    cpu_local, error,
    error::Error,
    info, irq,
    mm::Frame,
    prelude::*,
    sync::{LocalIrqDisabled, Mutex, SpinLock},
};

static VMX_LIFECYCLE_ACKS: AtomicUsize = AtomicUsize::new(0);
static VMX_LIFECYCLE_ERRORS: AtomicUsize = AtomicUsize::new(0);
static VMX_GUARD_STATE: Mutex<VmxGuardState> = Mutex::new(VmxGuardState {
    active_guards: 0,
    enabled: false,
});
static VMXON_REGIONS: SpinLock<BTreeMap<usize, Frame<()>>> = SpinLock::new(BTreeMap::new());

const NO_LOADED_CPU: usize = usize::MAX;
const VMCLEAR_PENDING: u8 = 0;
const VMCLEAR_SUCCEEDED: u8 = 1;
const VMCLEAR_FAILED: u8 = 2;

pub(super) struct VmcsTracking {
    region: Frame<()>,
    loaded_cpu: AtomicUsize,
    initialized: AtomicBool,
    launched: AtomicBool,
}

impl VmcsTracking {
    pub(super) fn new(region: Frame<()>) -> Self {
        Self {
            region,
            loaded_cpu: AtomicUsize::new(NO_LOADED_CPU),
            initialized: AtomicBool::new(false),
            launched: AtomicBool::new(false),
        }
    }

    pub(super) fn paddr(&self) -> Paddr {
        self.region.paddr()
    }

    pub(super) fn loaded_cpu(&self) -> Option<CpuId> {
        let cpu = self.loaded_cpu.load(Ordering::Acquire);
        (cpu != NO_LOADED_CPU).then(|| CpuId::try_from(cpu).unwrap())
    }

    fn set_loaded_cpu(&self, cpu: Option<CpuId>) {
        let cpu = cpu.map_or(NO_LOADED_CPU, |cpu| u32::from(cpu) as usize);
        self.loaded_cpu.store(cpu, Ordering::Release);
    }

    pub(super) fn initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    pub(super) fn set_initialized(&self, initialized: bool) {
        self.initialized.store(initialized, Ordering::Release);
    }

    pub(super) fn launched(&self) -> bool {
        self.launched.load(Ordering::Acquire)
    }

    pub(super) fn set_launched(&self, launched: bool) {
        self.launched.store(launched, Ordering::Release);
    }
}

struct LocalVmcsState {
    current: Option<Paddr>,
    active: BTreeMap<Paddr, Arc<VmcsTracking>>,
}

impl LocalVmcsState {
    const fn new() -> Self {
        Self {
            current: None,
            active: BTreeMap::new(),
        }
    }
}

struct VmclearRequest {
    paddr: Paddr,
    tracking: Arc<VmcsTracking>,
    result: AtomicU8,
}

impl VmclearRequest {
    fn new(paddr: Paddr, tracking: Arc<VmcsTracking>) -> Self {
        Self {
            paddr,
            tracking,
            result: AtomicU8::new(VMCLEAR_PENDING),
        }
    }
}

cpu_local! {
    static LOCAL_VMCS_STATE: RefCell<LocalVmcsState> = RefCell::new(LocalVmcsState::new());
    static VMCLEAR_REQUESTS: SpinLock<VecDeque<Arc<VmclearRequest>>, LocalIrqDisabled> =
        SpinLock::new(VecDeque::new());
}

struct VmxGuardState {
    active_guards: usize,
    enabled: bool,
}

/// Keeps VMX operation enabled while at least one guest execution object exists.
pub(crate) struct VmxGuard {
    _private: (),
}

/// Acquires a VMX lifecycle guard.
pub(crate) fn acquire_vmx() -> Result<VmxGuard> {
    let mut state = VMX_GUARD_STATE.lock();
    if state.active_guards == usize::MAX {
        return Err(Error::NotEnoughResources);
    }

    if !state.enabled {
        enable_vmx_on_all_cpus()?;
        state.enabled = true;
    }

    state.active_guards += 1;
    Ok(VmxGuard { _private: () })
}

impl Drop for VmxGuard {
    fn drop(&mut self) {
        let mut state = VMX_GUARD_STATE.lock();
        if state.active_guards == 0 {
            error!("hypervisor: VMX guard state underflow");
            return;
        }

        state.active_guards -= 1;
        if state.active_guards != 0 || !state.enabled {
            return;
        }

        match vmxoff_all_cpus() {
            Ok(()) => {
                state.enabled = false;
                info!("VMX disabled successfully");
            }
            Err(err) => {
                state.enabled = false;
                error!("hypervisor: failed to disable VMX on all CPUs: {:?}", err);
            }
        }
    }
}

pub(super) fn activate_vmcs(paddr: Paddr, tracking: &Arc<VmcsTracking>) -> Result<bool> {
    let preempt_guard = crate::task::disable_preempt();
    let current_cpu = preempt_guard.current_cpu();

    match tracking.loaded_cpu() {
        Some(cpu) if cpu == current_cpu => {
            let irq_guard = irq::disable_local();
            let local_state = LOCAL_VMCS_STATE.get_with(&irq_guard);
            let mut local_state = local_state.borrow_mut();
            debug_assert!(local_state.active.contains_key(&paddr));

            if local_state.current != Some(paddr) {
                vmptrld(paddr as u64)?;
                local_state.current = Some(paddr);
            }
            return Ok(false);
        }
        Some(old_cpu) => clear_vmcs_on_cpu(old_cpu, paddr, tracking.clone())?,
        None => {
            // Establish a known clear launch state before the first VMPTRLD.
            vmclear(paddr as u64)?;
            tracking.set_launched(false);
        }
    }

    let irq_guard = irq::disable_local();
    debug_assert_eq!(irq_guard.current_cpu(), current_cpu);
    vmptrld(paddr as u64)?;

    let local_state = LOCAL_VMCS_STATE.get_with(&irq_guard);
    let mut local_state = local_state.borrow_mut();
    let old_tracking = local_state.active.insert(paddr, tracking.clone());
    debug_assert!(old_tracking.is_none());
    local_state.current = Some(paddr);
    tracking.set_loaded_cpu(Some(current_cpu));
    Ok(true)
}

pub(super) fn deactivate_vmcs(paddr: Paddr, tracking: &Arc<VmcsTracking>) -> Result<()> {
    let Some(loaded_cpu) = tracking.loaded_cpu() else {
        return Ok(());
    };

    let preempt_guard = crate::task::disable_preempt();
    if loaded_cpu == preempt_guard.current_cpu() {
        clear_vmcs_on_current_cpu(paddr, tracking)
    } else {
        clear_vmcs_on_cpu(loaded_cpu, paddr, tracking.clone())
    }
}

fn clear_vmcs_on_cpu(cpu: CpuId, paddr: Paddr, tracking: Arc<VmcsTracking>) -> Result<()> {
    let request = Arc::new(VmclearRequest::new(paddr, tracking));
    VMCLEAR_REQUESTS
        .get_on_cpu(cpu)
        .lock()
        .push_back(request.clone());
    crate::smp::inter_processor_call(&crate::cpu::CpuSet::from(cpu), process_vmclear_requests);

    loop {
        match request.result.load(Ordering::Acquire) {
            VMCLEAR_PENDING => core::hint::spin_loop(),
            VMCLEAR_SUCCEEDED => return Ok(()),
            VMCLEAR_FAILED => return Err(Error::InvalidArgs),
            _ => unreachable!(),
        }
    }
}

fn process_vmclear_requests() {
    let current_cpu = CpuId::current_racy();
    loop {
        let Some(request) = VMCLEAR_REQUESTS.get_on_cpu(current_cpu).lock().pop_front() else {
            return;
        };

        let result = clear_vmcs_on_current_cpu(request.paddr, &request.tracking);
        let result = if result.is_ok() {
            VMCLEAR_SUCCEEDED
        } else {
            VMCLEAR_FAILED
        };
        request.result.store(result, Ordering::Release);
    }
}

fn clear_vmcs_on_current_cpu(paddr: Paddr, tracking: &Arc<VmcsTracking>) -> Result<()> {
    let irq_guard = irq::disable_local();
    let current_cpu = irq_guard.current_cpu();
    if tracking.loaded_cpu() != Some(current_cpu) {
        return Err(Error::InvalidArgs);
    }

    vmclear(paddr as u64)?;

    let local_state = LOCAL_VMCS_STATE.get_with(&irq_guard);
    let mut local_state = local_state.borrow_mut();
    let old_tracking = local_state.active.remove(&paddr);
    debug_assert!(
        old_tracking
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, tracking))
    );
    if local_state.current == Some(paddr) {
        local_state.current = None;
    }
    tracking.set_loaded_cpu(None);
    tracking.set_launched(false);
    Ok(())
}

fn clear_all_active_vmcs_on_current_cpu() -> Result<()> {
    let irq_guard = irq::disable_local();
    let local_state = LOCAL_VMCS_STATE.get_with(&irq_guard);
    let mut local_state = local_state.borrow_mut();

    while let Some((&paddr, tracking)) = local_state.active.first_key_value() {
        let tracking = tracking.clone();
        vmclear(paddr as u64)?;
        local_state.active.remove(&paddr);
        if local_state.current == Some(paddr) {
            local_state.current = None;
        }
        tracking.set_loaded_cpu(None);
        tracking.set_launched(false);
    }
    debug_assert!(local_state.current.is_none());
    Ok(())
}

fn enable_vmx_on_all_cpus() -> Result<()> {
    // Check CPUID for VMX support
    let cpuid_result = core::arch::x86_64::__cpuid(1);
    if (cpuid_result.ecx & (1 << 5)) == 0 {
        error!("VMX not supported by CPU");
        return Err(Error::NotEnoughResources);
    }

    if let Err(err) = vmxon_all_cpus() {
        error!("hypervisor: failed to enable VMX on all CPUs: {:?}", err);
        if let Err(rollback_err) = vmxoff_all_cpus() {
            error!(
                "hypervisor: failed to roll back partial VMX enablement: {:?}",
                rollback_err
            );
        }
        return Err(err);
    }

    info!("VMX initialized successfully");
    Ok(())
}

fn vmxon_all_cpus() -> Result<()> {
    run_vmx_lifecycle_on_all_cpus(vmxon_lifecycle_current_cpu)
}

fn vmxoff_all_cpus() -> Result<()> {
    run_vmx_lifecycle_on_all_cpus(vmxoff_lifecycle_current_cpu)
}

fn run_vmx_lifecycle_on_all_cpus(operation: fn()) -> Result<()> {
    VMX_LIFECYCLE_ACKS.store(0, Ordering::Release);
    VMX_LIFECYCLE_ERRORS.store(0, Ordering::Release);

    let targets = crate::cpu::CpuSet::new_full();
    let cpu_count = crate::cpu::num_cpus();
    crate::smp::inter_processor_call(&targets, operation);

    while VMX_LIFECYCLE_ACKS.load(Ordering::Acquire) < cpu_count {
        core::hint::spin_loop();
    }

    if VMX_LIFECYCLE_ERRORS.load(Ordering::Acquire) != 0 {
        return Err(Error::InvalidArgs);
    }
    Ok(())
}

fn vmxon_lifecycle_current_cpu() {
    enable_vmx_in_cr4();
    if vmxon_current_cpu().is_err() {
        VMX_LIFECYCLE_ERRORS.fetch_add(1, Ordering::AcqRel);
    }
    VMX_LIFECYCLE_ACKS.fetch_add(1, Ordering::AcqRel);
}

fn vmxoff_lifecycle_current_cpu() {
    let result = clear_all_active_vmcs_on_current_cpu().and_then(|_| vmxoff_current_cpu());
    if result.is_err() {
        VMX_LIFECYCLE_ERRORS.fetch_add(1, Ordering::AcqRel);
    }
    VMX_LIFECYCLE_ACKS.fetch_add(1, Ordering::AcqRel);
}

fn enable_vmx_in_cr4() {
    const CR4_VMXE: u64 = 1 << 13;

    // SAFETY: VMXON requires CR4.VMXE on the current CPU. This only sets that
    // architectural enable bit and preserves all other CR4 bits.
    unsafe {
        let mut cr4: u64;
        core::arch::asm!(
            "mov {}, cr4",
            out(reg) cr4,
            options(nomem, nostack)
        );
        cr4 |= CR4_VMXE;
        core::arch::asm!(
            "mov cr4, {}",
            in(reg) cr4,
            options(nostack)
        );
    }
}

fn vmxon_current_cpu() -> Result<()> {
    let preempt_guard = crate::task::disable_preempt();
    let cpu = u32::from(PinCurrentCpu::current_cpu(&preempt_guard)) as usize;
    if VMXON_REGIONS.lock().contains_key(&cpu) {
        return Ok(());
    }

    let vmxon_region = alloc_vmcs()?;
    let vmxon_region_paddr = vmxon_region.paddr();
    // SAFETY: The VMXON region is page-sized, aligned, initialized with the
    // current VMCS revision ID, and kept alive in `VMXON_REGIONS` until VMXOFF.
    unsafe {
        instructions::vmxon(vmxon_region_paddr)?;
    }

    let old_region = VMXON_REGIONS.lock().insert(cpu, vmxon_region);
    debug_assert!(old_region.is_none());
    Ok(())
}

fn vmxoff_current_cpu() -> Result<()> {
    let cpu = {
        let preempt_guard = crate::task::disable_preempt();
        let cpu = u32::from(PinCurrentCpu::current_cpu(&preempt_guard)) as usize;
        if !VMXON_REGIONS.lock().contains_key(&cpu) {
            return Ok(());
        }

        instructions::vmxoff()?;
        cpu
    };

    let _vmxon_region = VMXON_REGIONS.lock().remove(&cpu);
    Ok(())
}

#[cfg(ktest)]
pub(super) mod test_support {
    use crate::prelude::Result;

    pub(in crate::arch::vm) fn init_vmcs_revision() -> u32 {
        super::vmcs::vmcs_revision()
    }

    pub(in crate::arch::vm) fn vmxon_current_cpu() -> Result<()> {
        super::vmxon_current_cpu()
    }

    pub(in crate::arch::vm) fn vmxoff_current_cpu() -> Result<()> {
        super::vmxoff_current_cpu()
    }
}
