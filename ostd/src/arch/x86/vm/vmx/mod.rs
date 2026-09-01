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

use x86::msr::rdmsr;
use x86_64::registers::control::{Cr0, Cr4};

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
    cpu::{CpuId, CpuSet, PinCurrentCpu, all_cpus},
    cpu_local, error,
    error::Error,
    irq::{self, InterruptLevel},
    mm::{Frame, FrameAllocOptions, paddr_to_vaddr},
    prelude::*,
    sync::{LocalIrqDisabled, Mutex, SpinLock},
};

const IA32_FEATURE_CONTROL: u32 = 0x3a;
const IA32_VMX_BASIC: u32 = 0x480;
const IA32_VMX_CR0_FIXED0: u32 = 0x486;
const IA32_VMX_CR0_FIXED1: u32 = 0x487;
const IA32_VMX_CR4_FIXED0: u32 = 0x488;
const IA32_VMX_CR4_FIXED1: u32 = 0x489;
const CR4_VMXE: u64 = 1 << 13;

static VMX_GUARD_STATE: Mutex<VmxGuardState> = Mutex::new(VmxGuardState::new());

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

struct VmxCpuState {
    enabled: bool,
    region: Option<Frame<()>>,
    last_error: Option<Error>,
}

impl VmxCpuState {
    const fn new() -> Self {
        Self {
            enabled: false,
            region: None,
            last_error: None,
        }
    }
}

struct EnableError {
    error: Error,
    cleanup_complete: bool,
}

cpu_local! {
    static LOCAL_VMCS_STATE: RefCell<LocalVmcsState> = RefCell::new(LocalVmcsState::new());
    static VMCLEAR_REQUESTS: SpinLock<VecDeque<Arc<VmclearRequest>>, LocalIrqDisabled> =
        SpinLock::new(VecDeque::new());
    static VMX_CPU_STATE: SpinLock<VmxCpuState, LocalIrqDisabled> =
        SpinLock::new(VmxCpuState::new());
}

struct VmxGuardState {
    active_guards: usize,
    enabled: bool,
    poisoned: bool,
}

impl VmxGuardState {
    const fn new() -> Self {
        Self {
            active_guards: 0,
            enabled: false,
            poisoned: false,
        }
    }
}

/// Keeps VMX operation enabled while at least one guest execution object exists.
#[must_use]
pub(crate) struct VmxGuard {
    _private: (),
}

/// Acquires a VMX lifecycle guard.
pub(crate) fn acquire_vmx() -> Result<VmxGuard> {
    if !InterruptLevel::current().is_task_context()
        || !crate::arch::irq::is_local_enabled()
        || crate::smp::IPI_SENDER.get().is_none()
    {
        return Err(Error::InvalidArgs);
    }

    let mut state = VMX_GUARD_STATE.lock();
    if state.poisoned {
        return Err(Error::InvalidArgs);
    }
    if state.active_guards == usize::MAX {
        return Err(Error::NotEnoughResources);
    }

    if !state.enabled {
        match enable_vmx_on_all_cpus() {
            Ok(()) => state.enabled = true,
            Err(enable_error) => {
                state.enabled = !enable_error.cleanup_complete;
                state.poisoned = !enable_error.cleanup_complete;
                return Err(enable_error.error);
            }
        }
    }

    state.active_guards += 1;
    Ok(VmxGuard { _private: () })
}

impl Drop for VmxGuard {
    fn drop(&mut self) {
        debug_assert!(InterruptLevel::current().is_task_context());
        debug_assert!(crate::arch::irq::is_local_enabled());

        let mut state = VMX_GUARD_STATE.lock();
        if state.active_guards == 0 {
            error!("VMX guard state underflow");
            return;
        }

        state.active_guards -= 1;
        if state.active_guards != 0 {
            return;
        }

        match disable_vmx_on_all_cpus() {
            Ok(()) => state.enabled = false,
            Err(err) => {
                error!("failed to disable VMX on all CPUs: {:?}", err);
                state.poisoned = true;
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
    crate::smp::inter_processor_call(&CpuSet::from(cpu), process_vmclear_requests);

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

fn enable_vmx_on_all_cpus() -> core::result::Result<(), EnableError> {
    prepare_vmxon_regions().map_err(|error| EnableError {
        error,
        cleanup_complete: true,
    })?;

    let targets = CpuSet::new_full();
    let enable_error = match run_on_cpus(&targets, enable_vmx_on_current_cpu) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };
    cleanup_prepared_regions();

    let rollback_targets = enabled_cpus();
    if rollback_targets.is_empty() {
        return Err(EnableError {
            error: enable_error,
            cleanup_complete: true,
        });
    }

    let rollback_completed = run_on_cpus(&rollback_targets, disable_vmx_on_current_cpu).is_ok();
    cleanup_prepared_regions();

    Err(EnableError {
        error: enable_error,
        cleanup_complete: rollback_completed && enabled_cpus().is_empty(),
    })
}

fn disable_vmx_on_all_cpus() -> Result<()> {
    let targets = enabled_cpus();
    if targets.is_empty() {
        return Err(Error::InvalidArgs);
    }

    let result = run_on_cpus(&targets, disable_vmx_on_current_cpu);
    cleanup_prepared_regions();
    result
}

fn prepare_vmxon_regions() -> Result<()> {
    for cpu in all_cpus() {
        let region = match FrameAllocOptions::new().alloc_frame() {
            Ok(region) => region,
            Err(error) => {
                cleanup_prepared_regions();
                return Err(error);
            }
        };

        let mut state = VMX_CPU_STATE.get_on_cpu(cpu).lock();
        debug_assert!(state.region.is_none());
        state.region = Some(region);
        state.last_error = None;
    }
    Ok(())
}

fn cleanup_prepared_regions() {
    for cpu in all_cpus() {
        let region = {
            let mut state = VMX_CPU_STATE.get_on_cpu(cpu).lock();
            if state.enabled || state.region.is_none() {
                continue;
            }
            state.last_error = None;
            state.region.take()
        };
        drop(region);
    }
}

fn enabled_cpus() -> CpuSet {
    let mut enabled = CpuSet::new_empty();
    for cpu in all_cpus() {
        if VMX_CPU_STATE.get_on_cpu(cpu).lock().enabled {
            enabled.add(cpu);
        }
    }
    enabled
}

fn run_on_cpus(targets: &CpuSet, handler: fn()) -> Result<()> {
    crate::smp::inter_processor_call(targets, handler).wait();
    first_cpu_error(targets).map_or(Ok(()), Err)
}

fn first_cpu_error(targets: &CpuSet) -> Option<Error> {
    for cpu in targets.iter() {
        let state = VMX_CPU_STATE.get_on_cpu(cpu).lock();
        if let Some(error) = state.last_error {
            return Some(error);
        }
    }
    None
}

fn enable_vmx_on_current_cpu() {
    if let Err(error) = try_enable_vmx_on_current_cpu() {
        let irq_guard = irq::disable_local();
        VMX_CPU_STATE.get_with(&irq_guard).lock().last_error = Some(error);
    }
}

fn try_enable_vmx_on_current_cpu() -> Result<()> {
    let irq_guard = irq::disable_local();
    let state = VMX_CPU_STATE.get_with(&irq_guard);
    let region_paddr = {
        let state = state.lock();
        if state.enabled {
            return Err(Error::InvalidArgs);
        }
        state.region.as_ref().ok_or(Error::InvalidArgs)?.paddr()
    };

    let cr4 = Cr4::read_raw();
    if cr4 & CR4_VMXE != 0 {
        return Err(Error::InvalidArgs);
    }
    let vmx_cr4 = cr4 | CR4_VMXE;
    let revision_id = read_and_validate_capability(vmx_cr4)?;
    initialize_vmxon_region(region_paddr, revision_id);

    // SAFETY: `vmx_cr4` preserves the current `CR4` value, adds only
    // `CR4.VMXE`, and has been checked against the VMX fixed-bit MSRs.
    unsafe { Cr4::write_raw(vmx_cr4) };

    // SAFETY: The capability checks, control-register update, and initialized
    // region above establish the architectural prerequisites for `VMXON`.
    if let Err(error) = unsafe { instructions::vmxon(region_paddr) } {
        // SAFETY: A failed `VMXON` leaves this CPU outside VMX operation.
        unsafe { clear_vmx_enable() };
        return Err(error);
    }

    let mut state = state.lock();
    state.enabled = true;
    state.last_error = None;
    Ok(())
}

fn disable_vmx_on_current_cpu() {
    if let Err(error) = try_disable_vmx_on_current_cpu() {
        let irq_guard = irq::disable_local();
        VMX_CPU_STATE.get_with(&irq_guard).lock().last_error = Some(error);
    }
}

fn try_disable_vmx_on_current_cpu() -> Result<()> {
    let irq_guard = irq::disable_local();
    let state = VMX_CPU_STATE.get_with(&irq_guard);
    {
        let state = state.lock();
        if !state.enabled || state.region.is_none() {
            return Err(Error::InvalidArgs);
        }
    }

    clear_all_active_vmcs_on_current_cpu()?;
    instructions::vmxoff()?;

    // SAFETY: A successful `VMXOFF` leaves this CPU outside VMX operation.
    unsafe { clear_vmx_enable() };

    let mut state = state.lock();
    state.enabled = false;
    state.last_error = None;
    Ok(())
}

/// Clears `CR4.VMXE` without changing any other `CR4` bits.
///
/// # Safety
///
/// The current CPU must be outside VMX operation.
unsafe fn clear_vmx_enable() {
    let cr4 = Cr4::read_raw();
    // SAFETY: The caller guarantees that clearing `CR4.VMXE` is permitted.
    unsafe { Cr4::write_raw(cr4 & !CR4_VMXE) };
}

fn read_and_validate_capability(vmx_cr4: u64) -> Result<u32> {
    const FEATURE_CONTROL_LOCKED: u64 = 1;
    const FEATURE_CONTROL_VMX_OUTSIDE_SMX: u64 = 1 << 2;

    let has_vmx =
        crate::arch::cpu::cpuid::cpuid(1, 0).is_some_and(|result| result.ecx & (1 << 5) != 0);
    if !has_vmx {
        return Err(Error::NotEnoughResources);
    }

    // SAFETY: A CPU that enumerates VMX provides the architectural VMX MSRs
    // read below. This function runs independently on each target CPU.
    let (feature_control, vmx_basic, cr0_fixed0, cr0_fixed1, cr4_fixed0, cr4_fixed1) = unsafe {
        (
            rdmsr(IA32_FEATURE_CONTROL),
            rdmsr(IA32_VMX_BASIC),
            rdmsr(IA32_VMX_CR0_FIXED0),
            rdmsr(IA32_VMX_CR0_FIXED1),
            rdmsr(IA32_VMX_CR4_FIXED0),
            rdmsr(IA32_VMX_CR4_FIXED1),
        )
    };

    let required_feature_control = FEATURE_CONTROL_LOCKED | FEATURE_CONTROL_VMX_OUTSIDE_SMX;
    if feature_control & required_feature_control != required_feature_control {
        return Err(Error::AccessDenied);
    }

    if !control_register_is_valid(Cr0::read_raw(), cr0_fixed0, cr0_fixed1)
        || !control_register_is_valid(vmx_cr4, cr4_fixed0, cr4_fixed1)
    {
        return Err(Error::NotEnoughResources);
    }

    Ok(vmx_basic as u32 & 0x7fff_ffff)
}

fn control_register_is_valid(value: u64, fixed0: u64, fixed1: u64) -> bool {
    value & fixed0 == fixed0 && value & !fixed1 == 0
}

fn initialize_vmxon_region(region_paddr: Paddr, revision_id: u32) {
    let region_ptr = paddr_to_vaddr(region_paddr) as *mut u32;

    // SAFETY: `region_paddr` belongs to the live, exclusively owned frame in
    // the current CPU state. The frame is linearly mapped, page-aligned, and
    // was zero-initialized before this four-byte write.
    unsafe { region_ptr.write(revision_id) };
}

#[cfg(ktest)]
pub(super) mod test_support {
    use crate::prelude::Result;

    pub(in crate::arch::vm) fn init_vmcs_revision() -> u32 {
        super::vmcs::vmcs_revision()
    }

    pub(in crate::arch::vm) fn vmxon_current_cpu() -> Result<()> {
        super::try_enable_vmx_on_current_cpu()
    }

    pub(in crate::arch::vm) fn vmxoff_current_cpu() -> Result<()> {
        super::try_disable_vmx_on_current_cpu()
    }
}
