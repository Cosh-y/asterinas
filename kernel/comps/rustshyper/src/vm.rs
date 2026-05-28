//! VM (Virtual Machine) management for RustShyper

use alloc::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};
use core::sync::atomic::{AtomicU32, Ordering};

use ostd::{
    arch::{
        cpu::{context::FpuContext, cpuid::cpuid},
        tsc_freq,
        virt::*,
    },
    mm::{
        HasPaddr, PAGE_SIZE, VmIo, VmSpace, kspace::{read_bytes_from_paddr, read_u64_from_paddr}, vm_space::VmQueriedItem
    },
    sync::{Mutex, SpinLock, SpinLockGuard},
};

use crate::vcpu::{
    Vcpu, VcpuState, VcpuMpState, VcpuMsrs,
    reset_vcpu_for_init_locked, start_vcpu_from_sipi_locked,
};

use super::{
    emulate::apic::{
        ioapic_kick_irq, lapic_set_irr, Icr,
        icr_matches_destination, ApicTimer, Ioapic, Lapic, TscState, IOAPIC_NUM_PINS,
        default_lapic_ldr,
    },
    emulate::timer::{
        VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK,
        timer_deactivate_locked,
    },
    error::*,
    interrupt::{clear_event_injection, vmx_interrupt_snapshot,},
    utils::*,
};

const PAUSE_DIAG_LOG_INTERVAL: u64 = 1 << 20;
const LAPIC_FIXED_IPI_DIAG_LOG_INTERVAL: u64 = 1 << 12;
const LAPIC_INJECT_DIAG_LOG_INTERVAL: u64 = 1 << 12;
const LAPIC_TIMER_EXPIRE_DIAG_LOG_INTERVAL: u64 = 1 << 12;
const LAPIC_TIMER_INJECT_DIAG_LOG_INTERVAL: u64 = 1 << 12;
const LINUX_LOCAL_TIMER_VECTOR: u8 = 0xec;

/// Represents a virtual machine instance
pub struct Vm {
    /// VM ID
    id: u32,
    /// Memory regions mapped to this VM
    memory_regions: Mutex<BTreeMap<u32, MemoryRegion>>,
    /// EPT Table used by this VM
    ept: SpinLock<EptPageTable>,
    /// VCPUs belonging to this VM
    vcpus: Mutex<BTreeMap<u32, Arc<Vcpu>>>,
    /// Shared IOAPIC state.
    pub(crate) ioapic: SpinLock<Ioapic>,
    /// Next VCPU ID
    next_vcpu_id: AtomicU32,
}

/// Memory region mapped to a VM
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    /// Slot number
    pub slot: u32,
    /// Flags
    pub flags: u32,
    /// Guest physical address
    pub guest_phys_addr: GuestPhysAddr,
    /// Memory size
    pub memory_size: MemSize,
    /// Userspace virtual address
    pub userspace_addr: VirtAddr,
}

impl Vm {
    /// Creates a new VM instance
    pub fn new(id: u32) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            id,
            memory_regions: Mutex::new(BTreeMap::new()),
            ept: SpinLock::new(EptPageTable::new()?),
            vcpus: Mutex::new(BTreeMap::new()),
            ioapic: SpinLock::new(Ioapic {
                id: 1,
                ..Ioapic::default()
            }),
            next_vcpu_id: AtomicU32::new(0),
        }))
    }

    /// Gets the VM ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Sets a user memory region
    pub fn set_memory_region(&self, region: MemoryRegion, vm_space: &Arc<VmSpace>) -> Result<()> {
        self.map_memory_region(region, vm_space)?;
        self.memory_regions.lock().insert(region.slot, region);
        Ok(())
    }

    /// Maps a user memory region
    pub fn map_memory_region(&self, region: MemoryRegion, vm_space: &Arc<VmSpace>) -> Result<()> {
        if region.guest_phys_addr % PAGE_SIZE as u64 != 0 {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "guest physical address must be page aligned",
            ));
        }
        if region.userspace_addr % PAGE_SIZE != 0 {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "userspace address must be page aligned",
            ));
        }
        if region.memory_size == 0 || region.memory_size % PAGE_SIZE != 0 {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "memory size must be a non-zero multiple of PAGE_SIZE",
            ));
        }

        let userspace_end = region
            .userspace_addr
            .checked_add(region.memory_size)
            .ok_or(Error::with_message(
                Errno::InvalidArgs,
                "userspace address range overflows",
            ))?;

        let guest_end = region
            .guest_phys_addr
            .checked_add(region.memory_size as u64)
            .ok_or(Error::with_message(
                Errno::InvalidArgs,
                "guest physical address range overflows",
            ))?;

        // Asterinas does not expose a single "translate user VA to PA" API.
        // The supported interface today is VmSpace::cursor(...).query(), which
        // lets us inspect each mapped userspace page and recover its backing
        // frame or I/O-memory physical address.
        let mut userspace_addr = region.userspace_addr;
        let mut guest_phys_addr = region.guest_phys_addr;

        while userspace_addr < userspace_end && guest_phys_addr < guest_end {
            let hpa = query_userspace_page_hpa(vm_space, userspace_addr)?;
            let mut ept = self.ept.lock();
            ept.map_range(guest_phys_addr, hpa as _, PAGE_SIZE)
                .map_err(Error::from)?;
            userspace_addr += PAGE_SIZE;
            guest_phys_addr += PAGE_SIZE as u64;
        }

        Ok(())
    }

    pub fn get_eptp(&self) -> u64 {
        self.ept.lock().eptp()
    }

    pub fn translate_gpa_to_hpa(&self, gpa: u64) -> Result<u64> {
        self.ept.lock().translate(gpa).map_err(|_| Error::with_message(Errno::Fault, "guest GPA is not mapped in EPT"))
    }

    /// Creates a new VCPU for this VM
    pub fn create_vcpu(self: &Arc<Self>, vcpu_id: u32) -> Result<Arc<Vcpu>> {
        let mut vcpus = self.vcpus.lock();

        // Check if VCPU already exists
        if vcpus.contains_key(&vcpu_id) {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "VCPU with the same ID already exists",
            ));
        }

        let vcpu = Arc::new(Vcpu::new(vcpu_id, Arc::downgrade(self))?);

        vcpus.insert(vcpu_id, vcpu.clone());
        self.next_vcpu_id.fetch_max(vcpu_id + 1, Ordering::Relaxed);

        Ok(vcpu)
    }

    /// Gets a VCPU by ID
    pub fn get_vcpu(&self, vcpu_id: u32) -> Result<Arc<Vcpu>> {
        let vcpus = self.vcpus.lock();
        vcpus
            .get(&vcpu_id)
            .cloned()
            .ok_or_else(|| Error::with_message(Errno::InvalidArgs, "VCPU not found"))
    }

    /// Inject an IRQ line through the emulated I/O APIC.
    pub fn inject_irq_line(&self, irq: usize) -> Result<()> {
        if irq >= IOAPIC_NUM_PINS {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "IRQ line is out of range for the emulated I/O APIC",
            ));
        }

        let vcpus: alloc::vec::Vec<_> = self
            .vcpus
            .lock()
            .iter()
            .map(|(&vcpu_id, vcpu)| (vcpu_id, vcpu.clone()))
            .collect();

        if vcpus.is_empty() {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "cannot inject IRQ without any vCPU",
            ));
        }

        let mut ioapic = self.ioapic.lock();
        let mut state_guards: alloc::vec::Vec<_> = vcpus
            .iter()
            .map(|(vcpu_id, vcpu)| (*vcpu_id, vcpu.state.lock()))
            .collect();
        let lapic_ids: alloc::vec::Vec<_> = state_guards
            .iter()
            .map(|(_, state)| state.lapic.id)
            .collect();
        let mut lapics: alloc::vec::Vec<_> = lapic_ids
            .iter()
            .zip(state_guards.iter_mut())
            .map(|(lapic_id, (_, state))| (lapic_id, &mut state.lapic))
            .collect();

        ioapic_kick_irq(&mut ioapic, &mut lapics, irq);
        Ok(())
    }

    pub fn vcpu_count(&self) -> u32 {
        self.vcpus.lock().len() as u32
    }

    /// Delivers a Local APIC ICR write from one vCPU to its target vCPUs.
    pub fn deliver_lapic_icr(&self, source_vcpu_id: u32, icr: Icr) -> Result<()> {
        let vcpus: alloc::vec::Vec<_> = self
            .vcpus
            .lock()
            .iter()
            .map(|(&vcpu_id, vcpu)| (vcpu_id, vcpu.clone()))
            .collect();

        for (vcpu_id, vcpu) in vcpus {
            let mut state = vcpu.state.lock();
            if !icr_matches_destination(
                source_vcpu_id,
                vcpu_id,
                &state.lapic,
                &icr,
            ) {
                continue;
            }

            const APIC_ICR_DELIVERY_MODE_FIXED: u8 = 0;
            const APIC_ICR_DELIVERY_MODE_INIT: u8 = 5;
            const APIC_ICR_DELIVERY_MODE_STARTUP: u8 = 6;

            match icr.delivery_mode {
                APIC_ICR_DELIVERY_MODE_FIXED => {
                    if icr.vector >= 16 {
                        lapic_set_irr(&mut state.lapic, icr.vector);
                    }
                }
                APIC_ICR_DELIVERY_MODE_INIT => {
                    if icr.level != 0 {
                        reset_vcpu_for_init_locked(&mut state, vcpu_id);
                    }
                }
                APIC_ICR_DELIVERY_MODE_STARTUP => {
                    if state.mp_state == VcpuMpState::WaitForSipi {
                        start_vcpu_from_sipi_locked(&mut state, icr.vector, vcpu_id);
                    }
                }
                _ => {
                    log::error!(
                        "rustshyper: unsupported LAPIC ICR delivery mode {}",
                        icr.delivery_mode,
                    );
                }
            }
        }

        Ok(())
    }
}

fn query_userspace_page_hpa(vm_space: &Arc<VmSpace>, userspace_addr: VirtAddr) -> Result<PhysAddr> {
    debug_assert!(userspace_addr.is_multiple_of(PAGE_SIZE));

    loop {
        let page_range = userspace_addr..(userspace_addr + PAGE_SIZE);
        let preempt_guard = ostd::task::disable_preempt();
        let mut cursor = vm_space.cursor(&preempt_guard, &page_range).map_err(|_| {
            Error::with_message(
                Errno::Fault,
                "failed to create vm_space cursor for guest memory",
            )
        })?;

        let queried_item = cursor
            .query()
            .map_err(|_| {
                Error::with_message(
                    Errno::Fault,
                    "failed to query vm_space mapping for guest memory",
                )
            })?
            .1;

        match queried_item {
            Some(VmQueriedItem::MappedRam { frame, .. }) => return Ok(frame.paddr() as _),
            Some(VmQueriedItem::MappedIoMem { paddr, .. }) => return Ok(paddr as _),
            None => (),
        }

        drop(cursor);
        drop(preempt_guard);

        touch_userspace_page(vm_space, userspace_addr)?;
    }
}

fn touch_userspace_page(vm_space: &Arc<VmSpace>, userspace_addr: VirtAddr) -> Result<()> {
    let mut reader = vm_space.reader(userspace_addr, 1).map_err(|err| {
        let _ = err;
        Error::with_message(
            Errno::Fault,
            "failed to create userspace reader while faulting in guest memory",
        )
    })?;

    let _: u8 = reader.read_val().map_err(|err| {
        let _ = err;
        Error::with_message(
            Errno::Fault,
            "failed to fault in userspace page for guest memory",
        )
    })?;

    Ok(())
}

fn is_canonical_guest_kernel_pointer(value: u64) -> bool {
    value >= (!0_u64 << 47)
}

fn lapic_bitmap_has_vector(bitmap: &[u32; 8], vector: u8) -> bool {
    (bitmap[(vector / 32) as usize] & (1_u32 << (vector % 32))) != 0
}
