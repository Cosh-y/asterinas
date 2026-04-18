//! VM (Virtual Machine) management for RustShyper

use alloc::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};
use core::{
    arch::x86_64::CpuidResult,
    sync::atomic::{AtomicU32, Ordering},
};
use x86::vmx::vmcs::control::{
    EntryControls, ExitControls, PinbasedControls, PrimaryControls, SecondaryControls,
};
use x86_64::registers::control::{Cr0, Cr0Flags, Cr3, Cr4, Cr4Flags};

use ostd::{
    arch::{cpu::cpuid::cpuid, read_tsc, tsc_freq, virt::*},
    mm::{
        kspace::{read_bytes_from_paddr, read_u64_from_paddr},
        Frame, FrameAllocOptions, HasPaddr, VmIo, PAGE_SIZE,
    },
    sync::Mutex,
};

use super::{
    emulate::apic::{
        ioapic_kick_irq, lapic_check_pending_vector, lapic_on_timer_expire, lapic_set_irr,
        lapic_timer_deadline, lapic_timer_deadline_tsc, ApicTimer, Ioapic, Lapic,
        TimerExpireAction, TscState, IOAPIC_NUM_PINS,
    },
    error::*,
    interrupt::{
        has_pending_exception, inject_interrupt, inject_pending_exception,
        try_inject_pending_interrupt, ExceptionState, InterruptState,
    },
};

const X86_CR0_ET: u64 = 1 << 4;
const X86_CR0_NE: u64 = 1 << 5;
const X86_CR4_VMXE: u64 = 1 << 13;
const X86_EFER_LMA: u64 = 1 << 10;

/// Represents a virtual machine instance
pub struct Vm {
    /// VM ID
    id: u32,
    /// Memory regions mapped to this VM
    memory_regions: Mutex<BTreeMap<u32, MemoryRegion>>,
    /// EPT Table used by this VM
    ept: Mutex<EptPageTable>,
    /// VCPUs belonging to this VM
    vcpus: Mutex<BTreeMap<u32, Arc<Vcpu>>>,
    /// Shared IOAPIC state.
    pub(crate) ioapic: Mutex<Ioapic>,
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

/// VCPU (Virtual CPU) instance
pub struct Vcpu {
    /// id
    id: u32,
    /// Parent VM reference
    pub(crate) vm: Weak<Vm>,
    /// VCPU state
    pub(crate) state: Mutex<VcpuState>,
    /// VMCS physical address
    vmcs_phys: PhysAddr,
    /// IO bitmap A for trapping lower port range accesses.
    io_bitmap_a: Frame<()>,
    /// IO bitmap B for trapping upper port range accesses.
    io_bitmap_b: Frame<()>,
    /// MSR bitmap for trapping RDMSR/WRMSR accesses.
    msr_bitmap: Frame<()>,
}

/// VCPU state
#[derive(Debug, Default)]
pub struct VcpuState {
    /// General purpose registers
    pub regs: VcpuRegs,
    /// Special registers and descriptor tables provided by userspace.
    pub sregs: VcpuSregs,
    /// Running state
    pub running: bool,
    /// Vcpu has been launched
    pub launched: bool,
    /// VMCS initialized
    pub initialized: bool,
    /// Guest-visible MSR state emulated by the hypervisor.
    pub msrs: GuestMsrState,
    /// Pending exception injection state.
    pub exception: ExceptionState,
    /// Pending interrupt injection state.
    pub interrupt: InterruptState,
    /// Virtual LAPIC state for this vCPU.
    pub lapic: Lapic,
    /// APIC timer state.
    pub apic_timer: ApicTimer,
    /// TSC-tracking state for virtual timer emulation.
    pub tsc: TscState,
}

#[derive(Debug, Clone, Copy)]
pub struct GuestMsrState {
    pub apic_base: u64,
    pub efer: u64,
    pub pat: u64,
    pub fs_base: u64,
    pub gs_base: u64,
    pub kernel_gs_base: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub syscall_mask: u64,
    pub tsc_deadline: u64,
}

impl Default for GuestMsrState {
    fn default() -> Self {
        Self {
            apic_base: (0xFEE0_0000_u64) | (1 << 11) | (1 << 8),
            efer: Msr::IA32_EFER.read(),
            pat: Msr::IA32_PAT.read(),
            fs_base: 0,
            gs_base: 0,
            kernel_gs_base: 0,
            star: 0,
            lstar: 0,
            cstar: 0,
            syscall_mask: 0,
            tsc_deadline: 0,
        }
    }
}

impl Vm {
    /// Creates a new VM instance
    pub fn new(id: u32) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            id,
            memory_regions: Mutex::new(BTreeMap::new()),
            ept: Mutex::new(EptPageTable::new()?),
            vcpus: Mutex::new(BTreeMap::new()),
            ioapic: Mutex::new(Ioapic {
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
    pub fn set_memory_region(&self, region: MemoryRegion) -> Result<()> {
        self.map_memory_region(region)?;
        self.memory_regions.lock().insert(region.slot, region);
        Ok(())
    }

    /// Maps a user memory region
    pub fn map_memory_region(&self, region: MemoryRegion) -> Result<()> {
        use ostd::{
            mm::{vm_space::VmQueriedItem, HasPaddr, VmSpace, PAGE_SIZE},
            task::Task,
        };

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

        // Get the current task's VmSpace to translate user virtual address to host physical address
        let task = Task::current().ok_or(Error::with_message(
            Errno::NotFound,
            "No current task found",
        ))?;

        // Get the VmSpace from the current process
        let vm_space: &Arc<VmSpace> =
            task.data()
                .downcast_ref::<Arc<VmSpace>>()
                .ok_or(Error::with_message(
                    Errno::Fault,
                    "Failed to get VmSpace from current task",
                ))?;

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
        let preempt_guard = ostd::task::disable_preempt();
        let mut ept = self.ept.lock();
        let mut userspace_addr = region.userspace_addr;
        let mut guest_phys_addr = region.guest_phys_addr;

        while userspace_addr < userspace_end && guest_phys_addr < guest_end {
            let page_range = userspace_addr..(userspace_addr + PAGE_SIZE);
            let mut cursor = vm_space.cursor(&preempt_guard, &page_range)?;

            let hpa = match cursor.query()?.1 {
                Some(VmQueriedItem::MappedRam { frame, .. }) => frame.paddr(),
                Some(VmQueriedItem::MappedIoMem { paddr, .. }) => paddr,
                None => {
                    return Err(Error::with_message(
                        Errno::Fault,
                        "userspace page is not mapped",
                    ));
                }
            };

            ept.map_range(guest_phys_addr, hpa as _, PAGE_SIZE)?;
            userspace_addr += PAGE_SIZE;
            guest_phys_addr += PAGE_SIZE as u64;
        }

        Ok(())
    }

    pub fn get_eptp(&self) -> u64 {
        self.ept.lock().eptp()
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
}

impl Vcpu {
    pub fn new(id: u32, vm: Weak<Vm>) -> Result<Self> {
        use ostd::arch::virt::*;
        // Allocate VMCS
        let vmcs_phys = alloc_vmcs()?;
        let io_bitmap_a = FrameAllocOptions::new()
            .alloc_frame()
            .map_err(Error::from)?;
        let io_bitmap_b = FrameAllocOptions::new()
            .alloc_frame()
            .map_err(Error::from)?;
        let msr_bitmap = FrameAllocOptions::new()
            .alloc_frame()
            .map_err(Error::from)?;
        let all_ones = [0xff_u8; PAGE_SIZE];
        io_bitmap_a.write_bytes(0, &all_ones).map_err(Error::from)?;
        io_bitmap_b.write_bytes(0, &all_ones).map_err(Error::from)?;
        msr_bitmap.write_bytes(0, &all_ones).map_err(Error::from)?;
        let mut state = VcpuState::default();
        state.initialized = false;
        state.launched = false;
        state.msrs = GuestMsrState::default();
        if id != 0 {
            state.msrs.apic_base &= !(1 << 8);
        }
        state.lapic.id = id;
        state.lapic.ldr = (1_u32.checked_shl(id).unwrap_or(0)) << 24;
        state.apic_timer.lvt_timer_bits = 1 << 16;
        state.tsc.multiplier = (Msr::IA32_VMX_MISC.read() & 0x1f) as u8;

        Ok(Self {
            id,
            vm,
            vmcs_phys,
            io_bitmap_a,
            io_bitmap_b,
            msr_bitmap,
            state: Mutex::new(state),
        })
    }

    /// Runs the VCPU
    pub fn run(&self) -> Result<super::handler::RunStateMessage> {
        if !self.state.lock().initialized {
            let eptp = self
                .vm
                .upgrade()
                .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?
                .get_eptp();

            self.init(eptp)?;
        }

        use super::handler::vmexit_handler;
        loop {
            vmptrld(self.vmcs_phys)?;
            self.prepare_pending_events()?;
            let _irq_guard = ostd::irq::disable_local();
            self.prepare_guest_timing_before_entry()?;
            self.vmlaunch_or_vmresume()?;
            let exit_info = exit_info().map_err(Error::from)?;
            self.note_vmexit_tsc()?;
            if let Some(run_state) = vmexit_handler(self, &exit_info)? {
                return Ok(run_state);
            }
        }
    }

    fn init(&self, eptp: u64) -> Result<()> {
        vmclear(self.vmcs_phys)?;
        vmptrld(self.vmcs_phys)?;

        self.setup_vmcs(eptp)?;

        self.state.lock().initialized = true;

        Ok(())
    }

    /// Setup VMCS with initial guest state
    fn setup_vmcs(&self, eptp: u64) -> Result<()> {
        self.setup_vmcs_host()?;
        self.setup_vmcs_guest()?;
        self.setup_vmcs_controls(eptp)?;
        Ok(())
    }

    fn setup_vmcs_host(&self) -> Result<()> {
        VmcsHost64::IA32_PAT.write(Msr::IA32_PAT.read())?;
        VmcsHost64::IA32_EFER.write(Msr::IA32_EFER.read())?;

        VmcsHostNW::CR0.write(Cr0::read_raw() as _)?;
        VmcsHostNW::CR3.write(Cr3::read_raw().0.start_address().as_u64() as _)?; // TODO: check difference with JiaYuekai
        VmcsHostNW::CR4.write(Cr4::read_raw() as _)?;

        VmcsHost16::ES_SELECTOR.write(es().bits())?;
        VmcsHost16::CS_SELECTOR.write(cs().bits())?;
        VmcsHost16::SS_SELECTOR.write(ss().bits())?;
        VmcsHost16::DS_SELECTOR.write(ds().bits())?;
        VmcsHost16::FS_SELECTOR.write(fs().bits())?;
        VmcsHost16::GS_SELECTOR.write(gs().bits())?;
        VmcsHostNW::FS_BASE.write(Msr::IA32_FS_BASE.read() as _)?;
        VmcsHostNW::GS_BASE.write(Msr::IA32_GS_BASE.read() as _)?;

        let tr = tr();
        let mut gdtp = DescriptorTablePointer::default();
        let mut idtp = DescriptorTablePointer::default();
        sgdt(&mut gdtp);
        sidt(&mut idtp);

        VmcsHost16::TR_SELECTOR.write(tr.bits())?;
        VmcsHostNW::TR_BASE.write(get_tr_base(tr, &gdtp) as _)?;
        VmcsHostNW::GDTR_BASE.write(gdtp.base as _)?;
        VmcsHostNW::IDTR_BASE.write(idtp.base as _)?;
        VmcsHostNW::RIP.write(vm_exit_handler_virtaddr() as _)?;

        VmcsHostNW::IA32_SYSENTER_ESP.write(0)?;
        VmcsHostNW::IA32_SYSENTER_EIP.write(0)?;
        VmcsHost32::IA32_SYSENTER_CS.write(0)?;
        Ok(())
    }

    fn setup_vmcs_guest(&self) -> Result<()> {
        let state = self.state.lock();
        validate_guest_state(&state)?;

        let regs = state.regs;
        let sregs = state.sregs;
        let msrs = state.msrs;
        drop(state);

        let cr0_host_owned =
            Cr0Flags::NUMERIC_ERROR | Cr0Flags::NOT_WRITE_THROUGH | Cr0Flags::CACHE_DISABLE;
        let guest_cr0 = sanitize_guest_cr0(sregs.cr0);
        VmcsGuestNW::CR0.write(guest_cr0 as _)?;
        VmcsControlNW::CR0_GUEST_HOST_MASK.write((cr0_host_owned.bits()) as _)?;
        VmcsControlNW::CR0_READ_SHADOW.write(sregs.cr0 as _)?;

        let cr4_host_owned = Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS;
        let guest_cr4 = sanitize_guest_cr4(sregs.cr4);
        VmcsGuestNW::CR4.write(guest_cr4 as _)?;
        VmcsControlNW::CR4_GUEST_HOST_MASK.write(cr4_host_owned.bits() as _)?;
        VmcsControlNW::CR4_READ_SHADOW.write(sregs.cr4 as _)?;

        {
            use VmcsGuest16::*;
            use VmcsGuest32::*;
            use VmcsGuestNW::*;
            ES_SELECTOR.write(sregs.es.selector)?;
            ES_BASE.write(sregs.es.base as usize)?;
            ES_LIMIT.write(sregs.es.limit)?;
            ES_ACCESS_RIGHTS.write(segment_access_rights(&sregs.es))?;

            CS_SELECTOR.write(sregs.cs.selector)?;
            CS_BASE.write(sregs.cs.base as usize)?;
            CS_LIMIT.write(sregs.cs.limit)?;
            CS_ACCESS_RIGHTS.write(segment_access_rights(&sregs.cs))?;

            SS_SELECTOR.write(sregs.ss.selector)?;
            SS_BASE.write(sregs.ss.base as usize)?;
            SS_LIMIT.write(sregs.ss.limit)?;
            SS_ACCESS_RIGHTS.write(segment_access_rights(&sregs.ss))?;

            DS_SELECTOR.write(sregs.ds.selector)?;
            DS_BASE.write(sregs.ds.base as usize)?;
            DS_LIMIT.write(sregs.ds.limit)?;
            DS_ACCESS_RIGHTS.write(segment_access_rights(&sregs.ds))?;

            FS_SELECTOR.write(sregs.fs.selector)?;
            FS_BASE.write(sregs.fs.base as usize)?;
            FS_LIMIT.write(sregs.fs.limit)?;
            FS_ACCESS_RIGHTS.write(segment_access_rights(&sregs.fs))?;

            GS_SELECTOR.write(sregs.gs.selector)?;
            GS_BASE.write(sregs.gs.base as usize)?;
            GS_LIMIT.write(sregs.gs.limit)?;
            GS_ACCESS_RIGHTS.write(segment_access_rights(&sregs.gs))?;

            TR_SELECTOR.write(sregs.tr.selector)?;
            TR_BASE.write(sregs.tr.base as usize)?;
            TR_LIMIT.write(sregs.tr.limit)?;
            TR_ACCESS_RIGHTS.write(segment_access_rights(&sregs.tr))?;

            LDTR_SELECTOR.write(sregs.ldt.selector)?;
            LDTR_BASE.write(sregs.ldt.base as usize)?;
            LDTR_LIMIT.write(sregs.ldt.limit)?;
            LDTR_ACCESS_RIGHTS.write(segment_access_rights(&sregs.ldt))?;
        }

        VmcsGuestNW::GDTR_BASE.write(sregs.gdt.base as usize)?;
        VmcsGuest32::GDTR_LIMIT.write(sregs.gdt.limit as u32)?;
        VmcsGuestNW::IDTR_BASE.write(sregs.idt.base as usize)?;
        VmcsGuest32::IDTR_LIMIT.write(sregs.idt.limit as u32)?;

        VmcsGuestNW::CR3.write(sregs.cr3 as usize)?;
        VmcsGuestNW::DR7.write(0x400)?;
        VmcsGuestNW::RSP.write(regs.rsp as usize)?;
        VmcsGuestNW::RIP.write(regs.rip as usize)?;
        VmcsGuestNW::RFLAGS.write((regs.rflags | 0x2) as usize)?;
        VmcsGuestNW::PENDING_DBG_EXCEPTIONS.write(0)?;
        VmcsGuestNW::IA32_SYSENTER_ESP.write(0)?;
        VmcsGuestNW::IA32_SYSENTER_EIP.write(0)?;
        VmcsGuest32::IA32_SYSENTER_CS.write(0)?;

        VmcsGuest32::INTERRUPTIBILITY_STATE.write(0)?;
        VmcsGuest32::ACTIVITY_STATE.write(0)?;
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(0)?;

        VmcsGuest64::LINK_PTR.write(u64::MAX)?; // SDM Vol. 3C, Section 24.4.2
        VmcsGuest64::IA32_DEBUGCTL.write(0)?;
        VmcsGuest64::IA32_PAT.write(msrs.pat)?;
        VmcsGuest64::IA32_EFER.write(msrs.efer)?;
        VmcsControl64::TSC_OFFSET.write(0)?;
        Ok(())
    }

    fn setup_vmcs_controls(&self, eptp: u64) -> Result<()> {
        set_control(
            VmcsControl32::PINBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PINBASED_CTLS,
            Msr::IA32_VMX_PINBASED_CTLS.read() as u32,
            (PinbasedControls::from_bits_truncate(1 << 0)
                | PinbasedControls::NMI_EXITING
                | PinbasedControls::VMX_PREEMPTION_TIMER)
                .bits(),
            0,
        )?;

        set_control(
            VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PROCBASED_CTLS,
            Msr::IA32_VMX_PROCBASED_CTLS.read() as u32,
            (PrimaryControls::USE_TSC_OFFSETTING
                | PrimaryControls::USE_IO_BITMAPS
                | PrimaryControls::USE_MSR_BITMAPS
                | PrimaryControls::SECONDARY_CONTROLS)
                .bits(),
            (PrimaryControls::CR3_LOAD_EXITING | PrimaryControls::CR3_STORE_EXITING).bits(),
        )?;

        set_control(
            VmcsControl32::SECONDARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PROCBASED_CTLS2,
            0,
            (SecondaryControls::ENABLE_EPT
                | SecondaryControls::ENABLE_RDTSCP
                | SecondaryControls::UNRESTRICTED_GUEST)
                .bits(),
            0,
        )?;

        set_control(
            VmcsControl32::VMEXIT_CONTROLS,
            Msr::IA32_VMX_TRUE_EXIT_CTLS,
            Msr::IA32_VMX_EXIT_CTLS.read() as u32,
            (ExitControls::HOST_ADDRESS_SPACE_SIZE
                | ExitControls::SAVE_IA32_PAT
                | ExitControls::LOAD_IA32_PAT
                | ExitControls::SAVE_IA32_EFER
                | ExitControls::LOAD_IA32_EFER)
                .bits(),
            0,
        )?;

        let mut entry_controls =
            (EntryControls::LOAD_IA32_PAT | EntryControls::LOAD_IA32_EFER).bits();
        if self.state.lock().msrs.efer & X86_EFER_LMA != 0 {
            entry_controls |= EntryControls::IA32E_MODE_GUEST.bits();
        }

        set_control(
            VmcsControl32::VMENTRY_CONTROLS,
            Msr::IA32_VMX_TRUE_ENTRY_CTLS,
            Msr::IA32_VMX_ENTRY_CTLS.read() as u32,
            entry_controls,
            0,
        )?;

        // No MSR switches if hypervisor doesn't use and there is only one vCPU.
        VmcsControl32::VMEXIT_MSR_STORE_COUNT.write(0)?;
        VmcsControl32::VMEXIT_MSR_LOAD_COUNT.write(0)?;
        VmcsControl32::VMENTRY_MSR_LOAD_COUNT.write(0)?;

        // Pass-through exceptions. Intercept I/O and MSR accesses via bitmaps.
        VmcsControl32::EXCEPTION_BITMAP.write(0)?;
        VmcsControl64::IO_BITMAP_A_ADDR.write(self.io_bitmap_a.paddr() as u64)?;
        VmcsControl64::IO_BITMAP_B_ADDR.write(self.io_bitmap_b.paddr() as u64)?;
        VmcsControl64::MSR_BITMAPS_ADDR.write(self.msr_bitmap.paddr() as u64)?;

        // setup EPT
        VmcsControl64::EPTP.write(eptp)?;
        Ok(())
    }

    fn vmlaunch_or_vmresume(&self) -> Result<()> {
        let launched: u64 = if self.state.lock().launched { 1 } else { 0 };
        let ret = vcpu_run(&mut self.state.lock().regs, launched);
        if ret != 0 {
            return Err(Error::with_message(
                Errno::GuestRunFailed,
                "vcpu_run failed",
            ));
        }
        self.state.lock().launched = true;
        Ok(())
    }

    /// Queue a virtual interrupt for this vCPU's LAPIC.
    pub fn inject_interrupt(&self, vector: u32) -> Result<()> {
        if !(32..256).contains(&vector) {
            return Err(Error::with_message(
                Errno::InvalidArgs,
                "interrupt vector must be in the external-interrupt range [32, 255]",
            ));
        }

        let mut state = self.state.lock();
        lapic_set_irr(&mut state.lapic, vector as u8);
        Ok(())
    }

    /// Gets general purpose registers
    pub fn get_regs(&self) -> Result<VcpuRegs> {
        let state = self.state.lock();
        Ok(state.regs)
    }

    /// Sets general purpose registers
    pub fn set_regs(&self, regs: VcpuRegs) -> Result<()> {
        let mut state = self.state.lock();
        state.regs = regs;
        Ok(())
    }

    /// Gets special registers.
    pub fn get_sregs(&self) -> Result<VcpuSregs> {
        let state = self.state.lock();
        Ok(state.sregs)
    }

    /// Sets special registers.
    pub fn set_sregs(&self, sregs: VcpuSregs) -> Result<()> {
        let mut state = self.state.lock();
        state.sregs = sregs;
        state.msrs.efer = sregs.efer;
        state.msrs.fs_base = sregs.fs.base;
        state.msrs.gs_base = sregs.gs.base;
        Ok(())
    }

    pub(crate) fn emulate_cpuid(&self) -> Result<()> {
        let mut state = self.state.lock();
        let eax = state.regs.rax as u32;
        let ecx = state.regs.rcx as u32;
        let Some(CpuidResult {
            eax: eax_out,
            ebx: ebx_out,
            ecx: mut ecx_out,
            edx: edx_out,
        }) = cpuid(eax, ecx)
        else {
            state.regs.rax = 0;
            state.regs.rbx = 0;
            state.regs.rcx = 0;
            state.regs.rdx = 0;
            return Ok(());
        };

        if eax == 1 {
            ecx_out &= !(1 << 21);
            ecx_out &= !(1 << 5);
            ecx_out &= !(1 << 28);
            ecx_out &= !(1 << 26);
        }

        if eax == 7 && ecx == 0 {
            let masked_ebx = ebx_out & !(1 << 5);
            state.regs.rax = eax_out as u64;
            state.regs.rbx = masked_ebx as u64;
            state.regs.rcx = ecx_out as u64;
            state.regs.rdx = edx_out as u64;
            return Ok(());
        }

        state.regs.rax = eax_out as u64;
        state.regs.rbx = ebx_out as u64;
        state.regs.rcx = ecx_out as u64;
        state.regs.rdx = edx_out as u64;
        Ok(())
    }

    /// Record Guest TSC value at VM exit.
    pub(crate) fn note_vmexit_tsc(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.tsc.tsc_physical = state.tsc.tsc_offset.wrapping_add(read_tsc());
        Ok(())
    }

    /// Recompute the VMCS TSC offset right before VM-entry so the guest TSC
    /// resumes from the last frozen guest-visible value instead of jumping by
    /// the host time spent handling VM exits.
    pub(crate) fn freeze_guest_tsc(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.tsc.tsc_offset = state.tsc.tsc_physical.wrapping_sub(read_tsc());
        Ok(())
    }

    pub(crate) fn prepare_guest_timing_before_entry(&self) -> Result<()> {
        self.freeze_guest_tsc()?;

        let state = self.state.lock();
        let preemption_timer = compute_preemption_timer_value(&state);
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(preemption_timer)?;
        VmcsControl64::TSC_OFFSET.write(state.tsc.tsc_offset)?;
        Ok(())
    }

    pub(crate) fn handle_lapic_timer_write_effect(
        &self,
        effect: super::emulate::apic::LapicWriteEffect,
    ) -> Result<()> {
        let mut state = self.state.lock();
        match effect {
            super::emulate::apic::LapicWriteEffect::None => {}
            super::emulate::apic::LapicWriteEffect::StartTimer => {
                start_apic_timer_locked(&mut state);
            }
            super::emulate::apic::LapicWriteEffect::StartTimerDeadline => {
                start_apic_timer_deadline_locked(&mut state);
            }
        }
        Ok(())
    }

    pub(crate) fn handle_preemption_timer_expire(&self) -> Result<()> {
        let mut state = self.state.lock();
        if !state.tsc.activated {
            return Ok(());
        }

        let tsc = state.tsc;
        let action = lapic_on_timer_expire(&mut state.lapic, &mut state.apic_timer, &tsc);
        match action {
            TimerExpireAction::Rearm(deadline) => timer_activate_locked(&mut state, deadline),
            TimerExpireAction::Deactivate => timer_deactivate_locked(&mut state),
        }
        Ok(())
    }

    pub(crate) fn translate_guest_gpa(&self, gpa: u64) -> Result<u64> {
        let vm = self
            .vm
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::NotFound, "vm not found"))?;
        vm.ept
            .lock()
            .translate(gpa)
            .map_err(|_| Error::with_message(Errno::Fault, "guest GPA is not mapped in EPT"))
    }

    pub(crate) fn translate_guest_gva(&self, gva: u64) -> Result<u64> {
        const PTE_PRESENT: u64 = 1 << 0;
        const PTE_HUGE: u64 = 1 << 7;
        const PTE_ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;
        const PAGE_2M_MASK: u64 = (1 << 21) - 1;
        const PAGE_1G_MASK: u64 = (1 << 30) - 1;

        let cr0 = VmcsGuestNW::CR0.read().map_err(Error::from)? as u64;
        if (cr0 & (1 << 31)) == 0 {
            return Ok(gva);
        }

        let cr3 = (VmcsGuestNW::CR3.read().map_err(Error::from)? as u64) & !0xfff;
        let pml4e = self.read_guest_phys_u64(cr3 + (((gva >> 39) & 0x1ff) * 8))?;
        if (pml4e & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PML4 entry is not present",
            ));
        }

        let pdpte =
            self.read_guest_phys_u64((pml4e & PTE_ADDR_MASK) + (((gva >> 30) & 0x1ff) * 8))?;
        if (pdpte & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PDPT entry is not present",
            ));
        }
        if (pdpte & PTE_HUGE) != 0 {
            return Ok((pdpte & PTE_ADDR_MASK) | (gva & PAGE_1G_MASK));
        }

        let pde =
            self.read_guest_phys_u64((pdpte & PTE_ADDR_MASK) + (((gva >> 21) & 0x1ff) * 8))?;
        if (pde & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PD entry is not present",
            ));
        }
        if (pde & PTE_HUGE) != 0 {
            return Ok((pde & PTE_ADDR_MASK) | (gva & PAGE_2M_MASK));
        }

        let pte = self.read_guest_phys_u64((pde & PTE_ADDR_MASK) + (((gva >> 12) & 0x1ff) * 8))?;
        if (pte & PTE_PRESENT) == 0 {
            return Err(Error::with_message(
                Errno::Fault,
                "guest PT entry is not present",
            ));
        }

        Ok((pte & PTE_ADDR_MASK) | (gva & 0xfff))
    }

    pub(crate) fn read_guest_memory(&self, gva: u64, buf: &mut [u8]) -> Result<()> {
        for (index, byte) in buf.iter_mut().enumerate() {
            let gpa = self.translate_guest_gva(gva.wrapping_add(index as u64))?;
            let hpa = self.translate_guest_gpa(gpa)?;
            read_bytes_from_paddr(hpa as usize, core::slice::from_mut(byte));
        }
        Ok(())
    }

    fn read_guest_phys_u64(&self, gpa: u64) -> Result<u64> {
        let hpa = self.translate_guest_gpa(gpa)?;
        Ok(read_u64_from_paddr(hpa as usize))
    }

    fn prepare_pending_events(&self) -> Result<()> {
        let mut state = self.state.lock();

        if has_pending_exception(&state.exception) {
            let mut perf = [0_u32; 32];
            inject_pending_exception(&mut state.exception, &mut perf)?;
            return Ok(());
        }

        try_inject_pending_interrupt(&mut state.lapic, &mut state.interrupt)?;
        if state.interrupt.pending {
            return Ok(());
        }

        if let Some(vector) = lapic_check_pending_vector(&state.lapic) {
            let mut perf = [0_u32; 224];
            inject_interrupt(
                &mut state.lapic,
                &mut state.interrupt,
                &mut perf,
                u32::from(vector),
            )?;
        }

        Ok(())
    }
}

fn timer_deactivate_locked(state: &mut VcpuState) {
    state.tsc.activated = false;
    state.tsc.ddl_physical = 0;
}

fn timer_activate_locked(state: &mut VcpuState, deadline_ticks: u64) {
    state.tsc.activated = true;
    if deadline_ticks > state.tsc.tsc_physical {
        state.tsc.ddl_physical = deadline_ticks;
    } else {
        let tsc = state.tsc;
        let action = lapic_on_timer_expire(&mut state.lapic, &mut state.apic_timer, &tsc);
        match action {
            TimerExpireAction::Rearm(next_deadline) => timer_activate_locked(state, next_deadline),
            TimerExpireAction::Deactivate => timer_deactivate_locked(state),
        }
    }
}

fn start_apic_timer_locked(state: &mut VcpuState) {
    if let Some(deadline) = lapic_timer_deadline(&state.apic_timer, &state.tsc) {
        timer_activate_locked(state, deadline);
    } else {
        timer_deactivate_locked(state);
    }
}

fn start_apic_timer_deadline_locked(state: &mut VcpuState) {
    if let Some(deadline) = lapic_timer_deadline_tsc(&state.apic_timer, state.msrs.tsc_deadline) {
        timer_activate_locked(state, deadline);
    } else {
        timer_deactivate_locked(state);
    }
}

fn compute_preemption_timer_value(state: &VcpuState) -> u32 {
    let ticks = if state.tsc.activated {
        state
            .tsc
            .ddl_physical
            .saturating_sub(state.tsc.tsc_physical)
    } else {
        tsc_freq() / 20
    };

    let shifted = ticks >> state.tsc.multiplier;
    let shifted = if shifted == 0 { 1 } else { shifted };
    shifted.min(u64::from(u32::MAX)) as u32
}

fn sanitize_guest_cr0(value: u64) -> u64 {
    value | X86_CR0_ET | X86_CR0_NE
}

fn sanitize_guest_cr4(value: u64) -> u64 {
    value & !X86_CR4_VMXE
}

fn segment_access_rights(segment: &VcpuSegment) -> u32 {
    let mut rights = u32::from(segment.type_ & 0x0f);
    rights |= u32::from(segment.s & 0x1) << 4;
    rights |= u32::from(segment.dpl & 0x3) << 5;
    rights |= u32::from(segment.present & 0x1) << 7;
    rights |= u32::from(segment.avl & 0x1) << 12;
    rights |= u32::from(segment.l & 0x1) << 13;
    rights |= u32::from(segment.db & 0x1) << 14;
    rights |= u32::from(segment.g & 0x1) << 15;
    rights |= u32::from(segment.unusable & 0x1) << 16;
    rights
}

fn validate_guest_state(state: &VcpuState) -> Result<()> {
    if state.sregs.cs.present == 0 {
        return Err(Error::with_message(
            Errno::InvalidArgs,
            "guest special registers must be configured before first run",
        ));
    }

    if state.regs.rip == 0 {
        return Err(Error::with_message(
            Errno::InvalidArgs,
            "guest RIP must be configured before first run",
        ));
    }

    Ok(())
}
