//! VM (Virtual Machine) management for RustShyper

use alloc::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};
use core::{
    arch::x86_64::CpuidResult,
    sync::atomic::{AtomicU32, Ordering},
};

use ostd::{
    arch::{
        cpu::{context::FpuContext, cpuid::cpuid},
        read_tsc, tsc_freq,
        virt::*,
    },
    mm::{
        kspace::{read_bytes_from_paddr, read_u64_from_paddr},
        vm_space::VmQueriedItem,
        Frame, FrameAllocOptions, HasPaddr, VmIo, VmSpace, PAGE_SIZE,
    },
    sync::{Mutex, SpinLock},
};
use x86::vmx::vmcs::control::{
    EntryControls, ExitControls, PinbasedControls, PrimaryControls, SecondaryControls,
};
use x86_64::registers::control::{Cr0, Cr0Flags, Cr3, Cr4, Cr4Flags};

use super::{
    emulate::apic::{
        ioapic_kick_irq, lapic_check_pending_vector, lapic_on_timer_expire,
        lapic_timer_deadline, lapic_timer_deadline_tsc, ApicTimer, Ioapic, Lapic,
        TimerExpireAction, TscState, IOAPIC_NUM_PINS,
    },
    error::*,
    interrupt::{
        clear_event_injection, clear_interrupt_shadow_after_hlt, has_pending_exception,
        inject_lapic_interrupt, inject_pending_exception, queue_external_interrupt,
        try_inject_pending_interrupt, ExceptionState, InterruptState,
    },
};

const X86_CR0_PE: u64 = 1 << 0;
const X86_CR0_ET: u64 = 1 << 4;
const X86_CR0_NE: u64 = 1 << 5;
const X86_CR4_VMXE: u64 = 1 << 13;
const X86_CR4_FSGSBASE: u64 = 1 << 16;
const X86_CR0_PG: u64 = 1 << 31;
const X86_EFER_LME: u64 = 1 << 8;
const X86_EFER_LMA: u64 = 1 << 10;
const PRIMARY_CTL_PAUSE_EXITING: u32 = 1 << 30;
const VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK: u8 = 0;
const VMX_PREEMPTION_TIMER_POLL_VALUE: u32 = 1_000;
const HLT_WAIT_MAX_TSC_FREQ_DIVISOR: u64 = 100;
const HLT_WAIT_MAX_FALLBACK_TICKS: u64 = 25_000_000;
const CPUID_TSC_CRYSTAL_HZ: u32 = 1_000_000;

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

/// VCPU (Virtual CPU) instance
pub struct Vcpu {
    /// id
    id: u32,
    /// Parent VM reference
    pub(crate) vm: Weak<Vm>,
    /// VCPU state
    pub(crate) state: SpinLock<VcpuState>,
    /// VMCS physical address
    vmcs_phys: PhysAddr,
    /// IO bitmap A for trapping lower port range accesses.
    io_bitmap_a: Frame<()>,
    /// IO bitmap B for trapping upper port range accesses.
    io_bitmap_b: Frame<()>,
    /// MSR bitmap for trapping RDMSR/WRMSR accesses.
    msr_bitmap: Frame<()>,
    /// Guest-owned FPU/SIMD context.
    guest_fpu: SpinLock<FpuContext>,
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
struct HostRunMsrs {
    star: u64,
    lstar: u64,
    cstar: u64,
    syscall_mask: u64,
    kernel_gs_base: u64,
}

impl HostRunMsrs {
    fn read_current() -> Self {
        Self {
            star: Msr::IA32_STAR.read(),
            lstar: Msr::IA32_LSTAR.read(),
            cstar: Msr::IA32_CSTAR.read(),
            syscall_mask: Msr::IA32_FMASK.read(),
            kernel_gs_base: Msr::IA32_KERNEL_GSBASE.read(),
        }
    }

    fn restore(self) {
        Msr::IA32_STAR.write(self.star);
        Msr::IA32_LSTAR.write(self.lstar);
        Msr::IA32_CSTAR.write(self.cstar);
        Msr::IA32_FMASK.write(self.syscall_mask);
        Msr::IA32_KERNEL_GSBASE.write(self.kernel_gs_base);
    }
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
    pub tsc_aux: u64,
    pub sysenter_cs: u64,
    pub sysenter_esp: u64,
    pub sysenter_eip: u64,
}

impl Default for GuestMsrState {
    fn default() -> Self {
        const APIC_BASE_BSP: u64 = 1 << 8;
        const APIC_BASE_ENABLE: u64 = 1 << 11;

        Self {
            apic_base: 0xFEE0_0000_u64 | APIC_BASE_BSP | APIC_BASE_ENABLE,
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
            tsc_aux: 0,
            sysenter_cs: 0,
            sysenter_esp: 0,
            sysenter_eip: 0,
        }
    }
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

impl Vcpu {
    /// Gets the VCPU ID
    pub fn id(&self) -> u32 {
        self.id
    }

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
        // Some virtualized environments expose enough VMX state to run the guest
        // but still #GP on RDMSR IA32_VMX_MISC (0x485). Use a marker value and
        // poll active virtual deadlines at a bounded interval.
        state.tsc.multiplier = VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK;

        Ok(Self {
            id,
            vm,
            vmcs_phys,
            io_bitmap_a,
            io_bitmap_b,
            msr_bitmap,
            guest_fpu: SpinLock::new(FpuContext::new()),
            state: SpinLock::new(state),
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

            let regs = self.get_regs()?;
            log::info!(
                "rustshyper: initializing vcpu id={} vmcs={:#x} eptp={:#x} rip={:#x} rsp={:#x} rflags={:#x}",
                self.id,
                self.vmcs_phys,
                eptp,
                regs.rip,
                regs.rsp,
                regs.rflags
            );
            self.init(eptp)?;
        }

        use super::handler::vmexit_handler;
        let mut host_fpu = FpuContext::new();
        loop {
            vmptrld(self.vmcs_phys)?;
            self.prepare_pending_events()?;
            let (exit_info, run_state) = {
                let _irq_guard = ostd::irq::disable_local();
                self.prepare_guest_timing_before_entry()?;
                let host_cr2 = read_cr2_raw();
                self.load_guest_cr2();
                let host_run_msrs = HostRunMsrs::read_current();
                self.load_guest_run_msrs();
                let run_result = {
                    let mut guest_fpu = self.guest_fpu.lock();
                    host_fpu.save();
                    guest_fpu.load();
                    let run_result = self.vmlaunch_or_vmresume();
                    guest_fpu.save();
                    host_fpu.load();
                    run_result
                };
                let msr_sync_result = self.save_guest_run_msrs();
                host_run_msrs.restore();
                let guest_cr2 = read_cr2_raw();
                write_cr2_raw(host_cr2);
                self.save_guest_cr2(guest_cr2);
                run_result?;
                msr_sync_result?;
                let exit_info = exit_info().map_err(Error::from)?;
                self.note_vmexit_tsc()?;
                let run_state = vmexit_handler(self, &exit_info)?;
                (exit_info, run_state)
            };

            if matches!(
                VmxExitReason::try_from(exit_info.exit_reason),
                Ok(VmxExitReason::HLT)
            ) && run_state.is_some()
                && self.wait_for_hlt_wakeup()?
            {
                clear_interrupt_shadow_after_hlt()?;
                super::handler::advance_guest_rip()?;
                continue;
            }

            if let Some(run_state) = run_state {
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

        let cr0_host_owned = Cr0Flags::from_bits_truncate(X86_CR0_PE | X86_CR0_PG)
            | Cr0Flags::NUMERIC_ERROR
            | Cr0Flags::NOT_WRITE_THROUGH
            | Cr0Flags::CACHE_DISABLE;
        let guest_cr0 = sanitize_guest_cr0(sregs.cr0);
        VmcsGuestNW::CR0.write(guest_cr0 as _)?;
        VmcsControlNW::CR0_GUEST_HOST_MASK.write((cr0_host_owned.bits()) as _)?;
        VmcsControlNW::CR0_READ_SHADOW.write(sregs.cr0 as _)?;

        let cr4_host_owned = Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS.bits() | X86_CR4_FSGSBASE;
        let guest_cr4 = sanitize_guest_cr4(sregs.cr4);
        VmcsGuestNW::CR4.write(guest_cr4 as _)?;
        VmcsControlNW::CR4_GUEST_HOST_MASK.write(cr4_host_owned as _)?;
        VmcsControlNW::CR4_READ_SHADOW.write(guest_cr4_read_shadow(sregs.cr4) as _)?;

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
        VmcsGuestNW::IA32_SYSENTER_ESP.write(msrs.sysenter_esp as usize)?;
        VmcsGuestNW::IA32_SYSENTER_EIP.write(msrs.sysenter_eip as usize)?;
        VmcsGuest32::IA32_SYSENTER_CS.write(msrs.sysenter_cs as u32)?;

        VmcsGuest32::INTERRUPTIBILITY_STATE.write(0)?;
        VmcsGuest32::ACTIVITY_STATE.write(0)?;
        VmcsGuest32::VMX_PREEMPTION_TIMER_VALUE.write(0)?;

        VmcsGuest64::LINK_PTR.write(u64::MAX)?; // SDM Vol. 3C, Section 24.4.2
        VmcsGuest64::IA32_DEBUGCTL.write(0)?;
        VmcsGuest64::IA32_PAT.write(msrs.pat)?;
        VmcsGuest64::IA32_EFER.write(sanitize_guest_efer(msrs.efer, sregs.cr0))?;
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
                | PrimaryControls::HLT_EXITING
                | PrimaryControls::USE_IO_BITMAPS
                | PrimaryControls::USE_MSR_BITMAPS
                | PrimaryControls::SECONDARY_CONTROLS)
                .bits()
                | PRIMARY_CTL_PAUSE_EXITING,
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
        let state = self.state.lock();
        if guest_ia32e_mode_active(state.msrs.efer, state.sregs.cr0) {
            entry_controls |= EntryControls::IA32E_MODE_GUEST.bits();
        }
        drop(state);

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
            let vm_instruction_error = VmcsReadOnly32::VM_INSTRUCTION_ERROR.read().ok();
            let exit_reason = VmcsReadOnly32::EXIT_REASON.read().ok();
            let guest_rip = VmcsGuestNW::RIP.read().ok();
            let guest_rsp = VmcsGuestNW::RSP.read().ok();
            let guest_rflags = VmcsGuestNW::RFLAGS.read().ok();
            log::error!(
                "rustshyper: {} failed for vcpu id={} vmcs={:#x} vm_instruction_error={:#x?} exit_reason={:#x?} guest_rip={:#x?} guest_rsp={:#x?} guest_rflags={:#x?}",
                if launched == 0 { "vmlaunch" } else { "vmresume" },
                self.id,
                self.vmcs_phys,
                vm_instruction_error,
                exit_reason,
                guest_rip,
                guest_rsp,
                guest_rflags
            );
            return Err(Error::with_message(
                Errno::GuestRunFailed,
                "vcpu_run failed",
            ));
        }
        self.state.lock().launched = true;
        Ok(())
    }

    /// Queues a virtual external interrupt for this vCPU.
    pub fn inject_interrupt(&self, vector: u32) -> Result<()> {
        let mut state = self.state.lock();
        queue_external_interrupt(&mut state.interrupt, vector)
    }

    /// Gets general purpose registers
    pub fn get_regs(&self) -> Result<VcpuRegs> {
        let mut state = self.state.lock();
        if state.initialized {
            vmptrld(self.vmcs_phys)?;
            state.regs.rsp = VmcsGuestNW::RSP.read().map_err(Error::from)? as u64;
            state.regs.rip = VmcsGuestNW::RIP.read().map_err(Error::from)? as u64;
            state.regs.rflags = VmcsGuestNW::RFLAGS.read().map_err(Error::from)? as u64;
        }
        Ok(state.regs)
    }

    /// Sets general purpose registers
    pub fn set_regs(&self, regs: VcpuRegs) -> Result<()> {
        let mut state = self.state.lock();
        state.regs = regs;
        if state.initialized {
            drop(state);
            vmptrld(self.vmcs_phys)?;
            VmcsGuestNW::RSP.write(regs.rsp as usize)
                .map_err(Error::from)?;
            VmcsGuestNW::RIP.write(regs.rip as usize)
                .map_err(Error::from)?;
            VmcsGuestNW::RFLAGS
                .write((regs.rflags | 0x2) as usize)
                .map_err(Error::from)?;
        }
        Ok(())
    }

    /// Gets special registers.
    pub fn get_sregs(&self) -> Result<VcpuSregs> {
        let mut state = self.state.lock();
        if state.initialized {
            vmptrld(self.vmcs_phys)?;
            sync_sregs_from_vmcs(&mut state.sregs)?;
            state.sregs.apic_base = state.msrs.apic_base;
        }
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

    pub(crate) fn guest_cr2(&self) -> u64 {
        self.state.lock().sregs.cr2
    }

    pub(crate) fn set_guest_cr2(&self, value: u64) {
        self.state.lock().sregs.cr2 = value;
    }

    pub(crate) fn emulate_cpuid(&self) -> Result<()> {
        const CPUID_1_ECX_VMX: u32 = 1 << 5;
        const CPUID_1_ECX_FMA: u32 = 1 << 12;
        const CPUID_1_ECX_PCID: u32 = 1 << 17;
        const CPUID_1_ECX_XSAVE: u32 = 1 << 26;
        const CPUID_1_ECX_OSXSAVE: u32 = 1 << 27;
        const CPUID_1_ECX_AVX: u32 = 1 << 28;
        const CPUID_7_EBX_FSGSBASE: u32 = 1 << 0;
        const CPUID_7_EBX_HLE: u32 = 1 << 4;
        const CPUID_7_EBX_AVX2: u32 = 1 << 5;
        const CPUID_7_EBX_RTM: u32 = 1 << 11;
        const CPUID_7_EBX_INVPCID: u32 = 1 << 10;
        const CPUID_7_EBX_AVX512F: u32 = 1 << 16;
        const CPUID_7_EBX_AVX512DQ: u32 = 1 << 17;
        const CPUID_7_EBX_AVX512CD: u32 = 1 << 28;
        const CPUID_7_EBX_AVX512BW: u32 = 1 << 30;
        const CPUID_7_EBX_AVX512VL: u32 = 1 << 31;
        const CPUID_7_ECX_AVX512VBMI: u32 = 1 << 1;
        const CPUID_7_ECX_VAES: u32 = 1 << 9;
        const CPUID_7_ECX_VPCLMULQDQ: u32 = 1 << 10;
        const CPUID_7_ECX_AVX512VNNI: u32 = 1 << 11;
        const CPUID_7_ECX_AVX512BITALG: u32 = 1 << 12;
        const CPUID_7_ECX_AVX512VPOPCNTDQ: u32 = 1 << 14;

        let mut state = self.state.lock();
        let eax = state.regs.rax as u32;
        let ecx = state.regs.rcx as u32;
        let (
            mut eax_out,
            mut ebx_out,
            mut ecx_out,
            mut edx_out,
        ) = if let Some(CpuidResult { eax, ebx, ecx, edx }) = cpuid(eax, ecx) {
            (eax, ebx, ecx, edx)
        } else {
            (0, 0, 0, 0)
        };

        if eax == 0 {
            eax_out = eax_out.max(0x16);
        }

        if eax == 1 {
            ecx_out &= !(CPUID_1_ECX_VMX
                | CPUID_1_ECX_FMA
                | CPUID_1_ECX_PCID
                | CPUID_1_ECX_XSAVE
                | CPUID_1_ECX_OSXSAVE
                | CPUID_1_ECX_AVX);
        }

        if eax == 7 && ecx == 0 {
            ebx_out &= !(CPUID_7_EBX_FSGSBASE
                | CPUID_7_EBX_HLE
                | CPUID_7_EBX_AVX2
                | CPUID_7_EBX_RTM
                | CPUID_7_EBX_INVPCID
                | CPUID_7_EBX_AVX512F
                | CPUID_7_EBX_AVX512DQ
                | CPUID_7_EBX_AVX512CD
                | CPUID_7_EBX_AVX512BW
                | CPUID_7_EBX_AVX512VL);
            ecx_out &= !(CPUID_7_ECX_AVX512VBMI
                | CPUID_7_ECX_VAES
                | CPUID_7_ECX_VPCLMULQDQ
                | CPUID_7_ECX_AVX512VNNI
                | CPUID_7_ECX_AVX512BITALG
                | CPUID_7_ECX_AVX512VPOPCNTDQ);
        }

        if eax == 0xd {
            eax_out = 0;
            ebx_out = 0;
            ecx_out = 0;
            edx_out = 0;
        }

        if eax == 0x15 {
            if let Some(tsc_mhz) = virtual_tsc_mhz() {
                eax_out = 1;
                ebx_out = tsc_mhz;
                ecx_out = CPUID_TSC_CRYSTAL_HZ;
                edx_out = 0;
            }
        }

        if eax == 0x16 {
            if let Some(tsc_mhz) = virtual_tsc_mhz() {
                eax_out = tsc_mhz;
                ebx_out = tsc_mhz;
                ecx_out = 0;
                edx_out = 0;
            }
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

    /// Refreshes the guest-visible TSC before VM-entry.
    pub(crate) fn refresh_guest_tsc(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.tsc.tsc_physical = state.tsc.tsc_offset.wrapping_add(read_tsc());
        Ok(())
    }

    fn load_guest_cr2(&self) {
        let cr2 = self.state.lock().sregs.cr2;
        write_cr2_raw(cr2);
    }

    fn save_guest_cr2(&self, cr2: u64) {
        self.state.lock().sregs.cr2 = cr2;
    }

    fn load_guest_run_msrs(&self) {
        let state = self.state.lock();

        Msr::IA32_STAR.write(state.msrs.star);
        Msr::IA32_LSTAR.write(state.msrs.lstar);
        Msr::IA32_CSTAR.write(state.msrs.cstar);
        Msr::IA32_FMASK.write(state.msrs.syscall_mask);
        Msr::IA32_KERNEL_GSBASE.write(state.msrs.kernel_gs_base);
    }

    fn save_guest_run_msrs(&self) -> Result<()> {
        let star = Msr::IA32_STAR.read();
        let lstar = Msr::IA32_LSTAR.read();
        let cstar = Msr::IA32_CSTAR.read();
        let syscall_mask = Msr::IA32_FMASK.read();
        let kernel_gs_base = Msr::IA32_KERNEL_GSBASE.read();
        let fs_base = VmcsGuestNW::FS_BASE.read().map_err(Error::from)? as u64;
        let gs_base = VmcsGuestNW::GS_BASE.read().map_err(Error::from)? as u64;

        let mut state = self.state.lock();
        state.msrs.star = star;
        state.msrs.lstar = lstar;
        state.msrs.cstar = cstar;
        state.msrs.syscall_mask = syscall_mask;
        state.msrs.kernel_gs_base = kernel_gs_base;
        state.msrs.fs_base = fs_base;
        state.msrs.gs_base = gs_base;
        state.sregs.fs.base = fs_base;
        state.sregs.gs.base = gs_base;
        Ok(())
    }

    pub(crate) fn prepare_guest_timing_before_entry(&self) -> Result<()> {
        self.refresh_guest_tsc()?;

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
        if state.tsc.ddl_physical > state.tsc.tsc_physical {
            return Ok(());
        }

        expire_lapic_timer_locked(&mut state);
        Ok(())
    }

    pub(crate) fn poll_expired_lapic_timer(&self) -> Result<bool> {
        let mut state = self.state.lock();
        if !state.tsc.activated || state.tsc.ddl_physical > state.tsc.tsc_physical {
            return Ok(false);
        }

        expire_lapic_timer_locked(&mut state);
        Ok(true)
    }

    pub(crate) fn wait_for_hlt_wakeup(&self) -> Result<bool> {
        {
            let mut state = self.state.lock();
            refresh_tsc_locked(&mut state);
            if state.interrupt.pending || lapic_check_pending_vector(&state.lapic).is_some() {
                return Ok(true);
            }
            if !state.tsc.activated {
                return Ok(false);
            }
            if state.tsc.ddl_physical <= state.tsc.tsc_physical {
                expire_lapic_timer_locked(&mut state);
                return Ok(true);
            }
        }

        let wait_started_tsc = read_tsc();
        let max_wait_ticks = hlt_wait_max_ticks();
        loop {
            let mut state = self.state.lock();
            refresh_tsc_locked(&mut state);
            if state.interrupt.pending || lapic_check_pending_vector(&state.lapic).is_some() {
                return Ok(true);
            }
            if state.tsc.activated && state.tsc.ddl_physical <= state.tsc.tsc_physical {
                expire_lapic_timer_locked(&mut state);
                return Ok(true);
            }
            debug_assert!(state.tsc.activated);
            let raw_tsc = read_tsc();
            if raw_tsc.saturating_sub(wait_started_tsc) >= max_wait_ticks {
                return Ok(false);
            }
            drop(state);

            core::hint::spin_loop();
        }
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

        clear_event_injection()?;

        if has_pending_exception(&state.exception) {
            let mut perf = [0_u32; 32];
            inject_pending_exception(&mut state.exception, &mut perf)?;
            return Ok(());
        }

        {
            let VcpuState {
                lapic, interrupt, ..
            } = &mut *state;
            try_inject_pending_interrupt(lapic, interrupt)?;
        }
        if state.interrupt.pending {
            return Ok(());
        }

        if let Some(vector) = lapic_check_pending_vector(&state.lapic) {
            let mut perf = [0_u32; 224];
            inject_lapic_interrupt(&mut state.lapic, &mut perf, u32::from(vector))?;
        }

        Ok(())
    }
}

fn timer_deactivate_locked(state: &mut VcpuState) {
    state.tsc.activated = false;
    state.tsc.ddl_physical = 0;
}

fn refresh_tsc_locked(state: &mut VcpuState) {
    state.tsc.tsc_physical = state.tsc.tsc_offset.wrapping_add(read_tsc());
}

fn expire_lapic_timer_locked(state: &mut VcpuState) {
    let tsc = state.tsc;
    let action = {
        let VcpuState {
            lapic, apic_timer, ..
        } = state;
        lapic_on_timer_expire(lapic, apic_timer, &tsc)
    };
    match action {
        TimerExpireAction::Rearm(deadline) => timer_activate_locked(state, deadline),
        TimerExpireAction::Deactivate => timer_deactivate_locked(state),
    }
}

fn virtual_tsc_mhz() -> Option<u32> {
    let mhz = (tsc_freq().saturating_add(500_000)) / 1_000_000;
    u32::try_from(mhz).ok().filter(|&mhz| mhz != 0)
}

fn hlt_wait_max_ticks() -> u64 {
    match tsc_freq() {
        0 => HLT_WAIT_MAX_FALLBACK_TICKS,
        freq => (freq / HLT_WAIT_MAX_TSC_FREQ_DIVISOR).max(1),
    }
}

fn timer_activate_locked(state: &mut VcpuState, deadline_ticks: u64) {
    state.tsc.activated = true;
    if deadline_ticks > state.tsc.tsc_physical {
        state.tsc.ddl_physical = deadline_ticks;
    } else {
        let tsc = state.tsc;
        let action = {
            let VcpuState {
                lapic, apic_timer, ..
            } = &mut *state;
            lapic_on_timer_expire(lapic, apic_timer, &tsc)
        };
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
    if !state.tsc.activated {
        return VMX_PREEMPTION_TIMER_POLL_VALUE;
    }

    let ticks = state
        .tsc
        .ddl_physical
        .saturating_sub(state.tsc.tsc_physical);

    if state.tsc.multiplier == VMX_PREEMPTION_TIMER_MULTIPLIER_FALLBACK {
        return ticks
            .min(u64::from(VMX_PREEMPTION_TIMER_POLL_VALUE))
            .max(1) as u32;
    }

    let shifted = ticks >> state.tsc.multiplier;
    let shifted = if shifted == 0 { 1 } else { shifted };
    shifted.min(u64::from(VMX_PREEMPTION_TIMER_POLL_VALUE)) as u32
}

pub(super) fn sanitize_guest_cr0(value: u64) -> u64 {
    let mut fixed0 = Msr::IA32_VMX_CR0_FIXED0.read();
    let fixed1 = Msr::IA32_VMX_CR0_FIXED1.read();

    // unrestricted guest relaxes PE/PG only
    fixed0 &= !(X86_CR0_PE | X86_CR0_PG);

    (value | fixed0 | X86_CR0_ET | X86_CR0_NE) & fixed1
}

pub(super) fn sanitize_guest_cr4(value: u64) -> u64 {
    let fixed0 = Msr::IA32_VMX_CR4_FIXED0.read();
    let fixed1 = Msr::IA32_VMX_CR4_FIXED1.read();

    // actual hardware CR4 in VMCS must satisfy fixed bits
    ((value | fixed0 | X86_CR4_VMXE) & fixed1) & !X86_CR4_FSGSBASE
}

pub(super) fn guest_cr4_read_shadow(value: u64) -> u64 {
    value & !X86_CR4_FSGSBASE
}

pub(super) fn sanitize_guest_efer(value: u64, guest_cr0: u64) -> u64 {
    let mut actual = value;

    if (value & X86_EFER_LME) != 0 && (guest_cr0 & X86_CR0_PG) != 0 {
        actual |= X86_EFER_LMA;
    } else {
        actual &= !X86_EFER_LMA;
    }

    actual
}

pub(super) fn guest_ia32e_mode_active(value: u64, guest_cr0: u64) -> bool {
    (sanitize_guest_efer(value, guest_cr0) & X86_EFER_LMA) != 0
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

fn sync_sregs_from_vmcs(sregs: &mut VcpuSregs) -> Result<()> {
    let cr0 = VmcsGuestNW::CR0.read().map_err(Error::from)? as u64;
    let cr0_mask = VmcsControlNW::CR0_GUEST_HOST_MASK
        .read()
        .map_err(Error::from)? as u64;
    let cr0_shadow = VmcsControlNW::CR0_READ_SHADOW
        .read()
        .map_err(Error::from)? as u64;
    sregs.cr0 = merge_guest_control_register(cr0, cr0_mask, cr0_shadow);

    let cr4 = VmcsGuestNW::CR4.read().map_err(Error::from)? as u64;
    let cr4_mask = VmcsControlNW::CR4_GUEST_HOST_MASK
        .read()
        .map_err(Error::from)? as u64;
    let cr4_shadow = VmcsControlNW::CR4_READ_SHADOW
        .read()
        .map_err(Error::from)? as u64;
    sregs.cr4 = merge_guest_control_register(cr4, cr4_mask, cr4_shadow);

    sregs.cr3 = VmcsGuestNW::CR3.read().map_err(Error::from)? as u64;
    sregs.efer = VmcsGuest64::IA32_EFER.read().map_err(Error::from)?;
    sregs.gdt = read_dtable_from_vmcs(VmcsGuestNW::GDTR_BASE, VmcsGuest32::GDTR_LIMIT)?;
    sregs.idt = read_dtable_from_vmcs(VmcsGuestNW::IDTR_BASE, VmcsGuest32::IDTR_LIMIT)?;

    sregs.cs = read_segment_from_vmcs(
        VmcsGuest16::CS_SELECTOR,
        VmcsGuestNW::CS_BASE,
        VmcsGuest32::CS_LIMIT,
        VmcsGuest32::CS_ACCESS_RIGHTS,
    )?;
    sregs.ds = read_segment_from_vmcs(
        VmcsGuest16::DS_SELECTOR,
        VmcsGuestNW::DS_BASE,
        VmcsGuest32::DS_LIMIT,
        VmcsGuest32::DS_ACCESS_RIGHTS,
    )?;
    sregs.es = read_segment_from_vmcs(
        VmcsGuest16::ES_SELECTOR,
        VmcsGuestNW::ES_BASE,
        VmcsGuest32::ES_LIMIT,
        VmcsGuest32::ES_ACCESS_RIGHTS,
    )?;
    sregs.fs = read_segment_from_vmcs(
        VmcsGuest16::FS_SELECTOR,
        VmcsGuestNW::FS_BASE,
        VmcsGuest32::FS_LIMIT,
        VmcsGuest32::FS_ACCESS_RIGHTS,
    )?;
    sregs.gs = read_segment_from_vmcs(
        VmcsGuest16::GS_SELECTOR,
        VmcsGuestNW::GS_BASE,
        VmcsGuest32::GS_LIMIT,
        VmcsGuest32::GS_ACCESS_RIGHTS,
    )?;
    sregs.ss = read_segment_from_vmcs(
        VmcsGuest16::SS_SELECTOR,
        VmcsGuestNW::SS_BASE,
        VmcsGuest32::SS_LIMIT,
        VmcsGuest32::SS_ACCESS_RIGHTS,
    )?;
    sregs.tr = read_segment_from_vmcs(
        VmcsGuest16::TR_SELECTOR,
        VmcsGuestNW::TR_BASE,
        VmcsGuest32::TR_LIMIT,
        VmcsGuest32::TR_ACCESS_RIGHTS,
    )?;
    sregs.ldt = read_segment_from_vmcs(
        VmcsGuest16::LDTR_SELECTOR,
        VmcsGuestNW::LDTR_BASE,
        VmcsGuest32::LDTR_LIMIT,
        VmcsGuest32::LDTR_ACCESS_RIGHTS,
    )?;

    Ok(())
}

fn merge_guest_control_register(value: u64, mask: u64, shadow: u64) -> u64 {
    (value & !mask) | (shadow & mask)
}

fn read_dtable_from_vmcs(base_field: VmcsGuestNW, limit_field: VmcsGuest32) -> Result<VcpuDtable> {
    Ok(VcpuDtable {
        base: base_field.read().map_err(Error::from)? as u64,
        limit: limit_field.read().map_err(Error::from)? as u16,
        padding: [0; 3],
    })
}

fn read_segment_from_vmcs(
    selector_field: VmcsGuest16,
    base_field: VmcsGuestNW,
    limit_field: VmcsGuest32,
    rights_field: VmcsGuest32,
) -> Result<VcpuSegment> {
    let rights = rights_field.read().map_err(Error::from)?;
    Ok(VcpuSegment {
        base: base_field.read().map_err(Error::from)? as u64,
        limit: limit_field.read().map_err(Error::from)?,
        selector: selector_field.read().map_err(Error::from)?,
        type_: (rights & 0x0f) as u8,
        s: ((rights >> 4) & 0x1) as u8,
        dpl: ((rights >> 5) & 0x3) as u8,
        present: ((rights >> 7) & 0x1) as u8,
        avl: ((rights >> 12) & 0x1) as u8,
        l: ((rights >> 13) & 0x1) as u8,
        db: ((rights >> 14) & 0x1) as u8,
        g: ((rights >> 15) & 0x1) as u8,
        unusable: ((rights >> 16) & 0x1) as u8,
        padding: 0,
    })
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
