use x86::{
    dtables::{self, DescriptorTablePointer},
    segmentation, task,
    vmx::vmcs::control::{
        EntryControls, ExitControls, PinbasedControls, PrimaryControls, SecondaryControls,
    },
};
use x86_64::registers::{
    control::{Cr0, Cr3, Cr4},
    model_specific::EferFlags,
};

use super::{
    control_regs::VcpuControlRegisters,
    types::{VcpuMsrs, VcpuRegs, VcpuSegment, VcpuSregs},
    vmx::*,
    x86::get_tr_base,
};
use crate::{
    Error,
    mm::{Frame, FrameAllocOptions, PAGE_SIZE, VmIo},
    prelude::*,
};

pub(crate) struct Vmcs {
    /// IO bitmap A for trapping lower port range accesses.
    io_bitmap_a: Frame<()>,
    /// IO bitmap B for trapping upper port range accesses.
    io_bitmap_b: Frame<()>,
    /// MSR bitmap for trapping RDMSR/WRMSR accesses.
    msr_bitmap: Frame<()>,
    /// VM-entry/exit MSR-load/store areas.
    ///
    /// The first list stores guest values and is shared by VM-entry load and
    /// VM-exit store. The second list contains the host values restored by
    /// VM-exit. Both lists remain private to OSTD.
    run_msr_area: Frame<()>,
    /// Tracks the VMCS region state and CPU residency.
    tracking: Arc<VmcsTracking>,
}

const RUN_MSR_INDICES: [u32; 5] = [
    Msr::IA32_STAR as u32,
    Msr::IA32_LSTAR as u32,
    Msr::IA32_CSTAR as u32,
    Msr::IA32_FMASK as u32,
    Msr::IA32_KERNEL_GSBASE as u32,
];
const RUN_MSR_COUNT: usize = RUN_MSR_INDICES.len();
const GUEST_RUN_MSR_AREA_OFFSET: usize = 0;
const HOST_RUN_MSR_AREA_OFFSET: usize = RUN_MSR_COUNT * size_of::<VmxMsrListEntry>();

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod)]
struct VmxMsrListEntry {
    index: u32,
    reserved: u32,
    value: u64,
}

pub(crate) struct VmcsGuestState {
    pub(crate) regs: VcpuRegs,
    pub(crate) sregs: VcpuSregs,
    pub(crate) control_regs: VcpuControlRegisters,
    pub(crate) msrs: VcpuMsrs,
}

impl VmcsGuestState {
    // fn regs(&self) -> VcpuRegs {
    //     self.regs
    // }

    // fn sregs(&self) -> VcpuSregs {
    //     self.sregs
    // }

    // fn msrs(&self) -> VcpuMsrs {
    //     self.msrs
    // }
}

impl Vmcs {
    pub fn new() -> Result<Self> {
        // Allocate VMCS
        let vmcs_region = alloc_vmcs()?;

        let io_bitmap_a = FrameAllocOptions::new().alloc_frame()?;
        let io_bitmap_b = FrameAllocOptions::new().alloc_frame()?;
        let msr_bitmap = FrameAllocOptions::new().alloc_frame()?;
        let run_msr_area = FrameAllocOptions::new().alloc_frame()?;
        let all_ones = [0xff_u8; PAGE_SIZE];
        let all_zeros = [0_u8; PAGE_SIZE];
        io_bitmap_a.write_bytes(0, &all_ones)?;
        io_bitmap_b.write_bytes(0, &all_ones)?;
        msr_bitmap.write_bytes(0, &all_ones)?;
        configure_guest_owned_msrs(&msr_bitmap)?;
        run_msr_area.write_bytes(0, &all_zeros)?;

        Ok(Self {
            io_bitmap_a,
            io_bitmap_b,
            msr_bitmap,
            run_msr_area,
            tracking: Arc::new(VmcsTracking::new(vmcs_region)),
        })
    }

    fn vmcs_phys(&self) -> Paddr {
        self.tracking.paddr()
    }

    pub fn load(&mut self, vmcs_guest_state: VmcsGuestState, eptp: u64) -> Result<()> {
        let cpu_changed = activate_vmcs(self.vmcs_phys(), &self.tracking)?;

        if !self.tracking.initialized() {
            self.setup_vmcs(vmcs_guest_state, eptp)?;
            self.tracking.set_initialized(true);
            self.tracking.set_launched(false);
        } else if cpu_changed {
            if let Err(err) = self.setup_vmcs_host() {
                if let Err(clear_err) = deactivate_vmcs(self.vmcs_phys(), &self.tracking) {
                    warn!(
                        "hypervisor: failed to clear VMCS after host-state setup failure: {:?}",
                        clear_err
                    );
                }
                return Err(err);
            }
        }

        // Unlike most VMCS host-state fields, FS.base belongs to the current
        // userspace thread rather than to the physical CPU.  A VMCS can be
        // reused by another thread without changing CPUs, so refresh it for
        // every run.  Otherwise VM exit may restore a stale TLS base from the
        // previous runner.
        self.refresh_vmcs_host_task_state()?;
        Ok(())
    }

    pub fn launched(&self) -> bool {
        self.tracking.launched()
    }

    pub(crate) fn initialized(&self) -> bool {
        self.tracking.initialized()
    }

    pub(crate) fn activate_for_access(&self) -> Result<()> {
        let cpu_changed = activate_vmcs(self.vmcs_phys(), &self.tracking)?;
        if cpu_changed {
            self.setup_vmcs_host()?;
        }
        Ok(())
    }

    pub(crate) fn write_guest_run_msrs(&self, values: [u64; RUN_MSR_COUNT]) -> Result<()> {
        self.write_run_msr_list(GUEST_RUN_MSR_AREA_OFFSET, values)
    }

    pub(crate) fn read_guest_run_msrs(&self) -> Result<[u64; RUN_MSR_COUNT]> {
        self.read_run_msr_list(GUEST_RUN_MSR_AREA_OFFSET)
    }

    pub(crate) fn write_host_run_msrs(&self, values: [u64; RUN_MSR_COUNT]) -> Result<()> {
        self.write_run_msr_list(HOST_RUN_MSR_AREA_OFFSET, values)
    }

    fn write_run_msr_list(&self, base_offset: usize, values: [u64; RUN_MSR_COUNT]) -> Result<()> {
        for (position, (&index, value)) in RUN_MSR_INDICES.iter().zip(values).enumerate() {
            let entry = VmxMsrListEntry {
                index,
                reserved: 0,
                value,
            };
            self.run_msr_area.write_val(
                base_offset + position * size_of::<VmxMsrListEntry>(),
                &entry,
            )?;
        }
        Ok(())
    }

    fn read_run_msr_list(&self, base_offset: usize) -> Result<[u64; RUN_MSR_COUNT]> {
        let mut values = [0_u64; RUN_MSR_COUNT];
        for (position, (&expected_index, value)) in
            RUN_MSR_INDICES.iter().zip(values.iter_mut()).enumerate()
        {
            let entry: VmxMsrListEntry = self
                .run_msr_area
                .read_val(base_offset + position * size_of::<VmxMsrListEntry>())?;
            if entry.index != expected_index || entry.reserved != 0 {
                return Err(Error::InvalidArgs);
            }
            *value = entry.value;
        }
        Ok(values)
    }

    pub fn set_launched(&mut self, value: bool) {
        self.tracking.set_launched(value)
    }

    /// Setup VMCS with initial guest state
    fn setup_vmcs(&self, vmcs_guest_state: VmcsGuestState, eptp: u64) -> Result<()> {
        self.setup_vmcs_host()?;
        self.setup_vmcs_guest(&vmcs_guest_state)?;
        self.setup_vmcs_controls(&vmcs_guest_state, eptp)?;
        Ok(())
    }

    fn setup_vmcs_host(&self) -> Result<()> {
        VmcsHost64::IA32_PAT.write(Msr::IA32_PAT.read())?;
        VmcsHost64::IA32_EFER.write(Msr::IA32_EFER.read())?;

        VmcsHostNW::CR0.write(Cr0::read_raw() as _)?;
        VmcsHostNW::CR3.write(Cr3::read_raw().0.start_address().as_u64() as _)?; // TODO: check difference with JiaYuekai
        VmcsHostNW::CR4.write(Cr4::read_raw() as _)?;

        VmcsHost16::ES_SELECTOR.write(segmentation::es().bits())?;
        VmcsHost16::CS_SELECTOR.write(segmentation::cs().bits())?;
        VmcsHost16::SS_SELECTOR.write(segmentation::ss().bits())?;
        VmcsHost16::DS_SELECTOR.write(segmentation::ds().bits())?;
        VmcsHost16::FS_SELECTOR.write(segmentation::fs().bits())?;
        VmcsHost16::GS_SELECTOR.write(segmentation::gs().bits())?;
        VmcsHostNW::FS_BASE.write(Msr::IA32_FS_BASE.read() as _)?;
        VmcsHostNW::GS_BASE.write(Msr::IA32_GS_BASE.read() as _)?;

        // SAFETY: STR only reads the current task-register selector.
        let tr = unsafe { task::tr() };
        let mut gdtp = DescriptorTablePointer::<u64>::default();
        let mut idtp = DescriptorTablePointer::<u64>::default();
        // SAFETY: SGDT/SIDT only read descriptor-table registers into local memory.
        unsafe {
            dtables::sgdt(&mut gdtp);
            dtables::sidt(&mut idtp);
        }

        VmcsHost16::TR_SELECTOR.write(tr.bits())?;
        VmcsHostNW::TR_BASE.write(get_tr_base(tr, &gdtp) as _)?;
        VmcsHostNW::GDTR_BASE.write(gdtp.base as usize)?;
        VmcsHostNW::IDTR_BASE.write(idtp.base as usize)?;
        VmcsHostNW::RIP.write(vm_exit_handler_virtaddr() as _)?;

        VmcsHostNW::IA32_SYSENTER_ESP.write(Msr::IA32_SYSENTER_ESP.read() as usize)?;
        VmcsHostNW::IA32_SYSENTER_EIP.write(Msr::IA32_SYSENTER_EIP.read() as usize)?;
        VmcsHost32::IA32_SYSENTER_CS.write(Msr::IA32_SYSENTER_CS.read() as u32)?;
        Ok(())
    }

    fn refresh_vmcs_host_task_state(&self) -> Result<()> {
        VmcsHostNW::FS_BASE.write(Msr::IA32_FS_BASE.read() as _)
    }

    fn setup_vmcs_guest(&self, vmcs_guest_state: &VmcsGuestState) -> Result<()> {
        let regs = vmcs_guest_state.regs;
        let sregs = vmcs_guest_state.sregs;
        let control_regs = vmcs_guest_state.control_regs;
        let msrs = vmcs_guest_state.msrs;

        let cr0 = control_regs.cr0();
        VmcsGuestNW::CR0.write(cr0.real() as _)?;
        VmcsControlNW::CR0_GUEST_HOST_MASK.write(cr0.host_mask() as _)?;
        VmcsControlNW::CR0_READ_SHADOW.write(cr0.read_shadow() as _)?;

        let cr4 = control_regs.cr4();
        VmcsGuestNW::CR4.write(cr4.real() as _)?;
        VmcsControlNW::CR4_GUEST_HOST_MASK.write(cr4.host_mask() as _)?;
        VmcsControlNW::CR4_READ_SHADOW.write(cr4.read_shadow() as _)?;

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
        VmcsGuest64::IA32_EFER.write(msrs.efer)?;
        VmcsControl64::TSC_OFFSET.write(0)?;
        Ok(())
    }

    fn setup_vmcs_controls(&self, vmcs_guest_state: &VmcsGuestState, eptp: u64) -> Result<()> {
        set_control(
            VmcsControl32::PINBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PINBASED_CTLS,
            Msr::IA32_VMX_PINBASED_CTLS.read() as u32,
            (PinbasedControls::EXTERNAL_INTERRUPT_EXITING
                | PinbasedControls::NMI_EXITING
                | PinbasedControls::VMX_PREEMPTION_TIMER)
                .bits(),
            0,
        )?;

        let secondary_cap = Msr::IA32_VMX_PROCBASED_CTLS2.read();
        let secondary_allowed1 = (secondary_cap >> 32) as u32;
        let supports_pause_loop_exiting =
            (secondary_allowed1 & SecondaryControls::PAUSE_LOOP_EXITING.bits()) != 0;
        let pause_exiting_fallback = if supports_pause_loop_exiting {
            0
        } else {
            PrimaryControls::PAUSE_EXITING.bits()
        };

        set_control(
            VmcsControl32::PRIMARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_TRUE_PROCBASED_CTLS,
            Msr::IA32_VMX_PROCBASED_CTLS.read() as u32,
            (PrimaryControls::USE_TSC_OFFSETTING
                | PrimaryControls::HLT_EXITING
                // | PrimaryControls::RDTSC_EXITING
                | PrimaryControls::USE_IO_BITMAPS
                | PrimaryControls::USE_MSR_BITMAPS
                | PrimaryControls::SECONDARY_CONTROLS)
                .bits()
                | pause_exiting_fallback,
            (PrimaryControls::CR3_LOAD_EXITING | PrimaryControls::CR3_STORE_EXITING).bits(),
        )?;

        let pause_loop_exiting = if supports_pause_loop_exiting {
            SecondaryControls::PAUSE_LOOP_EXITING
        } else {
            SecondaryControls::empty()
        };
        set_control(
            VmcsControl32::SECONDARY_PROCBASED_EXEC_CONTROLS,
            Msr::IA32_VMX_PROCBASED_CTLS2,
            0,
            (SecondaryControls::ENABLE_EPT
                | SecondaryControls::ENABLE_RDTSCP
                | SecondaryControls::UNRESTRICTED_GUEST
                | pause_loop_exiting)
                .bits(),
            0,
        )?;
        if supports_pause_loop_exiting {
            const VMX_PAUSE_LOOP_EXIT_GAP: u32 = 1_000_000;
            const VMX_PAUSE_LOOP_EXIT_WINDOW: u32 = 4096;
            VmcsControl32::PLE_GAP.write(VMX_PAUSE_LOOP_EXIT_GAP)?;
            VmcsControl32::PLE_WINDOW.write(VMX_PAUSE_LOOP_EXIT_WINDOW)?;
        }

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
        let msrs = vmcs_guest_state.msrs;
        if msrs.efer & EferFlags::LONG_MODE_ACTIVE.bits() != 0 {
            entry_controls |= EntryControls::IA32E_MODE_GUEST.bits();
        }

        set_control(
            VmcsControl32::VMENTRY_CONTROLS,
            Msr::IA32_VMX_TRUE_ENTRY_CTLS,
            Msr::IA32_VMX_ENTRY_CTLS.read() as u32,
            entry_controls,
            0,
        )?;

        let guest_run_msr_addr =
            self.run_msr_area.paddr() as u64 + GUEST_RUN_MSR_AREA_OFFSET as u64;
        let host_run_msr_addr = self.run_msr_area.paddr() as u64 + HOST_RUN_MSR_AREA_OFFSET as u64;
        VmcsControl64::VMEXIT_MSR_STORE_ADDR.write(guest_run_msr_addr)?;
        VmcsControl64::VMEXIT_MSR_LOAD_ADDR.write(host_run_msr_addr)?;
        VmcsControl64::VMENTRY_MSR_LOAD_ADDR.write(guest_run_msr_addr)?;
        VmcsControl32::VMEXIT_MSR_STORE_COUNT.write(RUN_MSR_COUNT as u32)?;
        VmcsControl32::VMEXIT_MSR_LOAD_COUNT.write(RUN_MSR_COUNT as u32)?;
        VmcsControl32::VMENTRY_MSR_LOAD_COUNT.write(RUN_MSR_COUNT as u32)?;

        // Pass-through exceptions. Intercept I/O and MSR accesses via bitmaps.
        VmcsControl32::EXCEPTION_BITMAP.write(0)?;
        VmcsControl64::IO_BITMAP_A_ADDR.write(self.io_bitmap_a.paddr() as u64)?;
        VmcsControl64::IO_BITMAP_B_ADDR.write(self.io_bitmap_b.paddr() as u64)?;
        VmcsControl64::MSR_BITMAPS_ADDR.write(self.msr_bitmap.paddr() as u64)?;

        // setup EPT
        VmcsControl64::EPTP.write(eptp)?;
        Ok(())
    }
}

fn configure_guest_owned_msrs(msr_bitmap: &Frame<()>) -> Result<()> {
    const IA32_TSC: u32 = 0x10;
    const IA32_SYSENTER_CS: u32 = 0x174;
    const IA32_SYSENTER_ESP: u32 = 0x175;
    const IA32_SYSENTER_EIP: u32 = 0x176;

    // TSC writes remain trapped so the kernel can update TSC offset and the
    // paravirtual clock together. VMX applies TSC offsetting to direct reads.
    allow_msr_access(msr_bitmap, IA32_TSC, MsrBitmapAccess::Read)?;

    // FS/GS and SYSENTER are architecturally represented in VMCS guest state;
    // KERNEL_GS_BASE is owned by the VM-entry/exit MSR lists above.
    for msr in [
        Msr::IA32_FS_BASE as u32,
        Msr::IA32_GS_BASE as u32,
        Msr::IA32_KERNEL_GSBASE as u32,
        IA32_SYSENTER_CS,
        IA32_SYSENTER_ESP,
        IA32_SYSENTER_EIP,
    ] {
        allow_msr_access(msr_bitmap, msr, MsrBitmapAccess::Read)?;
        allow_msr_access(msr_bitmap, msr, MsrBitmapAccess::Write)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum MsrBitmapAccess {
    Read,
    Write,
}

fn allow_msr_access(msr_bitmap: &Frame<()>, msr: u32, access: MsrBitmapAccess) -> Result<()> {
    let Some((byte_offset, bit_mask)) = msr_bitmap_location(msr, access) else {
        return Err(Error::InvalidArgs);
    };
    let mut byte: u8 = msr_bitmap.read_val(byte_offset)?;
    byte &= !bit_mask;
    msr_bitmap.write_val(byte_offset, &byte)
}

fn msr_bitmap_location(msr: u32, access: MsrBitmapAccess) -> Option<(usize, u8)> {
    let access_offset = match access {
        MsrBitmapAccess::Read => 0,
        MsrBitmapAccess::Write => 0x800,
    };
    let (range_offset, bit) = if msr <= 0x1fff {
        (0, msr as usize)
    } else if (0xc000_0000..=0xc000_1fff).contains(&msr) {
        (0x400, (msr & 0x1fff) as usize)
    } else {
        return None;
    };
    Some((access_offset + range_offset + bit / 8, 1 << (bit % 8)))
}

#[cfg(ktest)]
mod tests {
    use super::{MsrBitmapAccess, msr_bitmap_location};
    use crate::prelude::*;

    #[ktest]
    fn msr_bitmap_locates_low_and_high_ranges() {
        assert_eq!(
            msr_bitmap_location(0x10, MsrBitmapAccess::Read),
            Some((0x2, 0x1))
        );
        assert_eq!(
            msr_bitmap_location(0x10, MsrBitmapAccess::Write),
            Some((0x802, 0x1))
        );
        assert_eq!(
            msr_bitmap_location(0xc000_0100, MsrBitmapAccess::Read),
            Some((0x420, 0x1))
        );
        assert_eq!(
            msr_bitmap_location(0xc000_0100, MsrBitmapAccess::Write),
            Some((0xc20, 0x1))
        );
    }

    #[ktest]
    fn msr_bitmap_rejects_uncovered_ranges() {
        assert_eq!(
            msr_bitmap_location(0x4000_0000, MsrBitmapAccess::Read),
            None
        );
    }
}

impl Drop for Vmcs {
    fn drop(&mut self) {
        if let Err(err) = deactivate_vmcs(self.vmcs_phys(), &self.tracking) {
            warn!("hypervisor: failed to clear VMCS during drop: {:?}", err);
        }
    }
}

// TODO: clear up the following code.
pub(super) fn segment_access_rights(segment: &VcpuSegment) -> u32 {
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
