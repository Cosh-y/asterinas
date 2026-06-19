use ostd::{
    arch::vm::{
        GuestContext, VcpuDtable as ArchVcpuDtable, VcpuRegs as ArchVcpuRegs,
        VcpuSegment as ArchVcpuSegment, VcpuSregs as ArchVcpuSregs,
    },
    mm::VmIo,
    vm::GuestInterruptPort,
};

use super::{
    apic::Lapic,
    ioctl::{
        KVM_RUN_EXIT_REASON_OFFSET, KVM_RUN_MMAP_SIZE, KVM_RUN_STRUCT_SIZE, VcpuDtable, VcpuRegs,
        VcpuSegment, VcpuSregs,
    },
    vm::Vm,
};
use crate::{
    prelude::*,
    vm::vmo::{Vmo, VmoOptions},
};

const HLT_WAKEUP_WAIT_TSC_DIVISOR: u64 = 10_000;
const HLT_WAKEUP_WAIT_FALLBACK_TICKS: u64 = 250_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PioDirection {
    In,
    Out,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PendingPioOperation {
    pub direction: PioDirection,
    pub size: u8,
    pub instruction_len: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PendingMmioOperation {
    pub is_write: bool,
    pub instruction_len: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PendingOperation {
    Pio(PendingPioOperation),
    Mmio(PendingMmioOperation),
}

pub struct Vcpu {
    pub(super) id: u32,
    pub(super) vm: Weak<Vm>,
    pub(super) guest_context: Mutex<GuestContext>,
    pub(super) lapic: SpinLock<Lapic>,
    run_page: Arc<Vmo>,
    pending_operation: Mutex<Option<PendingOperation>>,
}

impl Vcpu {
    pub(super) fn new(id: u32, vm: &Arc<Vm>, lapic: Lapic) -> Result<Arc<Self>> {
        let run_page = VmoOptions::new(KVM_RUN_MMAP_SIZE).alloc()?;
        Ok(Arc::new(Self {
            id,
            vm: Arc::downgrade(vm),
            guest_context: Mutex::new(GuestContext::new(id)?),
            lapic: SpinLock::new(lapic),
            run_page,
            pending_operation: Mutex::new(None),
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

    pub(super) fn run_page(&self) -> Arc<Vmo> {
        self.run_page.clone()
    }

    pub(super) fn read_run_val<T: Pod>(&self, offset: usize) -> Result<T> {
        self.run_page.read_val(offset).map_err(Error::from)
    }

    pub(super) fn write_run_val<T: Pod>(&self, offset: usize, value: &T) -> Result<()> {
        self.run_page.write_val(offset, value).map_err(Error::from)
    }

    pub(super) fn read_run_bytes(&self, offset: usize, buffer: &mut [u8]) -> Result<()> {
        self.run_page
            .read_bytes(offset, buffer)
            .map_err(Error::from)
    }

    pub(super) fn write_run_bytes(&self, offset: usize, buffer: &[u8]) -> Result<()> {
        self.run_page
            .write_bytes(offset, buffer)
            .map_err(Error::from)
    }

    pub(super) fn clear_run_output(&self) -> Result<()> {
        self.run_page
            .clear(KVM_RUN_EXIT_REASON_OFFSET..KVM_RUN_STRUCT_SIZE)
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
        Ok(kvm_regs_from_arch(context.regs()))
    }

    pub fn set_regs(&self, regs: VcpuRegs) -> Result<()> {
        let mut context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot set regs while vCPU is running");
        }
        context.set_regs(arch_regs_from_kvm(regs));
        Ok(())
    }

    pub fn get_sregs(&self) -> Result<VcpuSregs> {
        let context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot get sregs while vCPU is running");
        }
        Ok(kvm_sregs_from_arch(context.sregs()))
    }

    pub fn set_sregs(&self, sregs: VcpuSregs) -> Result<()> {
        let mut context = self.guest_context.lock();
        if context.is_running() {
            return_errno_with_message!(Errno::EBUSY, "cannot set sregs while vCPU is running");
        }
        context.set_sregs(arch_sregs_from_kvm(sregs));
        Ok(())
    }

    pub fn receive_sipi(&self, vector: u8) {
        self.guest_context.lock().receive_sipi(vector);
    }

    pub(super) fn wait_for_hlt_wakeup(&self) -> bool {
        use ostd::arch::{read_tsc, tsc_freq};
        let wait_max_ticks = match tsc_freq() {
            0 => HLT_WAKEUP_WAIT_FALLBACK_TICKS,
            freq => (freq / HLT_WAKEUP_WAIT_TSC_DIVISOR).max(1),
        };
        let start_tsc = read_tsc();
        loop {
            if let Some(tsc_deadline) = self.lapic.lock().timer.deadline_tsc
                && self.guest_context().guest_tsc() >= tsc_deadline
            {
                return true;
            }

            // TODO: decide timer expire in deadline mode

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

fn kvm_regs_from_arch(regs: ArchVcpuRegs) -> VcpuRegs {
    VcpuRegs {
        rax: regs.rax,
        rbx: regs.rbx,
        rcx: regs.rcx,
        rdx: regs.rdx,
        rsi: regs.rsi,
        rdi: regs.rdi,
        rsp: regs.rsp,
        rbp: regs.rbp,
        r8: regs.r8,
        r9: regs.r9,
        r10: regs.r10,
        r11: regs.r11,
        r12: regs.r12,
        r13: regs.r13,
        r14: regs.r14,
        r15: regs.r15,
        rip: regs.rip,
        rflags: regs.rflags,
    }
}

fn arch_regs_from_kvm(regs: VcpuRegs) -> ArchVcpuRegs {
    ArchVcpuRegs {
        rax: regs.rax,
        rbx: regs.rbx,
        rcx: regs.rcx,
        rdx: regs.rdx,
        rsi: regs.rsi,
        rdi: regs.rdi,
        rbp: regs.rbp,
        rsp: regs.rsp,
        r8: regs.r8,
        r9: regs.r9,
        r10: regs.r10,
        r11: regs.r11,
        r12: regs.r12,
        r13: regs.r13,
        r14: regs.r14,
        r15: regs.r15,
        rip: regs.rip,
        rflags: regs.rflags,
    }
}

fn kvm_sregs_from_arch(sregs: ArchVcpuSregs) -> VcpuSregs {
    VcpuSregs {
        cs: kvm_segment_from_arch(sregs.cs),
        ds: kvm_segment_from_arch(sregs.ds),
        es: kvm_segment_from_arch(sregs.es),
        fs: kvm_segment_from_arch(sregs.fs),
        gs: kvm_segment_from_arch(sregs.gs),
        ss: kvm_segment_from_arch(sregs.ss),
        tr: kvm_segment_from_arch(sregs.tr),
        ldt: kvm_segment_from_arch(sregs.ldt),
        gdt: kvm_dtable_from_arch(sregs.gdt),
        idt: kvm_dtable_from_arch(sregs.idt),
        cr0: sregs.cr0,
        cr2: sregs.cr2,
        cr3: sregs.cr3,
        cr4: sregs.cr4,
        cr8: 0,
        efer: sregs.efer,
        apic_base: sregs.apic_base,
        interrupt_bitmap: sregs.interrupt_bitmap,
    }
}

fn arch_sregs_from_kvm(sregs: VcpuSregs) -> ArchVcpuSregs {
    ArchVcpuSregs {
        cs: arch_segment_from_kvm(sregs.cs),
        ds: arch_segment_from_kvm(sregs.ds),
        es: arch_segment_from_kvm(sregs.es),
        fs: arch_segment_from_kvm(sregs.fs),
        gs: arch_segment_from_kvm(sregs.gs),
        ss: arch_segment_from_kvm(sregs.ss),
        tr: arch_segment_from_kvm(sregs.tr),
        ldt: arch_segment_from_kvm(sregs.ldt),
        gdt: arch_dtable_from_kvm(sregs.gdt),
        idt: arch_dtable_from_kvm(sregs.idt),
        cr0: sregs.cr0,
        cr2: sregs.cr2,
        cr3: sregs.cr3,
        cr4: sregs.cr4,
        efer: sregs.efer,
        apic_base: sregs.apic_base,
        interrupt_bitmap: sregs.interrupt_bitmap,
    }
}

fn kvm_segment_from_arch(segment: ArchVcpuSegment) -> VcpuSegment {
    VcpuSegment {
        base: segment.base,
        limit: segment.limit,
        selector: segment.selector,
        type_: segment.type_,
        present: segment.present,
        dpl: segment.dpl,
        db: segment.db,
        s: segment.s,
        l: segment.l,
        g: segment.g,
        avl: segment.avl,
        unusable: segment.unusable,
        padding: segment.padding,
    }
}

fn arch_segment_from_kvm(segment: VcpuSegment) -> ArchVcpuSegment {
    ArchVcpuSegment {
        base: segment.base,
        limit: segment.limit,
        selector: segment.selector,
        type_: segment.type_,
        present: segment.present,
        dpl: segment.dpl,
        db: segment.db,
        s: segment.s,
        l: segment.l,
        g: segment.g,
        avl: segment.avl,
        unusable: segment.unusable,
        padding: segment.padding,
    }
}

fn kvm_dtable_from_arch(dtable: ArchVcpuDtable) -> VcpuDtable {
    VcpuDtable {
        base: dtable.base,
        limit: dtable.limit,
        padding: dtable.padding,
    }
}

fn arch_dtable_from_kvm(dtable: VcpuDtable) -> ArchVcpuDtable {
    ArchVcpuDtable {
        base: dtable.base,
        limit: dtable.limit,
        padding: dtable.padding,
    }
}
