use ostd::{
    arch::vm::{
        GuestContext, GuestExitInfo, GuestInterrupt, GuestTimerInstant, VcpuRunState, VmxExitReason,
    },
    mm::{CachePolicy, FrameAllocOptions, PAGE_SIZE, PageFlags, PageProperty, UFrame, VmIo},
    prelude::ktest,
    task::disable_preempt,
    vm::{GuestInterruptPort, GuestMode, GuestPhysMemSpace, GuestRunResult, GuestTimerPort},
};

const IA32_SYSENTER_EIP: u32 = 0x176;
const IA32_PAT: u32 = 0x277;
const OUTPUT_BYTE: u8 = b'A';

// 16-bit real-mode guest:
//     mov al, 'A'
//     mov dx, 0x3f8
//     out dx, al
//     hlt
const GUEST_CODE: &[u8] = &[0xb0, OUTPUT_BYTE, 0xba, 0xf8, 0x03, 0xee, 0xf4];

struct NoGuestEvents;

impl GuestInterruptPort for NoGuestEvents {
    fn query_pending_interrupt(&self) -> Option<GuestInterrupt> {
        None
    }

    fn accept_interrupt(&self, _interrupt: GuestInterrupt) {
        unreachable!("the test guest has no pending interrupts");
    }
}

impl GuestTimerPort for NoGuestEvents {
    fn poll_deadline(&self, _current: GuestTimerInstant) -> Option<GuestTimerInstant> {
        None
    }
}

fn run_until_vmexit(
    guest_mode: &GuestMode,
    context: &mut GuestContext,
    guest_mem: &GuestPhysMemSpace,
) -> GuestExitInfo {
    let no_events = NoGuestEvents;
    loop {
        match guest_mode
            .execute(context, guest_mem, &no_events, &no_events)
            .expect("guest entry or exit handling failed")
        {
            GuestRunResult::VmExit(exit) => return exit,
            GuestRunResult::HostInterrupt => continue,
            GuestRunResult::WaitForSipi => {
                panic!("the bootstrap vCPU unexpectedly waited for SIPI")
            }
        }
    }
}

#[ktest]
fn hypervisor_runs_outb_then_hlt() {
    let guest_mode =
        GuestMode::new().expect("nested VMX is required for the hypervisor execution test");
    let guest_mem =
        GuestPhysMemSpace::new().expect("EPT is required for the hypervisor execution test");

    let guest_frame = FrameAllocOptions::new()
        .alloc_frame()
        .expect("failed to allocate guest RAM");
    guest_frame
        .write_bytes(0, GUEST_CODE)
        .expect("failed to load guest code");
    let guest_frame: UFrame = guest_frame.into();

    {
        let preempt_guard = disable_preempt();
        let mut cursor = guest_mem
            .cursor_mut(&preempt_guard, &(0..PAGE_SIZE))
            .expect("failed to edit the guest physical address space");
        cursor.map(
            guest_frame,
            PageProperty::new_user(PageFlags::RWX, CachePolicy::Writeback),
        );
    }

    let mut context = GuestContext::new(0).expect("failed to create the bootstrap vCPU");
    let mut regs = context.regs();
    regs.rip = 0;
    context.set_regs(regs);
    let mut sregs = context.sregs();
    sregs.cs.selector = 0;
    sregs.cs.base = 0;
    context.set_sregs(sregs);
    assert_eq!(context.rip(), 0);

    let io_exit = run_until_vmexit(&guest_mode, &mut context, &guest_mem);
    assert_eq!(io_exit.exit_reason, VmxExitReason::IO_INSTRUCTION as u32);
    assert_eq!(io_exit.guest_rip, 5);
    assert_eq!((io_exit.exit_qualification >> 16) & 0xffff, 0x3f8_u64);
    assert_eq!(context.regs().rax & 0xff, usize::from(b'A'));

    // Emulate the OUT by consuming it, then resume at HLT.
    context.advance_rip(u64::from(io_exit.instruction_len));
    let hlt_exit = run_until_vmexit(&guest_mode, &mut context, &guest_mem);
    assert_eq!(hlt_exit.exit_reason, VmxExitReason::HLT as u32);
    assert_eq!(hlt_exit.guest_rip, 6);
    assert_eq!(hlt_exit.instruction_len, 1);
    assert_eq!(context.run_state(), VcpuRunState::Halted);
}

#[ktest]
fn init_and_sipi_follow_x86_state_transitions() {
    const PROCESSOR_SIGNATURE: u32 = 0x0006_06a1;
    const TEST_PAT: u64 = 0x0001_0203_0405_0607;
    const TEST_SYSENTER_EIP: u64 = 0x1234_5678;

    let mut ap = GuestContext::new(1).expect("failed to create the application vCPU");
    assert_eq!(ap.run_state(), VcpuRunState::Uninitialized);
    assert_eq!(ap.rip(), 0xfff0);
    assert_eq!(ap.sregs().cs.selector, 0xf000);
    assert_eq!(ap.sregs().cs.base, 0xffff_0000);
    assert_eq!(ap.sregs().cr0, 0x6000_0010);

    let apic_base = ap.sregs().apic_base;
    let mut sregs = ap.sregs();
    sregs.cr0 = (sregs.cr0 | (1 << 31)) & !(1 << 29);
    sregs.cr3 = 0x4000;
    sregs.cr4 = 0x20;
    sregs.efer = 0x500;
    ap.set_sregs(sregs);
    assert!(ap.write_msr(IA32_PAT, TEST_PAT));
    assert!(ap.write_msr(IA32_SYSENTER_EIP, TEST_SYSENTER_EIP));

    assert!(ap.receive_init(PROCESSOR_SIGNATURE));
    assert_eq!(ap.run_state(), VcpuRunState::WaitForSipi);
    assert_eq!(ap.rip(), 0xfff0);
    assert_eq!(ap.regs().rdx, PROCESSOR_SIGNATURE as usize);
    assert_eq!(ap.sregs().cs.selector, 0xf000);
    assert_eq!(ap.sregs().cs.base, 0xffff_0000);
    assert_eq!(ap.sregs().cr0, 0x4000_0010);
    assert_eq!(ap.sregs().cr3, 0);
    assert_eq!(ap.sregs().cr4, 0);
    assert_eq!(ap.sregs().efer, 0);
    assert_eq!(ap.sregs().apic_base, apic_base);
    assert_eq!(ap.read_msr(IA32_PAT), Some(TEST_PAT));
    assert_eq!(ap.read_msr(IA32_SYSENTER_EIP), Some(TEST_SYSENTER_EIP));

    let mut regs = ap.regs();
    regs.rax = 0xfeed_face;
    ap.set_regs(regs);
    let mut sregs = ap.sregs();
    sregs.ds.selector = 0x20;
    sregs.ds.base = 0x200;
    ap.set_sregs(sregs);

    ap.receive_sipi(0x08);
    assert_eq!(ap.run_state(), VcpuRunState::Runnable);
    assert_eq!(ap.rip(), 0);
    assert_eq!(ap.sregs().cs.selector, 0x0800);
    assert_eq!(ap.sregs().cs.base, 0x8000);
    assert_eq!(ap.regs().rax, 0xfeed_face);
    assert_eq!(ap.regs().rdx, PROCESSOR_SIGNATURE as usize);
    assert_eq!(ap.sregs().ds.selector, 0x20);
    assert_eq!(ap.sregs().ds.base, 0x200);
    assert_eq!(ap.read_msr(IA32_PAT), Some(TEST_PAT));
    assert_eq!(ap.read_msr(IA32_SYSENTER_EIP), Some(TEST_SYSENTER_EIP));

    ap.receive_sipi(0x09);
    assert_eq!(ap.sregs().cs.selector, 0x0800);
    assert_eq!(ap.sregs().cs.base, 0x8000);

    let mut bsp = GuestContext::new(0).expect("failed to create the bootstrap vCPU");
    assert!(bsp.receive_init(PROCESSOR_SIGNATURE));
    assert_eq!(bsp.run_state(), VcpuRunState::Runnable);
    assert_eq!(bsp.rip(), 0xfff0);
}
