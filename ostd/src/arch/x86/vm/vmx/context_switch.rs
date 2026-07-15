// SPDX-License-Identifier: MPL-2.0

use crate::arch::vm::types::VcpuRegs;

/// Enters or resumes the guest and returns after a VM exit.
pub(crate) fn vcpu_run(guest_regs_ptr: *mut VcpuRegs, launched: u64) -> u64 {
    unsafe { __rkvm_vcpu_run(guest_regs_ptr, launched) }
}

/// Returns the assembly VM-exit handler entry address for VMCS host state.
pub(crate) fn vm_exit_handler_virtaddr() -> usize {
    __rkvm_vm_exit_handler as *const () as usize
}

unsafe extern "C" {
    unsafe fn __rkvm_vcpu_run(regs_ptr: *mut VcpuRegs, launched: u64) -> u64;
    unsafe fn __rkvm_vm_exit_handler();
}

core::arch::global_asm!(
    r#"
    .global __rkvm_vcpu_run
    .global __rkvm_vm_exit_handler

    # args: rdi = regs_ptr, rsi = launched (bool/u64)
    __rkvm_vcpu_run:
        # Save Callee-Saved Host Registers (according to System V AMD64 ABI)
        push rbp
        push rbx
        push r12
        push r13
        push r14
        push r15

        # save guest regs pointer
        push rdi

        # Save Host RSP to VMCS
        # so that we can restore host regs after VM Exit
        # VMCS_HOST_RSP = 0x6C14
        mov rdx, 0x6C14
        vmwrite rdx, rsp
        jna .VmwriteFail

        # save launched flag to stack
        push rsi

        # restore guest registers from GuestRegs struct
        mov rax, [rdi + 0x00]
        mov rbx, [rdi + 0x08]
        mov rcx, [rdi + 0x10]
        mov rdx, [rdi + 0x18]
        mov rsi, [rdi + 0x20]
        ## skip rdi
        mov rbp, [rdi + 0x30]
        ## skip rsp
        mov r8,  [rdi + 0x40]
        mov r9,  [rdi + 0x48]
        mov r10, [rdi + 0x50]
        mov r11, [rdi + 0x58]
        mov r12, [rdi + 0x60]
        mov r13, [rdi + 0x68]
        mov r14, [rdi + 0x70]
        mov r15, [rdi + 0x78]
        mov rdi, [rdi + 0x28]  # restore rdi last

        # Check if we should VMLAUNCH or VMRESUME
        cmp qword ptr [rsp], 0
        jne .DoResume

    .DoLaunch:
        vmlaunch
        jmp .LaunchFail

    .DoResume:
        vmresume
        jmp .LaunchFail

    # This is where HOST_RIP should point
    __rkvm_vm_exit_handler:
        # restore guest regs struct pointer
        # after xchg [rsp] = guest rdi value; rdi = host rdi val = guest regs ptr
        xchg rdi, [rsp]

        # Save guest registers to GuestRegs struct
        mov [rdi + 0x00], rax
        mov [rdi + 0x08], rbx
        mov [rdi + 0x10], rcx
        mov [rdi + 0x18], rdx
        mov [rdi + 0x20], rsi
        ## skip rdi
        mov [rdi + 0x30], rbp
        ## skip rsp
        mov [rdi + 0x40], r8
        mov [rdi + 0x48], r9
        mov [rdi + 0x50], r10
        mov [rdi + 0x58], r11
        mov [rdi + 0x60], r12
        mov [rdi + 0x68], r13
        mov [rdi + 0x70], r14
        mov [rdi + 0x78], r15

        pop rax  # get guest rdi value
        mov [rdi + 0x28], rax  # save guest rdi value

        # Restore Host Registers
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp

        # Return 0 (Success/Exit occurred)
        xor rax, rax
        ret

    .LaunchFail:
        # Failure path
        pop rax  # discard launched flag
        pop rax  # discard guest regs pointer
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp

        # Return error code (just 1 for simplicity)
        mov rax, 1
        ret

    .VmwriteFail:
        # Failure path before launched flag is pushed
        pop rax  # discard guest regs pointer
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp

        # Return error code (just 1 for simplicity)
        mov rax, 1
        ret
    "#
);
