// SPDX-License-Identifier: MPL-2.0

//! Decodes and completes guest port-I/O instructions.

use ostd::arch::vm::{GuestContext, GuestExitInfo, VcpuSegment, X86GprIndex};

use super::{guest_address::translate_gva_to_gpa, vm_memory::VmMemory};
use crate::prelude::*;

const MAX_INSN_LENGTH: usize = 15;
const RFLAGS_DIRECTION: u64 = 1 << 10;
const EFER_LONG_MODE_ACTIVE: u64 = 1 << 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PioDirection {
    In,
    Out,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PioOperation {
    direction: PioDirection,
    size: u8,
    port: u16,
    instruction_len: u32,
    string: Option<StringPioOperation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StringPioOperation {
    address_size: u8,
    segment: SegmentRegister,
    repeated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SegmentRegister {
    Cs,
    Ss,
    Ds,
    Es,
    Fs,
    Gs,
}

impl PioOperation {
    pub(super) fn decode(
        context: &GuestContext,
        vm_memory: &VmMemory,
        exit_info: &GuestExitInfo,
    ) -> Result<Option<Self>> {
        let qualification = exit_info.exit_qualification;
        let size = ((qualification & 0b111) as u8).saturating_add(1);
        if !matches!(size, 1 | 2 | 4) {
            return Ok(None);
        }

        let direction = if qualification & (1 << 3) != 0 {
            PioDirection::In
        } else {
            PioDirection::Out
        };
        let is_string = qualification & (1 << 4) != 0;
        let repeated = qualification & (1 << 5) != 0;
        let string = if is_string {
            let Some(string) = decode_string_instruction(
                context,
                vm_memory,
                direction,
                size,
                repeated,
                exit_info.instruction_len,
            )?
            else {
                return Ok(None);
            };
            Some(string)
        } else {
            if repeated {
                return Ok(None);
            }
            None
        };

        Ok(Some(Self {
            direction,
            size,
            port: ((qualification >> 16) & 0xffff) as u16,
            instruction_len: exit_info.instruction_len,
            string,
        }))
    }

    pub(super) fn direction(self) -> PioDirection {
        self.direction
    }

    pub(super) fn size(self) -> u8 {
        self.size
    }

    pub(super) fn port(self) -> u16 {
        self.port
    }

    pub(super) fn batch_count(self, context: &GuestContext, data_capacity: usize) -> u32 {
        let max_count = data_capacity / usize::from(self.size);
        let remaining = self.remaining_count(context);
        u32::try_from(remaining.min(u64::try_from(max_count).unwrap_or(u64::MAX)))
            .unwrap_or(u32::MAX)
    }

    pub(super) fn output_data(
        self,
        context: &GuestContext,
        vm_memory: &VmMemory,
        count: u32,
    ) -> Result<Vec<u8>> {
        if self.direction != PioDirection::Out {
            return_errno_with_message!(Errno::EINVAL, "PIO operation is not an output");
        }

        let data_len = usize::try_from(count)?
            .checked_mul(usize::from(self.size))
            .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
        let mut data = vec![0_u8; data_len];
        if let Some(string) = self.string {
            let start = address_register(context, X86GprIndex::Rsi, string.address_size);
            for index in 0..count {
                let offset = string_offset(
                    start,
                    index,
                    self.size,
                    string.address_size,
                    direction_flag(context),
                );
                let linear = segmented_linear_address(context, string.segment, offset)?;
                let data_offset = usize::try_from(index)? * usize::from(self.size);
                read_guest_linear(
                    context,
                    vm_memory,
                    linear,
                    &mut data[data_offset..data_offset + usize::from(self.size)],
                )?;
            }
        } else {
            let value = context.gpr(X86GprIndex::Rax).to_le_bytes();
            data.copy_from_slice(&value[..usize::from(self.size)]);
        }
        Ok(data)
    }

    pub(super) fn output_values(self, data: &[u8]) -> Result<Vec<u64>> {
        if !data.len().is_multiple_of(usize::from(self.size)) {
            return_errno_with_message!(Errno::EINVAL, "PIO output buffer has an invalid length");
        }

        Ok(data
            .chunks_exact(usize::from(self.size))
            .map(|bytes| {
                let mut value = [0_u8; size_of::<u64>()];
                value[..bytes.len()].copy_from_slice(bytes);
                u64::from_le_bytes(value)
            })
            .collect())
    }

    pub(super) fn complete(
        self,
        context: &mut GuestContext,
        vm_memory: &VmMemory,
        count: u32,
        input_data: Option<&[u8]>,
    ) -> Result<()> {
        let Some(string) = self.string else {
            if self.direction == PioDirection::In {
                let data = input_data
                    .ok_or_else(|| Error::with_message(Errno::EINVAL, "missing PIO input data"))?;
                if data.len() != usize::from(self.size) {
                    return_errno_with_message!(
                        Errno::EINVAL,
                        "PIO input buffer has an invalid length"
                    );
                }
                let mut bytes = [0_u8; size_of::<u64>()];
                bytes[..data.len()].copy_from_slice(data);
                let value = u64::from_le_bytes(bytes);
                if self.size == 4 && is_64_bit_mode(context) {
                    context.set_gpr(X86GprIndex::Rax, 8, value);
                } else {
                    context.set_gpr(X86GprIndex::Rax, self.size, value);
                }
            }
            context.advance_rip(u64::from(self.instruction_len));
            return Ok(());
        };

        if self.direction == PioDirection::In {
            let data = input_data.ok_or_else(|| {
                Error::with_message(Errno::EINVAL, "missing string PIO input data")
            })?;
            let expected_len = usize::try_from(count)?
                .checked_mul(usize::from(self.size))
                .ok_or_else(|| Error::new(Errno::EOVERFLOW))?;
            if data.len() != expected_len {
                return_errno_with_message!(
                    Errno::EINVAL,
                    "string PIO input buffer has an invalid length"
                );
            }

            let start = address_register(context, X86GprIndex::Rdi, string.address_size);
            for index in 0..count {
                let offset = string_offset(
                    start,
                    index,
                    self.size,
                    string.address_size,
                    direction_flag(context),
                );
                let linear = segmented_linear_address(context, SegmentRegister::Es, offset)?;
                let data_offset = usize::try_from(index)? * usize::from(self.size);
                write_guest_linear(
                    context,
                    vm_memory,
                    linear,
                    &data[data_offset..data_offset + usize::from(self.size)],
                )?;
            }
        }

        let index_register = match self.direction {
            PioDirection::In => X86GprIndex::Rdi,
            PioDirection::Out => X86GprIndex::Rsi,
        };
        advance_address_register(
            context,
            index_register,
            string.address_size,
            self.size,
            count,
        );

        let completed = if string.repeated {
            decrement_count_register(context, string.address_size, count)
        } else {
            true
        };
        if completed {
            context.advance_rip(u64::from(self.instruction_len));
        }
        Ok(())
    }

    fn remaining_count(self, context: &GuestContext) -> u64 {
        match self.string {
            Some(string) if string.repeated => {
                address_register(context, X86GprIndex::Rcx, string.address_size)
            }
            _ => 1,
        }
    }
}

fn decode_string_instruction(
    context: &GuestContext,
    vm_memory: &VmMemory,
    direction: PioDirection,
    size: u8,
    repeated: bool,
    instruction_len: u32,
) -> Result<Option<StringPioOperation>> {
    let instruction_len = usize::try_from(instruction_len)?;
    if instruction_len == 0 || instruction_len > MAX_INSN_LENGTH {
        return Ok(None);
    }

    let sregs = context.sregs();
    let long_mode = is_64_bit_mode(context);
    let code_linear = if long_mode {
        context.rip()
    } else {
        sregs
            .cs
            .base
            .checked_add(context.rip())
            .ok_or_else(|| Error::new(Errno::EFAULT))?
    };
    let mut bytes = [0_u8; MAX_INSN_LENGTH];
    read_guest_linear(
        context,
        vm_memory,
        code_linear,
        &mut bytes[..instruction_len],
    )?;

    let default_address_size = if long_mode {
        8
    } else if sregs.cs.db != 0 {
        4
    } else {
        2
    };
    Ok(decode_string_bytes(
        &bytes[..instruction_len],
        direction,
        size,
        repeated,
        long_mode,
        default_address_size,
    ))
}

fn decode_string_bytes(
    bytes: &[u8],
    direction: PioDirection,
    size: u8,
    repeated: bool,
    long_mode: bool,
    default_address_size: u8,
) -> Option<StringPioOperation> {
    let mut offset = 0;
    let mut address_override = false;
    let mut segment_override = None;
    let mut decoded_rep = false;
    while offset < bytes.len() {
        let prefix = bytes[offset];
        let handled = match prefix {
            0x26 => {
                segment_override = Some(SegmentRegister::Es);
                true
            }
            0x2e => {
                segment_override = Some(SegmentRegister::Cs);
                true
            }
            0x36 => {
                segment_override = Some(SegmentRegister::Ss);
                true
            }
            0x3e => {
                segment_override = Some(SegmentRegister::Ds);
                true
            }
            0x64 => {
                segment_override = Some(SegmentRegister::Fs);
                true
            }
            0x65 => {
                segment_override = Some(SegmentRegister::Gs);
                true
            }
            0x67 => {
                address_override = true;
                true
            }
            0xf2 | 0xf3 => {
                decoded_rep = true;
                true
            }
            0x66 => true,
            byte if long_mode && byte & 0xf0 == 0x40 => true,
            _ => false,
        };
        if !handled {
            break;
        }
        offset += 1;
    }

    if decoded_rep != repeated || offset + 1 != bytes.len() {
        return None;
    }
    let opcode = bytes[offset];
    let opcode_matches = matches!(
        (direction, size, opcode),
        (PioDirection::In, 1, 0x6c)
            | (PioDirection::Out, 1, 0x6e)
            | (PioDirection::In, 2 | 4, 0x6d)
            | (PioDirection::Out, 2 | 4, 0x6f)
    );
    if !opcode_matches {
        return None;
    }

    let address_size = if address_override {
        match default_address_size {
            8 => 4,
            4 => 2,
            _ => 4,
        }
    } else {
        default_address_size
    };

    Some(StringPioOperation {
        address_size,
        segment: match direction {
            PioDirection::In => SegmentRegister::Es,
            PioDirection::Out => segment_override.unwrap_or(SegmentRegister::Ds),
        },
        repeated,
    })
}

fn read_guest_linear(
    context: &GuestContext,
    vm_memory: &VmMemory,
    linear: u64,
    bytes: &mut [u8],
) -> Result<()> {
    let mut completed = 0;
    while completed < bytes.len() {
        let current = linear
            .checked_add(u64::try_from(completed)?)
            .ok_or_else(|| Error::new(Errno::EFAULT))?;
        let gva = usize::try_from(current).map_err(|_| Error::new(Errno::EFAULT))?;
        let gpa = translate_gva_to_gpa(context, vm_memory, gva)?;
        let page_remaining = PAGE_SIZE - (gva & (PAGE_SIZE - 1));
        let copy_len = page_remaining.min(bytes.len() - completed);
        vm_memory.read_bytes(gpa, &mut bytes[completed..completed + copy_len])?;
        completed += copy_len;
    }
    Ok(())
}

fn write_guest_linear(
    context: &GuestContext,
    vm_memory: &VmMemory,
    linear: u64,
    bytes: &[u8],
) -> Result<()> {
    let mut completed = 0;
    while completed < bytes.len() {
        let current = linear
            .checked_add(u64::try_from(completed)?)
            .ok_or_else(|| Error::new(Errno::EFAULT))?;
        let gva = usize::try_from(current).map_err(|_| Error::new(Errno::EFAULT))?;
        let gpa = translate_gva_to_gpa(context, vm_memory, gva)?;
        let page_remaining = PAGE_SIZE - (gva & (PAGE_SIZE - 1));
        let copy_len = page_remaining.min(bytes.len() - completed);
        vm_memory.write_bytes(gpa, &bytes[completed..completed + copy_len])?;
        completed += copy_len;
    }
    Ok(())
}

fn segmented_linear_address(
    context: &GuestContext,
    segment: SegmentRegister,
    offset: u64,
) -> Result<u64> {
    segment_base(context, segment)
        .checked_add(offset)
        .ok_or_else(|| Error::new(Errno::EFAULT))
}

fn segment_base(context: &GuestContext, segment: SegmentRegister) -> u64 {
    let sregs = context.sregs();
    if is_64_bit_mode(context) && !matches!(segment, SegmentRegister::Fs | SegmentRegister::Gs) {
        return 0;
    }
    segment_value(&sregs, segment).base
}

fn segment_value(sregs: &ostd::arch::vm::VcpuSregs, segment: SegmentRegister) -> VcpuSegment {
    match segment {
        SegmentRegister::Cs => sregs.cs,
        SegmentRegister::Ss => sregs.ss,
        SegmentRegister::Ds => sregs.ds,
        SegmentRegister::Es => sregs.es,
        SegmentRegister::Fs => sregs.fs,
        SegmentRegister::Gs => sregs.gs,
    }
}

fn is_64_bit_mode(context: &GuestContext) -> bool {
    let sregs = context.sregs();
    sregs.efer & EFER_LONG_MODE_ACTIVE != 0 && sregs.cs.l != 0
}

fn direction_flag(context: &GuestContext) -> bool {
    context.regs().rflags & RFLAGS_DIRECTION != 0
}

fn address_register(context: &GuestContext, register: X86GprIndex, address_size: u8) -> u64 {
    context.gpr(register) & address_mask(address_size)
}

fn string_offset(
    start: u64,
    index: u32,
    element_size: u8,
    address_size: u8,
    decrement: bool,
) -> u64 {
    let delta = u64::from(index).wrapping_mul(u64::from(element_size));
    let value = if decrement {
        start.wrapping_sub(delta)
    } else {
        start.wrapping_add(delta)
    };
    value & address_mask(address_size)
}

fn advance_address_register(
    context: &mut GuestContext,
    register: X86GprIndex,
    address_size: u8,
    element_size: u8,
    count: u32,
) {
    let start = address_register(context, register, address_size);
    let delta = u64::from(count).wrapping_mul(u64::from(element_size));
    let value = if direction_flag(context) {
        start.wrapping_sub(delta)
    } else {
        start.wrapping_add(delta)
    } & address_mask(address_size);
    write_address_register(context, register, address_size, value);
}

/// Decrements CX/ECX/RCX and returns whether the repeated operation completed.
fn decrement_count_register(context: &mut GuestContext, address_size: u8, count: u32) -> bool {
    let remaining = address_register(context, X86GprIndex::Rcx, address_size)
        .wrapping_sub(u64::from(count))
        & address_mask(address_size);
    write_address_register(context, X86GprIndex::Rcx, address_size, remaining);
    remaining == 0
}

fn write_address_register(
    context: &mut GuestContext,
    register: X86GprIndex,
    address_size: u8,
    value: u64,
) {
    if address_size == 4 && is_64_bit_mode(context) {
        // A 32-bit address-register write zero-extends on x86-64.
        context.set_gpr(register, 8, value as u32 as u64);
    } else {
        context.set_gpr(register, address_size, value);
    }
}

fn address_mask(address_size: u8) -> u64 {
    match address_size {
        2 => u64::from(u16::MAX),
        4 => u64::from(u32::MAX),
        _ => u64::MAX,
    }
}
