// SPDX-License-Identifier: MPL-2.0

//! Decodes guest instructions that access emulated MMIO regions.

use ostd::arch::vm::{GuestContext, X86GprIndex};

use super::{guest_address::translate_gva_to_gpa, vm_memory::VmMemory};
use crate::prelude::*;

const MAX_INSN_LENGTH: usize = 15;
const EFER_LONG_MODE_ACTIVE: u64 = 1 << 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InstructionMode {
    long_mode: bool,
    default_operand_size: u8,
    default_address_size: u8,
}

impl InstructionMode {
    fn from_context(context: &GuestContext) -> Self {
        let sregs = context.sregs();
        let long_mode = sregs.efer & EFER_LONG_MODE_ACTIVE != 0 && sregs.cs.l != 0;
        Self {
            long_mode,
            default_operand_size: if !long_mode && sregs.cs.db == 0 { 2 } else { 4 },
            default_address_size: if long_mode {
                8
            } else if sregs.cs.db != 0 {
                4
            } else {
                2
            },
        }
    }

    fn operand_size(self, rex_w: bool, overridden: bool) -> u8 {
        if rex_w {
            8
        } else if overridden {
            match self.default_operand_size {
                2 => 4,
                _ => 2,
            }
        } else {
            self.default_operand_size
        }
    }

    fn address_size(self, overridden: bool) -> u8 {
        if !overridden {
            return self.default_address_size;
        }
        match self.default_address_size {
            8 => 4,
            4 => 2,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MmioDirection {
    /// Reads from MMIO regions
    Read,
    /// Writes to MMIO regions
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MmioRegister {
    index: X86GprIndex,
    high_byte: bool,
}

impl MmioRegister {
    fn read(self, context: &GuestContext, size: u8) -> u64 {
        let value = context.gpr(self.index);
        let value = if self.high_byte { value >> 8 } else { value };
        value & value_mask(size)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MmioReadExtension {
    None,
    Sign,
    Zero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MmioReadTarget {
    register: MmioRegister,
    size: u8,
    extension: MmioReadExtension,
}

impl MmioReadTarget {
    fn write(self, context: &mut GuestContext, access_size: u8, value: u64) {
        let value = match self.extension {
            MmioReadExtension::None | MmioReadExtension::Zero => value & value_mask(access_size),
            MmioReadExtension::Sign => sign_extend(value, access_size),
        } & value_mask(self.size);

        if self.register.high_byte {
            let old_value = context.gpr(self.register.index);
            let new_value = (old_value & !0xff00) | (value << 8);
            context.set_gpr(self.register.index, 8, new_value);
        } else if self.size == 4 {
            // Writes to a 32-bit GPR zero the upper half in 64-bit mode.
            context.set_gpr(self.register.index, 8, value);
        } else {
            context.set_gpr(self.register.index, self.size, value);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MmioOperand {
    Register(MmioRegister),
    Immediate(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MmioInstruction {
    direction: MmioDirection,
    size: u8,
    read_target: Option<MmioReadTarget>,
    write_operand: Option<MmioOperand>,
    len: u8,
}

impl MmioInstruction {
    pub(super) fn direction(self) -> MmioDirection {
        self.direction
    }

    pub(super) fn size(self) -> u8 {
        self.size
    }

    pub(super) fn len(self) -> u8 {
        self.len
    }

    pub(super) fn write_value(self, context: &GuestContext) -> Option<u64> {
        match self.write_operand? {
            MmioOperand::Register(register) => Some(register.read(context, self.size)),
            MmioOperand::Immediate(value) => Some(value & value_mask(self.size)),
        }
    }

    pub(super) fn complete_read(self, context: &mut GuestContext, value: u64) -> Result<()> {
        let Some(target) = self.read_target else {
            return_errno_with_message!(Errno::EINVAL, "MMIO instruction is not a read");
        };
        target.write(context, self.size, value);
        Ok(())
    }
}

/// Fetches and decodes the instruction at the guest's current RIP.
pub(super) fn decode_current_mmio_instruction(
    context: &GuestContext,
    vm_memory: &VmMemory,
) -> Result<Option<MmioInstruction>> {
    let mode = InstructionMode::from_context(context);
    let guest_rip = context.rip();
    let code_vaddr = if mode.long_mode {
        guest_rip
    } else {
        context
            .sregs()
            .cs
            .base
            .checked_add(guest_rip)
            .ok_or_else(|| Error::new(Errno::EFAULT))?
    };
    let code_vaddr = usize::try_from(code_vaddr)?;
    let mut bytes = [0_u8; MAX_INSN_LENGTH];
    let mut bytes_read = 0;

    while bytes_read < bytes.len() {
        let guest_vaddr = code_vaddr
            .checked_add(bytes_read)
            .ok_or_else(|| Error::new(Errno::EFAULT))?;
        let guest_paddr = translate_gva_to_gpa(context, vm_memory, guest_vaddr)?;
        let page_remaining = PAGE_SIZE - (guest_vaddr & (PAGE_SIZE - 1));
        let read_len = page_remaining.min(bytes.len() - bytes_read);
        vm_memory.read_bytes(guest_paddr, &mut bytes[bytes_read..bytes_read + read_len])?;
        bytes_read += read_len;

        if let Some(instruction) = decode_mmio_instruction(&bytes[..bytes_read], mode) {
            return Ok(Some(instruction));
        }
    }

    error!(
        "hypervisor: failed to decode MMIO instruction: rip={:#x}, bytes={:02x?}",
        guest_rip,
        &bytes[..bytes_read]
    );
    Ok(None)
}

fn decode_mmio_instruction(bytes: &[u8], mode: InstructionMode) -> Option<MmioInstruction> {
    let mut offset = 0usize;
    let mut operand_size_override = false;
    let mut address_size_override = false;
    let mut rex = None;

    // Prefix
    // Volume 2A, 2.1.1 Instruction Prefixes
    while offset < bytes.len() {
        let byte = bytes[offset];
        match byte {
            0x66 => {
                operand_size_override = true;
                rex = None;
            }
            0x67 => {
                address_size_override = true;
                rex = None;
            }
            0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65 | 0xf0 | 0xf2 | 0xf3 => {
                rex = None;
            }
            byte if mode.long_mode && (byte & 0xf0) == 0x40 => rex = Some(byte),
            _ => break,
        }
        offset += 1;
    }

    let opcode = *bytes.get(offset)?;
    offset += 1;

    let rex_value = rex.unwrap_or(0);
    let rex_w = (rex_value & 0x08) != 0;
    let operand_size = mode.operand_size(rex_w, operand_size_override);
    let address_size = mode.address_size(address_size_override);

    // A0-A3 encode the memory offset directly after the opcode and always use
    // AL/AX/EAX/RAX as the register operand, so they have no ModRM byte.
    if matches!(opcode, 0xa0..=0xa3) {
        return decode_moffs_instruction(bytes, offset, opcode, operand_size, address_size);
    }

    // MOV Opcode
    // Volume 2, 4.3 Instructions(M-U)
    let (direction, size, immediate_size, read_target) = match opcode {
        // MOVZX & MOVSX
        0x0f => {
            let secondary_opcode = *bytes.get(offset)?;
            offset += 1;
            let destination_size = operand_size;
            match secondary_opcode {
                0xb6 => (
                    MmioDirection::Read,
                    1,
                    0,
                    Some((destination_size, MmioReadExtension::Zero, false)),
                ),
                0xb7 => (
                    MmioDirection::Read,
                    2,
                    0,
                    Some((destination_size, MmioReadExtension::Zero, false)),
                ),
                0xbe => (
                    MmioDirection::Read,
                    1,
                    0,
                    Some((destination_size, MmioReadExtension::Sign, false)),
                ),
                0xbf => (
                    MmioDirection::Read,
                    2,
                    0,
                    Some((destination_size, MmioReadExtension::Sign, false)),
                ),
                _ => return None,
            }
        }
        // MOVSXD
        0x63 if rex_w => (
            MmioDirection::Read,
            4,
            0,
            Some((8, MmioReadExtension::Sign, false)),
        ),
        // MOV
        0x88 => (MmioDirection::Write, 1, 0, None),
        0x8a => (
            MmioDirection::Read,
            1,
            0,
            Some((1, MmioReadExtension::None, true)),
        ),
        0x89 => (MmioDirection::Write, operand_size, 0, None),
        0x8b => {
            let size = operand_size;
            (
                MmioDirection::Read,
                size,
                0,
                Some((size, MmioReadExtension::None, false)),
            )
        }
        0xc6 => (MmioDirection::Write, 1, 1, None),
        0xc7 => (
            MmioDirection::Write,
            operand_size,
            if operand_size == 2 { 2 } else { 4 },
            None,
        ),
        _ => return None,
    };

    // modrm
    // bit: 7 6 | 5 4 3 | 2 1 0
    //      mod |  reg  |  r/m
    // mod: r/m is reg or memory
    //      11    for reg
    //      other for memory
    // Volume 2, 2.1 Instruction Format...
    let (modrm, new_offset) = decode_modrm(bytes, offset, address_size)?;
    offset = new_offset;
    let mode = modrm >> 6;
    if mode == 0b11 {
        return None;
    }

    let opcode_extension = (modrm >> 3) & 0x7;
    let register_encoding = opcode_extension | ((rex_value & 0x04) << 1);
    let (read_target, write_operand) = match direction {
        MmioDirection::Read => {
            let (target_size, extension, allow_high_byte) = read_target?;
            let register = decode_register(register_encoding, allow_high_byte && rex.is_none())?;
            (
                Some(MmioReadTarget {
                    register,
                    size: target_size,
                    extension,
                }),
                None,
            )
        }
        MmioDirection::Write if immediate_size == 0 => {
            let register = decode_register(register_encoding, size == 1 && rex.is_none())?;
            (None, Some(MmioOperand::Register(register)))
        }
        MmioDirection::Write => {
            if opcode_extension != 0 {
                return None;
            }
            let mut immediate = read_le_immediate(bytes, offset, immediate_size)?;
            offset += immediate_size;
            if size == 8 {
                immediate = (immediate as u32 as i32 as i64) as u64;
            }
            (None, Some(MmioOperand::Immediate(immediate)))
        }
    };

    let len = u8::try_from(offset).ok()?;
    if usize::from(len) > MAX_INSN_LENGTH {
        return None;
    }
    Some(MmioInstruction {
        direction,
        size,
        read_target,
        write_operand,
        len,
    })
}

fn decode_moffs_instruction(
    bytes: &[u8],
    offset: usize,
    opcode: u8,
    operand_size: u8,
    address_size: u8,
) -> Option<MmioInstruction> {
    let end = offset.checked_add(usize::from(address_size))?;
    if end > bytes.len() || end > MAX_INSN_LENGTH {
        return None;
    }

    let accumulator = MmioRegister {
        index: X86GprIndex::Rax,
        high_byte: false,
    };
    let size = if matches!(opcode, 0xa0 | 0xa2) {
        1
    } else {
        operand_size
    };
    let (direction, read_target, write_operand) = match opcode {
        0xa0 | 0xa1 => (
            MmioDirection::Read,
            Some(MmioReadTarget {
                register: accumulator,
                size,
                extension: MmioReadExtension::None,
            }),
            None,
        ),
        0xa2 | 0xa3 => (
            MmioDirection::Write,
            None,
            Some(MmioOperand::Register(accumulator)),
        ),
        _ => return None,
    };

    Some(MmioInstruction {
        direction,
        size,
        read_target,
        write_operand,
        len: u8::try_from(end).ok()?,
    })
}

fn decode_modrm(bytes: &[u8], offset: usize, address_size: u8) -> Option<(u8, usize)> {
    let modrm = *bytes.get(offset)?;
    let mode = modrm >> 6;
    let rm = modrm & 0x7;
    let mut end = offset.checked_add(1)?;

    if mode == 0b11 {
        return Some((modrm, end));
    }

    if address_size == 2 {
        end = match mode {
            0 if rm == 0x6 => end.checked_add(2)?,
            0 => end,
            1 => end.checked_add(1)?,
            2 => end.checked_add(2)?,
            _ => return None,
        };
    } else if matches!(address_size, 4 | 8) {
        if rm == 0x4 {
            let sib = *bytes.get(end)?;
            end = end.checked_add(1)?;
            if mode == 0 && (sib & 0x7) == 0x5 {
                end = end.checked_add(4)?;
            }
        } else if mode == 0 && rm == 0x5 {
            end = end.checked_add(4)?;
        }
        end = match mode {
            0 => end,
            1 => end.checked_add(1)?,
            2 => end.checked_add(4)?,
            _ => return None,
        };
    } else {
        return None;
    }

    if end > bytes.len() || end > MAX_INSN_LENGTH {
        return None;
    }
    Some((modrm, end))
}

fn decode_register(encoding: u8, allow_high_byte: bool) -> Option<MmioRegister> {
    let (encoding, high_byte) = if allow_high_byte && (4..8).contains(&encoding) {
        (encoding - 4, true)
    } else {
        (encoding, false)
    };
    let index = X86GprIndex::from_x86_reg_encoding(encoding).ok()?;
    Some(MmioRegister { index, high_byte })
}

fn sign_extend(value: u64, size: u8) -> u64 {
    match size {
        1 => (value as u8 as i8 as i64) as u64,
        2 => (value as u16 as i16 as i64) as u64,
        4 => (value as u32 as i32 as i64) as u64,
        _ => value,
    }
}

fn read_le_immediate(bytes: &[u8], offset: usize, size: usize) -> Option<u64> {
    let end = offset.checked_add(size)?;
    if end > bytes.len() || end > MAX_INSN_LENGTH {
        return None;
    }

    let mut value = 0_u64;
    for (index, byte) in bytes[offset..end].iter().enumerate() {
        value |= u64::from(*byte) << (index * 8);
    }
    Some(value)
}

fn value_mask(size: u8) -> u64 {
    match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    }
}
