from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ISAMeta(BaseModel):
    name: str = Field(description="Human-readable ISA name, e.g. 'ARM Cortex-M'")
    endian: Literal["little", "big"] = Field(description="Byte order")
    word_size_bits: int = Field(description="Native word size in bits (32 or 64)")
    alignment: int = Field(description="Minimum instruction alignment in bytes")
    instruction_sizes_bits: list[int] = Field(
        description="Possible instruction widths in bits, e.g. [16, 32] for Thumb+ARM"
    )


class RegisterDef(BaseModel):
    name: str = Field(description="Canonical register name, e.g. 'R0' or 'X0'")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names, e.g. ['SP', 'LR', 'PC'] for R13/R14/R15",
    )
    size_bits: int = Field(description="Register width in bits")
    description: str = Field(description="Purpose or role of this register")


class InstructionDef(BaseModel):
    mnemonic: str = Field(description="Instruction mnemonic, e.g. 'ADD' or 'LDR'")
    variants: list[str] = Field(
        default_factory=list,
        description="Assembly syntax variants, e.g. ['ADD Rd, Rn, Rm', 'ADD Rd, #imm12']",
    )
    encoding_bits: int = Field(description="Instruction encoding width in bits (16 or 32)")
    bit_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Bit field definitions: field_name -> 'high:low' bit range",
    )
    bit_constraints: dict[str, str] = Field(
        default_factory=dict,
        description="Required bit values: field_name -> binary string",
    )
    operands: list[str] = Field(
        default_factory=list,
        description="Operand names in order, e.g. ['Rd', 'Rn', 'imm12']",
    )
    semantics: str = Field(
        description="Natural-language description of the instruction's operation"
    )
    pcode_hint: str = Field(
        default="",
        description="SLEIGH P-code approximation of the semantics",
    )


class ISASpec(BaseModel):
    meta: ISAMeta
    registers: list[RegisterDef] = Field(default_factory=list)
    instructions: list[InstructionDef] = Field(default_factory=list)
