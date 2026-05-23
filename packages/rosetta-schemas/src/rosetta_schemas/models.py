from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EncodingStyle = Literal["opcode_table", "fixed_word", "variable_prefix"]


class ISAMeta(BaseModel):
    name: str = Field(description="Human-readable ISA name, e.g. 'ARM Cortex-M'")
    endian: Literal["little", "big", "bi"] = Field(
        description="Byte order; 'bi' generates both LE and BE variants"
    )
    word_size_bits: int = Field(description="Native word size in bits (32 or 64)")
    alignment: int = Field(description="Minimum instruction alignment in bytes")
    instruction_sizes_bits: list[int] = Field(
        description="Possible instruction widths in bits, e.g. [16, 32] for Thumb+ARM"
    )
    variant: str = Field(
        default="default",
        description="ISA version variant for the Ghidra language ID, e.g. 'v7', 'v8', 'Cortex'. "
                    "Appears as the 4th segment: PROCESSOR:ENDIAN:SIZE:variant",
    )
    encoding_style: EncodingStyle = Field(
        default="fixed_word",
        description=(
            "Instruction encoding family. "
            "'opcode_table': single opcode byte + variable operand bytes (6502, Z80, M7700). "
            "'fixed_word': fixed-width word with named bit fields (ARM, MIPS, RISC-V). "
            "'variable_prefix': multi-byte opcode with prefix bytes (x86)."
        ),
    )
    opcode_prefixes: list[int] = Field(
        default_factory=list,
        description=(
            "Prefix bytes that introduce secondary opcode tables, e.g. [0xCE, 0xCF] for M37700. "
            "Each value causes a full 256-entry scan of the prefixed opcode space. "
            "Empty for ISAs without prefix bytes."
        ),
    )


class OpcodeDef(BaseModel):
    """One row in a CISC opcode table (opcode_table encoding style)."""

    opcode: int = Field(description="Opcode byte value 0x00–0xFF")
    prefix: int | None = Field(
        default=None,
        description="Prefix byte for multi-byte opcode tables, e.g. 0x89 for M7700 MPY group",
    )
    mnemonic: str = Field(description="Instruction mnemonic, e.g. 'LDA'")

    @field_validator("mnemonic")
    @classmethod
    def _strip_mnemonic(cls, v: str) -> str:
        """Keep only the first word and uppercase it — LLMs sometimes embed operand syntax."""
        word = v.strip().split()[0] if v.strip() else "UNK"
        return word.upper()

    mode: str = Field(
        description=(
            "Addressing mode identifier, e.g. 'imp', 'imm', 'dp', 'dp,X', 'abs', "
            "'abs,X', 'abs,Y', '(dp)', '(dp,X)', '(dp),Y', 'rel', 'rel16', "
            "'long', 'long,X', 'sr', '(sr),Y', 'acc', 'mvn', 'mvp'"
        )
    )
    operand_bytes: int = Field(
        description="Number of operand bytes following the opcode byte(s). "
                    "E.g. 0 for implied, 1 for dp/rel, 2 for abs, 3 for long."
    )
    description: str = Field(default="", description="Brief description of the operation")
    pcode_body: str = Field(
        default="",
        description="SLEIGH pcode body for this constructor, e.g. 'A = *[ram]:2 w1;'. "
                    "Empty means the generator will use a mode-based stub.",
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
    opcode_map: list[OpcodeDef] = Field(default_factory=list)
