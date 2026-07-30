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
    # --- Backend-neutral structured-decode extensions (Tier 1) ---
    # These are generic ISA concepts (a VLIW DSP just exercises them hardest).
    # Values are supplied per-ISA as data (examples/*.toml) and consumed by the
    # renderer; they carry over unchanged to a future QEMU/TCG backend.
    isa_variants: list[str] = Field(
        default_factory=list,
        description=(
            "ISA family/feature variants that share one encoding, e.g. "
            "['C62x','C64x','C67x','C67x+']. When non-empty the SLEIGH backend "
            "emits one <language> per entry (all sharing the superset .sla). "
            "Empty means a single language keyed on `variant`."
        ),
    )
    parallel_field: str | None = Field(
        default=None,
        description=(
            "Name of the bit field that marks parallel issue with the next "
            "instruction word (VLIW p-bit, e.g. 'p'). Rendered as a '||' display "
            "prefix when set. None for non-VLIW ISAs."
        ),
    )
    predicate_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Leading predication field group rendered as a conditional display "
            "prefix, e.g. ['creg','z'] for TMS320 conditional execution. Empty "
            "for unconditional ISAs."
        ),
    )
    field_attachments: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "SLEIGH `attach names` data shared across all instructions: bit field "
            "name -> ordered symbol names indexed by field value (e.g. "
            "'s' -> ['1','2'] for the C6x register-side bit). Renders raw bit "
            "fields as symbolic operands. Per-instruction attachments are not "
            "needed when the field layout is uniform."
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
    isa_variants: list[str] = Field(
        default_factory=list,
        description=(
            "ISA families that support this instruction (from the manual's "
            "'Compatibility' line), e.g. ['C67x','C67x+']. Informational in "
            "Tier 1 (the superset language decodes all); enables per-family "
            "gating in Tier 2. A subset of ISAMeta.isa_variants."
        ),
    )
    functional_unit: str = Field(
        default="",
        description=(
            "Functional unit letter this instruction issues on, e.g. 'S','L',"
            "'M','D' for TMS320C6x. Combined with the side/attachment field it "
            "renders the '.unit' qualifier (e.g. '.S1'). Empty when N/A."
        ),
    )


class ISASpec(BaseModel):
    meta: ISAMeta
    registers: list[RegisterDef] = Field(default_factory=list)
    instructions: list[InstructionDef] = Field(default_factory=list)
    opcode_map: list[OpcodeDef] = Field(default_factory=list)
