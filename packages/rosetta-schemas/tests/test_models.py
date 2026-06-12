"""Tests for rosetta_schemas models — no external dependencies required."""

import pytest
from rosetta_schemas.models import ISAMeta, ISASpec, InstructionDef, RegisterDef
from rosetta_schemas.state import (
    PipelineState,
    get_instructions,
    get_isa_spec,
    get_meta,
    get_registers,
)


# ---------------------------------------------------------------------------
# ISAMeta
# ---------------------------------------------------------------------------

def test_isa_meta_roundtrip():
    meta = ISAMeta(
        name="ARM",
        endian="little",
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[16, 32],
    )
    restored = ISAMeta.model_validate(meta.model_dump())
    assert restored == meta


def test_isa_meta_defaults_endian():
    with pytest.raises(Exception):
        ISAMeta(name="X", endian="middle", word_size_bits=32, alignment=4, instruction_sizes_bits=[32])


# ---------------------------------------------------------------------------
# RegisterDef
# ---------------------------------------------------------------------------

def test_register_def_roundtrip():
    reg = RegisterDef(name="R0", aliases=["A1"], size_bits=32, description="General purpose")
    assert RegisterDef.model_validate(reg.model_dump()) == reg


def test_register_def_empty_aliases():
    reg = RegisterDef(name="PC", size_bits=32, description="Program counter")
    assert reg.aliases == []


# ---------------------------------------------------------------------------
# InstructionDef
# ---------------------------------------------------------------------------

def test_instruction_def_roundtrip():
    instr = InstructionDef(
        mnemonic="ADD",
        variants=["ADD Rd, Rn, Rm"],
        encoding_bits=32,
        bit_fields={"opcode": "31:28"},
        bit_constraints={"opcode": "0000"},
        operands=["Rd", "Rn", "Rm"],
        semantics="Add Rn and Rm, store in Rd",
        pcode_hint="Rd = Rn + Rm;",
    )
    restored = InstructionDef.model_validate(instr.model_dump())
    assert restored == instr


def test_instruction_def_defaults():
    instr = InstructionDef(mnemonic="NOP", encoding_bits=32, semantics="No operation")
    assert instr.variants == []
    assert instr.bit_fields == {}
    assert instr.bit_constraints == {}
    assert instr.operands == []
    assert instr.pcode_hint == ""


# ---------------------------------------------------------------------------
# ISASpec
# ---------------------------------------------------------------------------

def test_isa_spec_roundtrip():
    meta = ISAMeta(name="Test", endian="big", word_size_bits=64, alignment=8, instruction_sizes_bits=[64])
    reg = RegisterDef(name="X0", size_bits=64, description="GP")
    instr = InstructionDef(mnemonic="NOP", encoding_bits=32, semantics="No-op")
    spec = ISASpec(meta=meta, registers=[reg], instructions=[instr])
    restored = ISASpec.model_validate(spec.model_dump())
    assert restored == spec


# ---------------------------------------------------------------------------
# PipelineState accessors
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> PipelineState:
    base: PipelineState = {"errors": [], "registers": [], "instructions": []}
    base.update(kwargs)
    return base


def test_get_meta_returns_none_when_absent():
    state = _make_state()
    assert get_meta(state) is None


def test_get_meta_deserializes():
    meta = ISAMeta(name="ARM", endian="little", word_size_bits=32, alignment=4, instruction_sizes_bits=[32])
    state = _make_state(meta=meta.model_dump())
    result = get_meta(state)
    assert result is not None
    assert result.name == "ARM"
    assert result.endian == "little"


def test_get_registers_empty():
    assert get_registers(_make_state()) == []


def test_get_registers_deserializes():
    regs = [
        RegisterDef(name="R0", size_bits=32, description="GP"),
        RegisterDef(name="R1", size_bits=32, description="GP"),
    ]
    state = _make_state(registers=[r.model_dump() for r in regs])
    result = get_registers(state)
    assert len(result) == 2
    assert result[0].name == "R0"


def test_get_instructions_deserializes():
    instrs = [InstructionDef(mnemonic="ADD", encoding_bits=32, semantics="Add")]
    state = _make_state(instructions=[i.model_dump() for i in instrs])
    result = get_instructions(state)
    assert result[0].mnemonic == "ADD"


def test_get_isa_spec_none_without_meta():
    assert get_isa_spec(_make_state()) is None


def test_get_isa_spec_builds_correctly():
    meta = ISAMeta(name="T", endian="little", word_size_bits=32, alignment=4, instruction_sizes_bits=[32])
    reg = RegisterDef(name="R0", size_bits=32, description="GP")
    instr = InstructionDef(mnemonic="NOP", encoding_bits=32, semantics="No-op")
    state = _make_state(
        meta=meta.model_dump(),
        registers=[reg.model_dump()],
        instructions=[instr.model_dump()],
    )
    spec = get_isa_spec(state)
    assert spec is not None
    assert spec.meta.name == "T"
    assert len(spec.registers) == 1
    assert len(spec.instructions) == 1
