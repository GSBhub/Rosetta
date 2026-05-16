"""Tests for extraction Pydantic schemas."""

import pytest
from pydantic import ValidationError

from rosetta.extraction.schemas import ISAMeta, ISASpec, InstructionDef, RegisterDef


def test_isa_meta_valid():
    m = ISAMeta(name="ARM", endian="little", word_size_bits=32, alignment=4, instruction_sizes_bits=[32])
    assert m.endian == "little"
    assert m.word_size_bits == 32


def test_isa_meta_rejects_bad_endian():
    with pytest.raises(ValidationError):
        ISAMeta(name="X", endian="medium", word_size_bits=32, alignment=4, instruction_sizes_bits=[32])


def test_register_defaults():
    r = RegisterDef(name="R0", size_bits=32, description="GP")
    assert r.aliases == []


def test_instruction_defaults():
    i = InstructionDef(mnemonic="NOP", semantics="Does nothing.", encoding_bits=32)
    assert i.variants == []
    assert i.bit_fields == {}
    assert i.bit_constraints == {}
    assert i.operands == []
    assert i.pcode_hint == ""


def test_isa_spec_roundtrip(minimal_spec):
    json_str = minimal_spec.model_dump_json()
    restored = ISASpec.model_validate_json(json_str)
    assert restored.meta.name == minimal_spec.meta.name
    assert len(restored.instructions) == len(minimal_spec.instructions)
    assert restored.instructions[0].mnemonic == "ADD"


def test_isa_spec_empty_registers():
    spec = ISASpec(
        meta=ISAMeta(name="Minimal", endian="big", word_size_bits=64, alignment=8, instruction_sizes_bits=[64]),
    )
    assert spec.registers == []
    assert spec.instructions == []
