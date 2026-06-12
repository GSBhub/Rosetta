"""Tests for SLEIGH module generation from ISASpec."""

import re
from pathlib import Path

import pytest

from rosetta_generate_sla.sla.module_generator import ModuleGenerator


def test_generates_four_files(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)

    for suffix in (".slaspec", ".pspec", ".cspec", ".ldefs"):
        assert (lang_dir / f"TestISA{suffix}").exists(), f"Missing TestISA{suffix}"


def test_slaspec_contains_endian(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.slaspec").read_text()
    assert "define endian=little" in text


def test_slaspec_contains_all_mnemonics(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.slaspec").read_text()
    for instr in minimal_spec.instructions:
        assert instr.mnemonic in text, f"Mnemonic {instr.mnemonic} missing from slaspec"


def test_slaspec_contains_pcode_hints(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.slaspec").read_text()
    # At least one P-code hint should appear
    assert "R0 = R1" in text or "R0 = R0 + R1" in text


def test_slaspec_contains_all_registers(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.slaspec").read_text()
    for reg in minimal_spec.registers:
        assert reg.name in text, f"Register {reg.name} missing from slaspec"


def test_ldefs_contains_processor_name(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.ldefs").read_text()
    assert "TestISA" in text
    assert "little" in text


def test_pspec_contains_pc(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.pspec").read_text()
    # PC register should appear as programcounter
    assert "PC" in text or "pc" in text.lower()


def test_cspec_contains_stack_pointer(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    text = (lang_dir / "TestISA.cspec").read_text()
    assert "SP" in text


def test_output_directory_structure(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    assert lang_dir == tmp_path / "TestISA" / "data" / "languages"


def test_64bit_word_size(tmp_path):
    from rosetta_schemas.models import ISAMeta, ISASpec, RegisterDef, InstructionDef
    spec = ISASpec(
        meta=ISAMeta(name="MyISA64", endian="big", word_size_bits=64, alignment=8, instruction_sizes_bits=[64]),
        registers=[RegisterDef(name="X0", size_bits=64, description="GP"), RegisterDef(name="PC", size_bits=64, description="PC")],
        instructions=[InstructionDef(mnemonic="NOP", semantics="No operation.", encoding_bits=64, pcode_hint="# nop")],
    )
    gen = ModuleGenerator()
    lang_dir = gen.generate(spec, "MyISA64", tmp_path)
    text = (lang_dir / "MyISA64.slaspec").read_text()
    assert "define endian=big" in text
    assert "NOP" in text
