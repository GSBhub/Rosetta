"""Tests for spec_loader: mnemonic/register extraction + include resolution."""

import pytest
from pathlib import Path

from rosetta.evaluation.spec_loader import (
    extract_mnemonics,
    extract_register_names,
    load_ghidra_reference,
    load_slaspec_text,
)
from tests.conftest import requires_ghidra, GHIDRA_HOME


# ---------------------------------------------------------------------------
# Unit tests on synthetic SLEIGH text (no Ghidra required)
# ---------------------------------------------------------------------------

SYNTHETIC_SLASPEC = """\
define endian=little;
define space ram type=ram_space size=4 wordsize=1 default;
define space register type=register_space size=4;

define register offset=0x00 size=4 [ R0 R1 R2 PC SP ];

define token instr32(32) opcode=(28,31);

:ADD R0, R1 is opcode=0b0000 { R0 = R0 + R1; }
:SUB R0, R1 is opcode=0b0001 { R0 = R0 - R1; }
:MOV R0, R1 is opcode=0b0010 { R0 = R1; }
:B   label  is opcode=0b1010 { goto [label]; }
:^instruction is opcode=0xF   { }
"""


def test_extract_mnemonics_basic():
    mnemonics = extract_mnemonics(SYNTHETIC_SLASPEC)
    assert "ADD" in mnemonics
    assert "SUB" in mnemonics
    assert "MOV" in mnemonics
    assert "B" in mnemonics


def test_extract_mnemonics_excludes_context_override():
    # :^instruction is a context-sensitive override, not a standalone instruction
    mnemonics = extract_mnemonics(SYNTHETIC_SLASPEC)
    assert "^INSTRUCTION" not in mnemonics
    assert "INSTRUCTION" not in mnemonics  # "^instruction" shouldn't match [A-Za-z] start


def test_extract_register_names_basic():
    regs = extract_register_names(SYNTHETIC_SLASPEC)
    assert "R0" in regs
    assert "R1" in regs
    assert "PC" in regs
    assert "SP" in regs


def test_load_slaspec_text_no_includes(tmp_path):
    spec = tmp_path / "test.slaspec"
    spec.write_text(":ADD rd is opcode=0 { rd = rd + 1; }")
    text = load_slaspec_text(spec)
    assert ":ADD" in text


def test_load_slaspec_text_with_include(tmp_path):
    inc = tmp_path / "defs.sinc"
    inc.write_text(":SUB rd is opcode=1 { rd = rd - 1; }")
    spec = tmp_path / "test.slaspec"
    spec.write_text('@include "defs.sinc"\n:ADD rd is opcode=0 { rd = rd + 1; }')
    text = load_slaspec_text(spec)
    assert ":ADD" in text
    assert ":SUB" in text


def test_load_slaspec_text_recursive_include(tmp_path):
    deep = tmp_path / "deep.sinc"
    deep.write_text(":MUL rd is opcode=2 { rd = rd * rd; }")
    mid = tmp_path / "mid.sinc"
    mid.write_text(f'@include "deep.sinc"\n:SUB rd is opcode=1 {{ rd = rd - 1; }}')
    spec = tmp_path / "test.slaspec"
    spec.write_text('@include "mid.sinc"\n:ADD rd is opcode=0 { rd = rd + 1; }')
    text = load_slaspec_text(spec)
    assert ":ADD" in text
    assert ":SUB" in text
    assert ":MUL" in text


def test_load_slaspec_text_no_duplicate_on_cycle(tmp_path):
    # Guard against infinite recursion if spec includes itself
    spec = tmp_path / "test.slaspec"
    spec.write_text('@include "test.slaspec"\n:NOP is opcode=0 { }')
    text = load_slaspec_text(spec)
    assert text.count(":NOP") == 1


# ---------------------------------------------------------------------------
# Integration tests against real Ghidra slaspec files
# ---------------------------------------------------------------------------

@requires_ghidra
def test_arm7_le_mnemonics_nonempty(ghidra_arm_languages_dir):
    arm7 = ghidra_arm_languages_dir / "ARM7_le.slaspec"
    text = load_slaspec_text(arm7)
    mnemonics = extract_mnemonics(text)
    # ARM7 should have at least 50 distinct mnemonics
    assert len(mnemonics) >= 50, f"Only found {len(mnemonics)} mnemonics in ARM7_le.slaspec"


@requires_ghidra
def test_arm7_le_known_mnemonics(ghidra_arm_languages_dir):
    arm7 = ghidra_arm_languages_dir / "ARM7_le.slaspec"
    text = load_slaspec_text(arm7)
    mnemonics = extract_mnemonics(text)
    for expected in ["ADD", "SUB", "MOV", "LDR", "STR", "B", "BL", "CMP", "AND", "ORR"]:
        assert expected in mnemonics, f"Expected {expected} in ARM7 mnemonics"


@requires_ghidra
def test_arm7_le_registers_nonempty(ghidra_arm_languages_dir):
    arm7 = ghidra_arm_languages_dir / "ARM7_le.slaspec"
    text = load_slaspec_text(arm7)
    regs = extract_register_names(text)
    for expected in {"R0", "R1", "SP", "PC"}:
        assert expected.upper() in regs, f"Expected {expected} in ARM7 registers"


@requires_ghidra
def test_load_ghidra_reference_by_lang_id():
    path = load_ghidra_reference(GHIDRA_HOME, "ARM:LE:32:v7")
    assert path.exists()
    assert path.name == "ARM7_le.slaspec"


@requires_ghidra
def test_load_ghidra_reference_by_processor_name():
    path = load_ghidra_reference(GHIDRA_HOME, "ARM")
    assert path.exists()
    assert path.suffix == ".slaspec"


@requires_ghidra
def test_all_arm_variants_have_mnemonics(ghidra_arm_languages_dir):
    for slaspec in sorted(ghidra_arm_languages_dir.glob("*.slaspec")):
        text = load_slaspec_text(slaspec)
        m = extract_mnemonics(text)
        assert len(m) > 0, f"{slaspec.name} yielded 0 mnemonics — include chain may be broken"
