"""Tests for SlaInstructionWriter — real Jinja + tmp_path, no external deps."""

import re
from pathlib import Path

import pytest
from rosetta_schemas.models import InstructionDef, ISAMeta, RegisterDef

from rosetta_generate_sla.writers.base import get_writer
from rosetta_generate_sla.writers.sla_writer import SlaInstructionWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _meta(endian="little") -> ISAMeta:
    return ISAMeta(
        name="TestISA",
        endian=endian,
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[32],
    )


def _registers() -> list[RegisterDef]:
    return [
        RegisterDef(name="PC", size_bits=32, description="Program counter"),
        RegisterDef(name="SP", size_bits=32, description="Stack pointer"),
    ]


def _instr(mnemonic: str, pcode: str = "local tmp:4 = 0;") -> InstructionDef:
    return InstructionDef(
        mnemonic=mnemonic,
        encoding_bits=32,
        semantics=f"{mnemonic} semantics",
        pcode_hint=pcode,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_writer_sla():
    w = get_writer("sla")
    assert isinstance(w, SlaInstructionWriter)


def test_get_writer_sleigh():
    w = get_writer("sleigh")
    assert isinstance(w, SlaInstructionWriter)


def test_get_writer_unknown():
    with pytest.raises(KeyError, match="Unknown output format"):
        get_writer("unknown_format_xyz")


def test_open_writes_four_files(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)

    lang_dir = w.lang_dir
    assert lang_dir is not None
    assert lang_dir.exists()

    names = {f.name for f in lang_dir.iterdir()}
    assert "TestISA.slaspec" in names
    assert "TestISA.pspec" in names
    assert "TestISA.cspec" in names
    assert "TestISA.ldefs" in names


def test_open_slaspec_has_no_constructors(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)

    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()
    # Header should be present but no instruction constructors (no `:MNEMONIC`)
    constructor_lines = [l for l in slaspec.splitlines() if re.match(r"^:[A-Z]", l)]
    assert constructor_lines == []


def test_write_instruction_appends_constructor(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    w.write_instruction(_instr("ADD", pcode="R0 = R1 + R2;"))

    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()
    assert ":ADD" in slaspec


def test_write_instruction_deduplicates(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    w.write_instruction(_instr("ADD"))
    w.write_instruction(_instr("ADD"))  # duplicate — should not be appended again

    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()
    count = len(re.findall(r"^:ADD\b", slaspec, re.MULTILINE))
    assert count == 1


def test_write_multiple_instructions(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    for mnemonic in ("ADD", "SUB", "MOV", "NOP"):
        w.write_instruction(_instr(mnemonic))

    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()
    for mnemonic in ("ADD", "SUB", "MOV", "NOP"):
        assert f":{mnemonic}" in slaspec


def test_bi_endian_writes_both_files(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta("bi"), registers=_registers(), processor_name="BiISA", out_dir=tmp_path)
    w.write_instruction(_instr("MOV"))

    lang_dir = w.lang_dir
    assert (lang_dir / "BiISA_le.slaspec").exists()
    assert (lang_dir / "BiISA_be.slaspec").exists()
    for path in [lang_dir / "BiISA_le.slaspec", lang_dir / "BiISA_be.slaspec"]:
        assert ":MOV" in path.read_text()


def test_lang_dir_set_after_open(tmp_path):
    w = SlaInstructionWriter()
    assert w.lang_dir is None
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    assert w.lang_dir is not None


def test_close_sets_lang_dir(tmp_path):
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    w.close()
    assert w.lang_dir is not None
