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


# ---------------------------------------------------------------------------
# Compile-safety regressions (output must be valid SLEIGH, not just present)
# ---------------------------------------------------------------------------

def test_register_block_has_no_duplicate_names(tmp_path):
    """Extraction emits dup register names; the slaspec must not (SLEIGH rejects them)."""
    regs = [
        RegisterDef(name="R15", size_bits=32, aliases=["PC"], description="pc"),
        RegisterDef(name="R15", size_bits=32, description="dup pc"),  # duplicate
        RegisterDef(name="r15", size_bits=32, description="case-dup"),  # case-insensitive dup
        RegisterDef(name="SP", size_bits=32, description="sp"),
    ]
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=regs, processor_name="TestISA", out_dir=tmp_path)
    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()
    block = re.search(r"define register[^\[]*\[([^\]]+)\]", slaspec, re.DOTALL).group(1)
    names = re.findall(r"[A-Za-z]\w*", block)
    assert len(names) == len({n.upper() for n in names}), f"duplicate register names: {names}"
    assert sorted({n.upper() for n in names}) == ["R15", "SP"]


def test_constructors_get_unique_patterns_not_epsilon(tmp_path):
    """Every constructor needs a distinct opcode pattern; all-epsilon would conflict."""
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    for mnem in ("ADD", "SUB", "MOV", "ORR"):
        w.write_instruction(_instr(mnem))
    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()

    assert "is epsilon" not in slaspec
    patterns = re.findall(r"^:\w+ is (op32stub=\d+)", slaspec, re.MULTILINE)
    assert len(patterns) == 4
    assert len(set(patterns)) == 4, f"opcode patterns not unique: {patterns}"


def test_streamed_instruction_of_undeclared_width_snaps_to_header_token(tmp_path):
    """A later instruction whose width the header never declared must not emit an
    undefined op{w}stub. In streaming mode the header is written once up front from
    meta.instruction_sizes_bits ([32] here); an instruction arriving with
    encoding_bits=16 would otherwise reference op16stub and fail to compile
    ('unknown family or operand'). It must snap to the header's op32stub."""
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    w.write_instruction(_instr("ADD"))  # 32-bit, declared
    narrow = InstructionDef(mnemonic="AESE", encoding_bits=16, semantics="narrow")
    w.write_instruction(narrow)
    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()

    # Only the header-declared stub token is referenced; the 16-bit one was snapped.
    assert "op16stub" not in slaspec
    stub_refs = re.findall(r"^:\w+ is (op\d+stub)=\d+", slaspec, re.MULTILINE)
    assert set(stub_refs) == {"op32stub"}
    # Stub indices stay globally unique across the snap.
    values = re.findall(r"op32stub=(\d+)", slaspec)
    assert len(values) == len(set(values))


def test_pcode_body_never_emits_hallucinated_identifiers(tmp_path):
    """Hint referencing undefined regs (rd/ra/rb) must become a comment + safe no-op."""
    w = SlaInstructionWriter()
    w.open(meta=_meta(), registers=_registers(), processor_name="TestISA", out_dir=tmp_path)
    w.write_instruction(_instr("ADD", pcode="rd = ra + rb;"))
    slaspec = (w.lang_dir / "TestISA.slaspec").read_text()

    body = re.search(r":ADD is[^\n]*\n\{\n(.*?)\n\}", slaspec, re.DOTALL).group(1)
    assert "rd = ra + rb" in body and "# rd = ra + rb" in body  # preserved as comment
    assert "local tmp:4 = 0;" in body                            # safe no-op statement
    # no executable (uncommented) line references the hallucinated identifiers
    code_lines = [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert all("ra" not in l and "rd" not in l for l in code_lines)
