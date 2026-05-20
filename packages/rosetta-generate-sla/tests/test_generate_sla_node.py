"""Tests for rosetta_generate_sla.node and ModuleGenerator — no external deps."""

import pytest
from rosetta_schemas.models import ISAMeta, ISASpec, InstructionDef, RegisterDef

from rosetta_generate_sla.node import generate_sla_node
from rosetta_generate_sla.sla.module_generator import ModuleGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _meta() -> ISAMeta:
    return ISAMeta(
        name="TestISA",
        endian="little",
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[32],
    )


def _registers() -> list[RegisterDef]:
    return [
        RegisterDef(name="R0", size_bits=32, description="General purpose"),
        RegisterDef(name="R1", size_bits=32, description="General purpose"),
        RegisterDef(name="PC", aliases=["R15"], size_bits=32, description="Program counter"),
        RegisterDef(name="SP", aliases=["R13"], size_bits=32, description="Stack pointer"),
    ]


def _instructions() -> list[InstructionDef]:
    return [
        InstructionDef(
            mnemonic="ADD",
            variants=["ADD Rd, Rn, Rm"],
            encoding_bits=32,
            bit_fields={"opcode": "31:28", "Rd": "15:12"},
            operands=["Rd", "Rn", "Rm"],
            semantics="Add Rn and Rm, store result in Rd",
            pcode_hint="Rd = Rn + Rm;",
        ),
        InstructionDef(
            mnemonic="NOP",
            encoding_bits=32,
            semantics="No operation",
        ),
    ]


def _spec() -> ISASpec:
    return ISASpec(meta=_meta(), registers=_registers(), instructions=_instructions())


def _state_from_spec(spec: ISASpec, tmp_path, **kwargs) -> dict:
    base = {
        "meta": spec.meta.model_dump(),
        "registers": [r.model_dump() for r in spec.registers],
        "instructions": [i.model_dump() for i in spec.instructions],
        "processor_name": "TestISA",
        "out_dir": str(tmp_path),
        "errors": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# ModuleGenerator unit tests
# ---------------------------------------------------------------------------

def test_module_generator_creates_all_files(tmp_path):
    generator = ModuleGenerator()
    spec = _spec()
    lang_dir = generator.generate(spec, "TestISA", tmp_path)

    assert (lang_dir / "TestISA.slaspec").exists()
    assert (lang_dir / "TestISA.pspec").exists()
    assert (lang_dir / "TestISA.cspec").exists()
    assert (lang_dir / "TestISA.ldefs").exists()


def test_module_generator_slaspec_contains_mnemonics(tmp_path):
    generator = ModuleGenerator()
    lang_dir = generator.generate(_spec(), "TestISA", tmp_path)
    content = (lang_dir / "TestISA.slaspec").read_text()
    assert "ADD" in content
    assert "NOP" in content


def test_module_generator_slaspec_contains_registers(tmp_path):
    generator = ModuleGenerator()
    lang_dir = generator.generate(_spec(), "TestISA", tmp_path)
    content = (lang_dir / "TestISA.slaspec").read_text()
    assert "R0" in content
    assert "R1" in content


def test_module_generator_ldefs_contains_processor_name(tmp_path):
    generator = ModuleGenerator()
    lang_dir = generator.generate(_spec(), "TestISA", tmp_path)
    content = (lang_dir / "TestISA.ldefs").read_text()
    assert "TestISA" in content


def test_module_generator_pspec_references_pc_sp(tmp_path):
    generator = ModuleGenerator()
    lang_dir = generator.generate(_spec(), "TestISA", tmp_path)
    content = (lang_dir / "TestISA.pspec").read_text()
    assert "PC" in content or "R15" in content


def test_module_generator_endian_in_slaspec(tmp_path):
    generator = ModuleGenerator()
    lang_dir = generator.generate(_spec(), "TestISA", tmp_path)
    content = (lang_dir / "TestISA.slaspec").read_text()
    assert "little" in content.lower() or "LITTLE_ENDIAN" in content


# ---------------------------------------------------------------------------
# generate_sla_node tests
# ---------------------------------------------------------------------------

def test_generate_sla_node_success(tmp_path):
    spec = _spec()
    state = _state_from_spec(spec, tmp_path)
    result = generate_sla_node(state)

    assert result["errors"] == []
    assert result["lang_dir"] is not None
    lang_dir_path = __import__("pathlib").Path(result["lang_dir"])
    assert (lang_dir_path / "TestISA.slaspec").exists()


def test_generate_sla_node_no_meta_returns_error(tmp_path):
    state = {
        "meta": None,
        "registers": [],
        "instructions": [],
        "processor_name": "X",
        "out_dir": str(tmp_path),
        "errors": [],
    }
    result = generate_sla_node(state)
    assert result["lang_dir"] is None
    assert any("no ISAMeta" in e for e in result["errors"])


def test_generate_sla_node_returns_no_new_errors_on_success(tmp_path):
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    state = _state_from_spec(_spec(), tmp_path, errors=["prior"])
    result = generate_sla_node(state)
    assert result["errors"] == []
    assert result["lang_dir"] is not None
