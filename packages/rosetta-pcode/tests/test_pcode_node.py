"""Tests for rosetta_pcode.node — no Ollama required."""

from unittest.mock import patch

from rosetta_schemas.models import InstructionDef

from rosetta_pcode.node import pcode_node


def _instr(mnemonic: str, pcode_hint: str = "") -> dict:
    return InstructionDef(
        mnemonic=mnemonic,
        encoding_bits=32,
        semantics=f"{mnemonic} semantics",
        pcode_hint=pcode_hint,
    ).model_dump()


def _state(**kwargs):
    base = {
        "instructions": [_instr("ADD"), _instr("SUB"), _instr("MOV")],
        "settings_dict": {},
        "errors": [],
    }
    base.update(kwargs)
    return base


def test_pcode_node_generates_hints():
    def fake_generate(instr, settings):
        return f"{instr.mnemonic}_pcode"

    with patch("rosetta_pcode.generator.generate_pcode", side_effect=fake_generate):
        result = pcode_node(_state())

    assert result["errors"] == []
    hints = {i["mnemonic"]: i["pcode_hint"] for i in result["instructions"]}
    assert hints["ADD"] == "ADD_pcode"
    assert hints["SUB"] == "SUB_pcode"
    assert hints["MOV"] == "MOV_pcode"


def test_pcode_node_skips_existing_hints():
    state = _state(instructions=[
        _instr("ADD", pcode_hint="existing"),
        _instr("SUB"),
    ])
    called_for = []

    def fake_generate(instr, settings):
        called_for.append(instr.mnemonic)
        return f"{instr.mnemonic}_new"

    with patch("rosetta_pcode.generator.generate_pcode", side_effect=fake_generate):
        result = pcode_node(state)

    assert "ADD" not in called_for
    assert "SUB" in called_for
    hints = {i["mnemonic"]: i["pcode_hint"] for i in result["instructions"]}
    assert hints["ADD"] == "existing"
    assert hints["SUB"] == "SUB_new"


def test_pcode_node_max_pcode_limit():
    called_for = []

    def fake_generate(instr, settings):
        called_for.append(instr.mnemonic)
        return "hint"

    with patch("rosetta_pcode.generator.generate_pcode", side_effect=fake_generate):
        result = pcode_node(_state(max_pcode=2))

    assert len(called_for) == 2
    assert len(result["instructions"]) == 3  # all instructions returned, only 2 have new hints


def test_pcode_node_empty_instructions():
    result = pcode_node(_state(instructions=[]))
    assert result["instructions"] == []
    assert result["errors"] == []


def test_pcode_node_handles_exception():
    with patch("rosetta_pcode.generator.generate_pcode", side_effect=RuntimeError("LLM down")):
        result = pcode_node(_state())

    assert any("pcode_node" in e for e in result["errors"])
    assert len(result["instructions"]) == 3


def test_pcode_node_returns_no_new_errors_on_success():
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    with patch("rosetta_pcode.generator.generate_pcode", return_value="hint"):
        result = pcode_node(_state(errors=["prior"]))

    assert result["errors"] == []
