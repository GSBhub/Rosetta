"""Tests for rosetta_registers.node (cursor-driven) — no Ollama required."""

from unittest.mock import MagicMock, patch

from rosetta_schemas.models import RegisterDef

from rosetta_registers.node import registers_node

_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_wrapper"
_PATCH_GRAPH  = "rosetta_registers.register_graph.build_register_graph"


def _state(**kwargs):
    base = {"db_path": "/tmp/db", "settings_dict": {}, "errors": []}
    base.update(kwargs)
    return base


def _reg(name: str, size: int = 32) -> RegisterDef:
    return RegisterDef(name=name, size_bits=size, description=f"{name} register")


def _make_fake_graph(registers: list[RegisterDef]):
    """Return a build_register_graph factory whose app returns the given registers."""
    def fake_build():
        app = MagicMock()
        app.invoke.return_value = {
            "registers": [r.model_dump() for r in registers],
            "errors": [],
        }
        return app
    return fake_build


def test_registers_node_success():
    regs = [_reg("R0"), _reg("PC"), _reg("SP")]
    with patch(_PATCH_CHROMA), patch(_PATCH_GRAPH, side_effect=_make_fake_graph(regs)):
        result = registers_node(_state())

    assert result["errors"] == []
    assert len(result["registers"]) == 3
    assert {r["name"] for r in result["registers"]} == {"R0", "PC", "SP"}


def test_registers_node_serializes_to_dicts():
    regs = [_reg("A"), _reg("X"), _reg("Y")]
    with patch(_PATCH_CHROMA), patch(_PATCH_GRAPH, side_effect=_make_fake_graph(regs)):
        result = registers_node(_state())

    for reg in result["registers"]:
        assert isinstance(reg, dict)
        assert "name" in reg
        assert "size_bits" in reg


def test_registers_node_empty_result():
    with patch(_PATCH_CHROMA), patch(_PATCH_GRAPH, side_effect=_make_fake_graph([])):
        result = registers_node(_state())

    assert result["registers"] == []
    assert result["errors"] == []


def test_registers_node_sets_db_path():
    captured = {}

    def fake_chroma(db_path, settings):
        captured["db_path"] = db_path
        return MagicMock()

    with patch(_PATCH_CHROMA, side_effect=fake_chroma), \
         patch(_PATCH_GRAPH, side_effect=_make_fake_graph([])):
        registers_node(_state(db_path="/special/db"))

    assert captured["db_path"] == "/special/db"


def test_registers_node_graph_exception_returns_error():
    def boom():
        raise RuntimeError("graph exploded")

    with patch(_PATCH_CHROMA), patch(_PATCH_GRAPH, side_effect=boom):
        result = registers_node(_state())

    assert result["registers"] == []
    assert any("registers_node" in e for e in result["errors"])


def test_registers_node_chroma_failure_returns_error():
    with patch(_PATCH_CHROMA, side_effect=RuntimeError("no chroma")):
        result = registers_node(_state())

    assert result["registers"] == []
    assert result["errors"]


def test_registers_node_returns_no_new_errors_on_success():
    regs = [_reg("PC")]
    with patch(_PATCH_CHROMA), patch(_PATCH_GRAPH, side_effect=_make_fake_graph(regs)):
        result = registers_node(_state(errors=["upstream error"]))

    assert result["errors"] == []
    assert len(result["registers"]) == 1
