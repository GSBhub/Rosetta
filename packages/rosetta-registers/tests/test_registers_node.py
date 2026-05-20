"""Tests for rosetta_registers.node — no Ollama required."""

from unittest.mock import MagicMock, patch

from rosetta_schemas.models import RegisterDef

from rosetta_registers.node import registers_node, _RegisterList

_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_wrapper"


def _state(**kwargs):
    base = {"db_path": "/tmp/db", "settings_dict": {}, "errors": []}
    base.update(kwargs)
    return base


def _fake_register_list():
    return _RegisterList(registers=[
        RegisterDef(name="R0", size_bits=32, description="General purpose"),
        RegisterDef(name="PC", aliases=["R15"], size_bits=32, description="Program counter"),
        RegisterDef(name="SP", aliases=["R13"], size_bits=32, description="Stack pointer"),
    ])


def test_registers_node_success():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value=_fake_register_list()):
        result = registers_node(_state())

    assert result["errors"] == []
    assert len(result["registers"]) == 3
    names = {r["name"] for r in result["registers"]}
    assert names == {"R0", "PC", "SP"}


def test_registers_node_serializes_to_dicts():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value=_fake_register_list()):
        result = registers_node(_state())

    for reg in result["registers"]:
        assert isinstance(reg, dict)
        assert "name" in reg
        assert "size_bits" in reg


def test_registers_node_sets_db_path():
    captured = {}

    def fake_query(prompt, *, schema=None, system_prompt=None, settings=None):
        captured["db_path"] = settings.db_path if settings else None
        return _fake_register_list()

    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", side_effect=fake_query):
        registers_node(_state(db_path="/special/db"))

    assert captured["db_path"] == "/special/db"


def test_registers_node_returns_empty_on_error():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", side_effect=RuntimeError("timeout")):
        result = registers_node(_state())

    assert result["registers"] == []
    assert any("registers_node" in e for e in result["errors"])


def test_registers_node_unexpected_type_returns_empty():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value="not a list"):
        result = registers_node(_state())

    assert result["registers"] == []


def test_registers_node_returns_no_new_errors_on_success():
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value=_fake_register_list()):
        result = registers_node(_state(errors=["upstream error"]))

    assert result["errors"] == []
    assert len(result["registers"]) == 3
