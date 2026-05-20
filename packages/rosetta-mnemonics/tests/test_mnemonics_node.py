"""Tests for rosetta_mnemonics.node — no Ollama required."""

from unittest.mock import MagicMock, patch

from rosetta_mnemonics.node import mnemonics_node

# discover_mnemonics is imported inside mnemonics_node; patch at its source module.
_PATCH = "rosetta_mnemonics.discovery.discover_mnemonics"
_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_collection"


def _state(**kwargs):
    base = {"db_path": "/tmp/db", "settings_dict": {}, "errors": []}
    base.update(kwargs)
    return base


def test_mnemonics_node_success():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, return_value=["ADD", "SUB", "MOV"]):
        result = mnemonics_node(_state())

    assert result["errors"] == []
    assert sorted(result["mnemonics"]) == ["ADD", "MOV", "SUB"]


def test_mnemonics_node_passes_db_path():
    captured = {}

    def fake_discover(db_path, settings, strategies=None):
        captured["db_path"] = db_path
        return []

    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, side_effect=fake_discover):
        mnemonics_node(_state(db_path="/my/db"))

    assert captured["db_path"] == "/my/db"


def test_mnemonics_node_filter_glob():
    all_mnemonics = ["ADD", "ADDC", "SUB", "MOV", "VADD", "VADD.F32"]
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, return_value=all_mnemonics):
        result = mnemonics_node(_state(filter_mnemonics="ADD*,MOV"))

    assert "ADD" in result["mnemonics"]
    assert "ADDC" in result["mnemonics"]
    assert "MOV" in result["mnemonics"]
    assert "SUB" not in result["mnemonics"]
    assert "VADD" not in result["mnemonics"]


def test_mnemonics_node_filter_multiple_patterns():
    all_mnemonics = ["ADD", "SUB", "MOV", "MUL", "NOP"]
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, return_value=all_mnemonics):
        result = mnemonics_node(_state(filter_mnemonics="ADD,MUL,NOP"))

    assert set(result["mnemonics"]) == {"ADD", "MUL", "NOP"}


def test_mnemonics_node_no_filter():
    all_mnemonics = ["ADD", "SUB", "MOV"]
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, return_value=all_mnemonics):
        result = mnemonics_node(_state())

    assert set(result["mnemonics"]) == {"ADD", "SUB", "MOV"}


def test_mnemonics_node_returns_empty_on_error():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, side_effect=RuntimeError("boom")):
        result = mnemonics_node(_state())

    assert result["mnemonics"] == []
    assert any("mnemonics_node" in e for e in result["errors"])


def test_mnemonics_node_returns_no_new_errors_on_success():
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch(_PATCH, return_value=["ADD"]):
        result = mnemonics_node(_state(errors=["prior"]))

    assert result["errors"] == []
