"""Tests for rosetta_meta.node — no Ollama required."""

from unittest.mock import MagicMock, patch

from rosetta_schemas.models import ISAMeta

from rosetta_meta.node import meta_node

_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_collection"


def _state(**kwargs):
    base = {"db_path": "/tmp/db", "settings_dict": {}, "errors": []}
    base.update(kwargs)
    return base


def _fake_meta():
    return ISAMeta(
        name="TestISA",
        endian="little",
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[32],
    )


def test_meta_node_success():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value=_fake_meta()):
        result = meta_node(_state())

    assert result["errors"] == []
    assert result["meta"]["name"] == "TestISA"
    assert result["meta"]["endian"] == "little"


def test_meta_node_sets_db_path_on_settings():
    captured = {}

    def fake_query(prompt, *, schema=None, system_prompt=None, settings=None):
        captured["db_path"] = settings.db_path if settings else None
        return _fake_meta()

    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", side_effect=fake_query):
        meta_node(_state(db_path="/custom/db"))

    assert captured["db_path"] == "/custom/db"


def test_meta_node_returns_fallback_on_error():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", side_effect=RuntimeError("LLM down")):
        result = meta_node(_state())

    assert result["meta"]["name"] == "Unknown"
    assert any("meta_node" in e for e in result["errors"])


def test_meta_node_unexpected_return_type_uses_fallback():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value="not a model"):
        result = meta_node(_state())

    assert result["meta"]["name"] == "Unknown"


def test_meta_node_serializes_to_dict():
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value=_fake_meta()):
        result = meta_node(_state())

    assert isinstance(result["meta"], dict)
    assert "name" in result["meta"]
    assert "instruction_sizes_bits" in result["meta"]


def test_meta_node_returns_no_new_errors_on_success():
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    with patch(_PATCH_CHROMA, return_value=MagicMock()), \
         patch("docquery.query", return_value=_fake_meta()):
        result = meta_node(_state(errors=["prior"]))

    assert result["errors"] == []
