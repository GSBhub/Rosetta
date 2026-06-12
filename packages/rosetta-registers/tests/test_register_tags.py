"""Register enumeration via docquery entity tags (no Ollama).

Mirrors the instruction decode-start gate: a tagged DB must let registers_node
enumerate every tagged register deterministically, with no LLM cursor calls.
"""

from unittest.mock import MagicMock, patch

from docquery.config import ENTITY_PREFIX
from rosetta_schemas.models import RegisterDef

from rosetta_registers.cursor import REGISTER_ENTITY
from rosetta_registers.node import registers_node

_KEY = f"{ENTITY_PREFIX}{REGISTER_ENTITY}"
_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_wrapper"
_PATCH_GATHER = "rosetta_registers.cursor.gather_register"
_PATCH_DISCOVER = "rosetta_registers.cursor.discover_next_register"


def _fake_vs(metadatas):
    collection = MagicMock()
    collection.get.return_value = {"metadatas": metadatas}
    return MagicMock(_collection=collection)


def _state(**kwargs):
    base = {"db_path": "/tmp/db", "settings_dict": {}, "max_iterations": 1000, "errors": []}
    base.update(kwargs)
    return base


def test_registers_enumerated_from_tags():
    tagged = ["R0", "R1", "SP", "PC", "CPSR"]
    metas = [{_KEY: ";".join(tagged), "page": 1}]

    def fake_gather(name, _settings):
        return RegisterDef(name=name, size_bits=32, description=f"{name} register")

    with (
        patch(_PATCH_CHROMA, return_value=_fake_vs(metas)),
        patch(_PATCH_GATHER, side_effect=fake_gather),
        patch(_PATCH_DISCOVER) as mock_discover,
    ):
        result = registers_node(_state())

    names = {r["name"] for r in result["registers"]}
    assert names == set(tagged)            # every tagged register enumerated
    assert result["errors"] == []
    mock_discover.assert_not_called()      # tag mode → no LLM cursor


def test_registers_fall_back_to_llm_when_untagged():
    metas = [{"page": 1}, {"page": 2}]  # no entity_register tags

    # LLM cursor yields one register then signals end.
    seq = [("R0", None), (None, None)]

    def fake_discover(last, seen, settings):
        return seq.pop(0) if seq else (None, None)

    def fake_gather(name, _settings):
        return RegisterDef(name=name, size_bits=32, description=name)

    with (
        patch(_PATCH_CHROMA, return_value=_fake_vs(metas)),
        patch(_PATCH_DISCOVER, side_effect=fake_discover),
        patch(_PATCH_GATHER, side_effect=fake_gather),
    ):
        result = registers_node(_state())

    assert {r["name"] for r in result["registers"]} == {"R0"}
