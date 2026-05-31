"""Tests for rosetta_instructions.node — no Ollama required."""

from unittest.mock import patch

from rosetta_schemas.models import InstructionDef

from rosetta_instructions.node import instructions_node

# Both are deferred imports; patch at their source modules.
_PATCH_EXTRACT = "rosetta_instructions.extractor.extract_instruction_async"
_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_wrapper"


def _instr(mnemonic: str, **kwargs) -> InstructionDef:
    return InstructionDef(mnemonic=mnemonic, encoding_bits=32, semantics=f"{mnemonic} semantics", **kwargs)


def _state(**kwargs):
    base = {
        "db_path": "/tmp/db",
        "settings_dict": {},
        "mnemonics": ["ADD", "SUB", "MOV"],
        "max_concurrent": 2,
        "errors": [],
    }
    base.update(kwargs)
    return base


def test_instructions_node_basic():
    with patch(_PATCH_EXTRACT) as mock_extract, patch(_PATCH_CHROMA):
        async def fake(mnemonic, settings, semaphore, executor):
            return _instr(mnemonic)

        mock_extract.side_effect = fake
        result = instructions_node(_state())

    assert result["errors"] == []
    assert len(result["instructions"]) == 3
    names = {i["mnemonic"] for i in result["instructions"]}
    assert names == {"ADD", "SUB", "MOV"}


def test_instructions_node_serializes_to_dicts():
    with patch(_PATCH_EXTRACT) as mock, patch(_PATCH_CHROMA):
        async def fake(mnemonic, settings, semaphore, executor):
            return _instr(mnemonic)

        mock.side_effect = fake
        result = instructions_node(_state())

    for instr in result["instructions"]:
        assert isinstance(instr, dict)
        assert "mnemonic" in instr
        assert "encoding_bits" in instr


def test_instructions_node_stop_after_mnemonics():
    result = instructions_node(_state(stop_after="mnemonics"))
    assert result["instructions"] == []
    assert result["errors"] == []


def test_instructions_node_stop_after_meta():
    result = instructions_node(_state(stop_after="meta"))
    assert result["instructions"] == []


def test_instructions_node_stop_after_stubs():
    result = instructions_node(_state(stop_after="stubs"))
    assert result["errors"] == []
    assert len(result["instructions"]) == 3
    mnemonics = {i["mnemonic"] for i in result["instructions"]}
    assert mnemonics == {"ADD", "SUB", "MOV"}
    for instr in result["instructions"]:
        assert "stub" in instr["semantics"].lower()


def test_instructions_node_max_instructions_cap():
    called_with = []

    with patch(_PATCH_EXTRACT) as mock, patch(_PATCH_CHROMA):
        async def fake(mnemonic, settings, semaphore, executor):
            called_with.append(mnemonic)
            return _instr(mnemonic)

        mock.side_effect = fake
        result = instructions_node(_state(
            mnemonics=["ADD", "SUB", "MOV", "MUL", "DIV"],
            max_instructions=2,
        ))

    assert len(called_with) == 2
    assert len(result["instructions"]) == 2


def test_instructions_node_sets_db_path():
    captured = {}

    with patch(_PATCH_EXTRACT) as mock, patch(_PATCH_CHROMA):
        async def fake(mnemonic, settings, semaphore, executor):
            captured["db_path"] = settings.db_path
            return _instr(mnemonic)

        mock.side_effect = fake
        instructions_node(_state(db_path="/special/db", mnemonics=["ADD"]))

    assert captured.get("db_path") == "/special/db"


def test_instructions_node_empty_mnemonics():
    with patch(_PATCH_CHROMA):
        result = instructions_node(_state(mnemonics=[]))

    assert result["instructions"] == []
    assert result["errors"] == []


def test_instructions_node_handles_exception():
    with patch(_PATCH_EXTRACT, side_effect=Exception("boom")), patch(_PATCH_CHROMA):
        result = instructions_node(_state())

    assert any("instructions_node" in e for e in result["errors"])
