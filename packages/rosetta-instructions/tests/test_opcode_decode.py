"""Tests for the opcode_table decode strategy — no Ollama required."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rosetta_schemas.models import OpcodeDef

from rosetta_instructions.opcode_decode import _opcode_space, run_opcode_scan

_PATCH_GATHER = "rosetta_instructions.opcode_cursor.gather_opcode_def"


def _meta(prefixes=None):
    return SimpleNamespace(opcode_prefixes=prefixes or [])


# ---------------------------------------------------------------------------
# _opcode_space
# ---------------------------------------------------------------------------

def test_opcode_space_default_single_table():
    space = _opcode_space(_meta(), set())
    assert len(space) == 256
    assert space[0] == (None, 0x00)
    assert space[-1] == (None, 0xFF)


def test_opcode_space_with_prefixes_is_ordered():
    space = _opcode_space(_meta([0x89, 0x42]), set())
    assert len(space) == 512
    assert space[0] == (0x89, 0x00)
    assert space[256] == (0x42, 0x00)


def test_opcode_space_skips_seen():
    space = _opcode_space(_meta(), {(None, 0x00), (None, 0x01)})
    assert (None, 0x00) not in space
    assert len(space) == 254


# ---------------------------------------------------------------------------
# run_opcode_scan
# ---------------------------------------------------------------------------

def _fake_gather(opcode_byte, prefix, settings):
    return OpcodeDef(opcode=opcode_byte, prefix=prefix, mnemonic="LDA", mode="imm", operand_bytes=1)


def test_run_opcode_scan_respects_max_iterations():
    writer = MagicMock()
    with patch(_PATCH_GATHER, side_effect=_fake_gather) as gather:
        opcode_map, errors = run_opcode_scan(writer, settings=None, meta=_meta(),
                                             max_iterations=3)
    assert gather.call_count == 3
    assert len(opcode_map) == 3
    assert errors == []
    # rows streamed to the writer's opcode-table seam
    writer.write_opcode_table.assert_called_once()
    assert len(writer.write_opcode_table.call_args.args[0]) == 3


def test_run_opcode_scan_returns_serialized_rows():
    writer = MagicMock()
    with patch(_PATCH_GATHER, side_effect=_fake_gather):
        opcode_map, _ = run_opcode_scan(writer, settings=None, meta=_meta(), max_iterations=2)
    assert opcode_map[0]["opcode"] == 0x00
    assert opcode_map[0]["mnemonic"] == "LDA"


def test_run_opcode_scan_iterates_prefixes():
    writer = MagicMock()
    with patch(_PATCH_GATHER, side_effect=_fake_gather) as gather:
        run_opcode_scan(writer, settings=None, meta=_meta([0x89]), max_iterations=4)
    # all four calls carry the prefix
    assert all(call.args[1] == 0x89 for call in gather.call_args_list)


def test_run_opcode_scan_collects_errors_but_continues():
    writer = MagicMock()

    def flaky(opcode_byte, prefix, settings):
        if opcode_byte == 0x01:
            raise RuntimeError("boom")
        return _fake_gather(opcode_byte, prefix, settings)

    with patch(_PATCH_GATHER, side_effect=flaky):
        opcode_map, errors = run_opcode_scan(writer, settings=None, meta=_meta(), max_iterations=3)
    assert len(opcode_map) == 2          # 0x00 and 0x02 succeeded
    assert any("0x01" in e for e in errors)
