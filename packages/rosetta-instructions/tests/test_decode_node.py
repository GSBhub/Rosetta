"""Tests for decode_node (reworked rosetta_instructions.node) — no Ollama required."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rosetta_schemas.models import InstructionDef, ISAMeta, RegisterDef

from rosetta_instructions.node import decode_node

# Patch targets — all lazy imports, patch at source module
_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_wrapper"
_PATCH_GRAPH = "rosetta_instructions.decode_graph.build_decode_graph"
_PATCH_WRITER = "rosetta_generate_sla.writers.base.get_writer"


def _meta(encoding_style: str = "fixed_word") -> ISAMeta:
    return ISAMeta(
        name="TestISA",
        endian="little",
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[32],
        encoding_style=encoding_style,
    )


def _registers() -> list[RegisterDef]:
    return [
        RegisterDef(name="PC", size_bits=32, description="program counter"),
        RegisterDef(name="SP", size_bits=32, description="stack pointer"),
    ]


def _state(tmp_path: Path, **kwargs) -> dict:
    base = {
        "db_path": "/tmp/db",
        "settings_dict": {},
        "meta": _meta().model_dump(),
        "registers": [r.model_dump() for r in _registers()],
        "processor_name": "TestISA",
        "out_dir": str(tmp_path),
        "output_format": "sla",
        "max_iterations": 5,
        "inter_chunk_sleep": 0.0,
        "resume": False,
        "debug_save_dir": None,
        "errors": [],
    }
    base.update(kwargs)
    return base


def _make_fake_graph(written_mnemonics: list[str], lang_dir: Path):
    """Return a mock build_decode_graph factory that returns a fake app.invoke."""
    def fake_build(writer):
        app = MagicMock()
        app.invoke.return_value = {
            "written": written_mnemonics,
            "opcode_map_rows": [],
            "errors": [],
            "seen": written_mnemonics,
        }
        return app
    return fake_build


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_decode_node_basic(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph(["ADD", "SUB"], lang_dir)),
    ):
        result = decode_node(_state(tmp_path))

    assert result["errors"] == []
    assert str(lang_dir) == result["lang_dir"]
    fake_writer.open.assert_called_once()
    fake_writer.close.assert_called_once()


def test_decode_node_lang_dir_set(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph(["NOP"], lang_dir)),
    ):
        result = decode_node(_state(tmp_path))

    assert result["lang_dir"] == str(lang_dir)


def test_decode_node_no_meta(tmp_path):
    state = _state(tmp_path)
    state.pop("meta")

    with patch(_PATCH_CHROMA):
        result = decode_node(state)

    assert result["lang_dir"] is None
    assert any("no meta" in e for e in result["errors"])


def test_decode_node_missing_db_path(tmp_path):
    state = _state(tmp_path)
    state.pop("db_path")

    result = decode_node(state)
    assert result["lang_dir"] is None
    assert result["errors"]


def test_decode_node_writer_open_failure(tmp_path):
    fake_writer = MagicMock()
    fake_writer.open.side_effect = RuntimeError("disk full")
    fake_writer.lang_dir = None

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
    ):
        result = decode_node(_state(tmp_path))

    assert result["lang_dir"] is None
    assert any("writer.open" in e for e in result["errors"])


def test_decode_node_subgraph_exception(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    def boom(writer):
        raise RuntimeError("graph exploded")

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=boom),
    ):
        result = decode_node(_state(tmp_path))

    assert any("subgraph" in e for e in result["errors"])
    fake_writer.close.assert_called_once()  # close still called


def test_decode_node_output_format_forwarded(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir
    captured = {}

    def capture_writer(fmt):
        captured["fmt"] = fmt
        return fake_writer

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, side_effect=capture_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph([], lang_dir)),
    ):
        decode_node(_state(tmp_path, output_format="sleigh"))

    assert captured["fmt"] == "sleigh"


_PATCH_OPCODE = "rosetta_instructions.opcode_decode.run_opcode_scan"


def test_decode_node_opcode_table_dispatches_to_opcode_scan(tmp_path):
    """opcode_table family routes to the opcode strategy, not the mnemonic graph."""
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir
    rows = [{"opcode": 0, "mnemonic": "LDA", "mode": "imm", "operand_bytes": 1}]

    state = _state(tmp_path, meta=_meta("opcode_table").model_dump())

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_OPCODE, return_value=(rows, [])) as scan,
        patch(_PATCH_GRAPH, side_effect=AssertionError("mnemonic graph must not run for CISC")),
    ):
        result = decode_node(state)

    scan.assert_called_once()
    assert result["opcode_map"] == rows
    assert result["errors"] == []
    assert result["lang_dir"] == str(lang_dir)
    fake_writer.close.assert_called_once()


def test_decode_node_variable_prefix_falls_back_to_mnemonic(tmp_path):
    """variable_prefix (x86) has no opcode-table strategy yet → mnemonic graph + a caveat."""
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    state = _state(tmp_path, meta=_meta("variable_prefix").model_dump())

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph(["MOV"], lang_dir)),
    ):
        result = decode_node(state)

    assert result["lang_dir"] == str(lang_dir)
    assert any("variable_prefix" in e for e in result["errors"])
