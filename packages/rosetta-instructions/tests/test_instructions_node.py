"""Backward-compat tests: instructions_node is an alias for decode_node.

The main coverage is in test_decode_node.py.  These tests verify the alias
exists and basic state contracts still hold.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rosetta_schemas.models import InstructionDef, ISAMeta, RegisterDef

from rosetta_instructions.node import instructions_node

_PATCH_CHROMA = "rosetta_utils.chroma.get_chroma_wrapper"
_PATCH_GRAPH = "rosetta_instructions.decode_graph.build_decode_graph"
_PATCH_WRITER = "rosetta_generate_sla.writers.base.get_writer"


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
        RegisterDef(name="PC", size_bits=32, description="program counter"),
        RegisterDef(name="SP", size_bits=32, description="stack pointer"),
    ]


def _make_fake_graph(written: list[str], lang_dir: Path):
    def fake_build(writer):
        app = MagicMock()
        app.invoke.return_value = {
            "written": written,
            "opcode_map_rows": [],
            "errors": [],
            "seen": written,
        }
        return app
    return fake_build


def _state(tmp_path: Path, **kwargs) -> dict:
    base = {
        "db_path": "/tmp/db",
        "settings_dict": {},
        "meta": _meta().model_dump(),
        "registers": [r.model_dump() for r in _registers()],
        "processor_name": "TestISA",
        "out_dir": str(tmp_path),
        "output_format": "sla",
        "errors": [],
    }
    base.update(kwargs)
    return base


def test_instructions_node_basic(tmp_path):
    """instructions_node alias: basic invocation returns instructions and lang_dir."""
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph(["ADD", "SUB"], lang_dir)),
    ):
        result = instructions_node(_state(tmp_path))

    assert result["errors"] == []
    assert result["lang_dir"] == str(lang_dir)


def test_instructions_node_serializes_to_dicts(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph([], lang_dir)),
    ):
        result = instructions_node(_state(tmp_path))

    assert isinstance(result["instructions"], list)


def test_instructions_node_stop_after_mnemonics(tmp_path):
    # stop_after is not used by decode_node; it completes normally.
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph([], lang_dir)),
    ):
        result = instructions_node(_state(tmp_path, stop_after="mnemonics"))

    assert isinstance(result["instructions"], list)


def test_instructions_node_stop_after_stubs(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph([], lang_dir)),
    ):
        result = instructions_node(_state(tmp_path, stop_after="stubs"))

    assert isinstance(result["instructions"], list)


def test_instructions_node_max_instructions_cap(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph(["ADD"], lang_dir)),
    ):
        result = instructions_node(_state(tmp_path, max_instructions=1))

    assert result["errors"] == []


def test_instructions_node_sets_db_path(tmp_path):
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    captured = {}

    def capture_chroma(db_path, settings):
        captured["db_path"] = db_path
        return MagicMock()

    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA, side_effect=capture_chroma),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph([], lang_dir)),
    ):
        instructions_node(_state(tmp_path, db_path="/special/db"))

    assert captured.get("db_path") == "/special/db"


def test_instructions_node_empty_mnemonics(tmp_path):
    # decode_node doesn't use mnemonics — it discovers them; empty mnemonics is fine.
    lang_dir = tmp_path / "TestISA" / "data" / "languages"
    lang_dir.mkdir(parents=True)
    fake_writer = MagicMock()
    fake_writer.lang_dir = lang_dir

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=_make_fake_graph([], lang_dir)),
    ):
        result = instructions_node(_state(tmp_path, mnemonics=[]))

    assert isinstance(result["instructions"], list)
    assert result["errors"] == []


def test_instructions_node_handles_exception(tmp_path):
    def boom(writer):
        raise RuntimeError("boom")

    fake_writer = MagicMock()
    fake_writer.lang_dir = None

    with (
        patch(_PATCH_CHROMA),
        patch(_PATCH_WRITER, return_value=fake_writer),
        patch(_PATCH_GRAPH, side_effect=boom),
    ):
        result = instructions_node(_state(tmp_path))

    assert any("subgraph" in e or "decode_node" in e for e in result["errors"])
