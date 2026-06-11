"""Tests for tag-based instruction discovery (docquery entity tags) + fallback.

Covers the "can we start decode for every instruction" invariant without any
live manual or Ollama: a fake Chroma collection returns entity-tagged metadata,
and we assert scan_db_for_mnemonics enumerates exactly those tagged mnemonics.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from docquery.config import ENTITY_PREFIX

from rosetta_instructions.discovery import (
    INSTRUCTION_ENTITY,
    read_tagged_mnemonics,
    scan_db_for_mnemonics,
)

_KEY = f"{ENTITY_PREFIX}{INSTRUCTION_ENTITY}"


def _settings(metadatas=None, documents=None):
    """Build a settings stub whose .vs._collection mimics a Chroma collection."""
    collection = MagicMock()
    collection.get.return_value = {
        "metadatas": metadatas if metadatas is not None else [],
        "documents": documents if documents is not None else [],
    }
    collection.count.return_value = len(documents or metadatas or [])
    return SimpleNamespace(vs=SimpleNamespace(_collection=collection))


# ---------------------------------------------------------------------------
# Tag mode
# ---------------------------------------------------------------------------

def test_read_tagged_mnemonics_distinct_and_page_ordered():
    metas = [
        {_KEY: "SUB", "page": 5},
        {_KEY: "ADD;AND", "page": 2},   # multi-entity chunk
        {"page": 3},                    # untagged chunk
        {_KEY: "ADD", "page": 9},       # duplicate of ADD → ignored
    ]
    out = read_tagged_mnemonics(_settings(metadatas=metas))
    # page-ordered (2 before 5), distinct, first-seen order within a chunk
    assert out == ["ADD", "AND", "SUB"]


def test_read_tagged_mnemonics_none_when_untagged():
    metas = [{"page": 1}, {"page": 2}]
    assert read_tagged_mnemonics(_settings(metadatas=metas)) is None


def test_scan_prefers_tags_over_frequency():
    metas = [{_KEY: "ADD", "page": 1}, {_KEY: "B;BL", "page": 2}]
    out = scan_db_for_mnemonics(_settings(metadatas=metas))
    assert out == ["ADD", "B", "BL"]


def test_scan_tag_mode_with_reference_filter():
    metas = [{_KEY: "ADD;BOGUS;SUB", "page": 1}]
    out = scan_db_for_mnemonics(
        _settings(metadatas=metas), reference_filter={"ADD", "SUB", "MUL"}
    )
    # intersection of tagged ∩ reference, sorted; BOGUS dropped, MUL absent
    assert out == ["ADD", "SUB"]


# ---------------------------------------------------------------------------
# Frequency fallback (no tags present)
# ---------------------------------------------------------------------------

def test_scan_falls_back_to_frequency_when_untagged():
    docs = ["ADD ADD ADD do things", "SUB SUB SUB and ADD again"]
    # metadatas carry no entity tags → fallback path
    out = scan_db_for_mnemonics(
        _settings(metadatas=[{}, {}], documents=docs), min_freq=3
    )
    assert "ADD" in out and "SUB" in out


# ---------------------------------------------------------------------------
# Decode-start invariant: queue covers every tagged instruction
# ---------------------------------------------------------------------------

def test_decode_starts_for_every_tagged_instruction(tmp_path):
    """Full gate: tagged DB → decode subgraph visits/writes every tagged mnemonic,
    terminating on queue-exhaustion (not the stall / max_iterations guards)."""
    from unittest.mock import patch

    from rosetta_schemas.models import InstructionDef, ISAMeta
    from rosetta_instructions.decode_graph import DecodeState, build_decode_graph

    tagged = ["ADD", "SUB", "AND", "ORR", "EOR"]
    metas = [{_KEY: ";".join(tagged), "page": 1}]
    settings = _settings(metadatas=metas)

    # Queue is built exactly the way decode_node does when no reference is given:
    # discover_node lazily calls scan_db_for_mnemonics on first iteration.
    queue = scan_db_for_mnemonics(settings)
    assert sorted(queue) == sorted(tagged)  # queued == tagged (decode can start for all)

    def fake_gather(current, next_, _settings):
        return InstructionDef(mnemonic=current, encoding_bits=32, semantics=f"{current}")

    writer = MagicMock()
    writer.lang_dir = tmp_path / "lang"
    writer.lang_dir.mkdir(parents=True)

    meta = ISAMeta(
        name="TestISA", endian="little", word_size_bits=32, alignment=4,
        instruction_sizes_bits=[32], encoding_style="fixed_word",
    ).model_dump()

    with (
        patch("rosetta_instructions.gather.gather_instruction", side_effect=fake_gather),
        patch("rosetta_instructions.gather.enrich_pcode", side_effect=lambda i, s: i),
    ):
        app = build_decode_graph(writer)
        initial: DecodeState = {
            "settings": settings,
            "meta": meta,
            "registers": [],
            "out_dir": str(tmp_path),
            "processor_name": "TestISA",
            "max_iterations": 1000,  # generous → loop must end on exhaustion, not this
            "inter_chunk_sleep": 0.0,
            "debug_save_dir": None,
            "resume": False,
            "last": None,
            "seen": [],
            "current": None,
            "next": None,
            "iterations": 0,
            "stall_count": 0,
            "mnemonic_queue": list(queue),
            "current_def": None,
            "written": [],
            "errors": [],
        }
        final = app.invoke(initial)

    # written ⊇ queued: every tagged instruction was emitted
    assert set(final["written"]) >= set(tagged)
    # terminated by exhaustion: nowhere near the iteration cap
    assert final["iterations"] < 1000
