"""Unit tests for rosetta.stage_runner — no Ollama or Ghidra required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rosetta.stage_runner import (
    STAGE_ORDER,
    build_initial_state,
    check_prereqs,
    load_checkpoint,
    merge,
    run_stage,
    save_checkpoint,
    summarize_and_warn,
)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_overwrites_scalar_keys():
    state = {"meta": {"name": "old"}, "errors": []}
    partial = {"meta": {"name": "new"}, "errors": []}
    result = merge(state, partial)
    assert result["meta"]["name"] == "new"


def test_merge_extends_errors():
    state = {"errors": ["prior"]}
    partial = {"errors": ["new1", "new2"]}
    result = merge(state, partial)
    assert result["errors"] == ["prior", "new1", "new2"]


def test_merge_errors_handles_none_in_state():
    state = {"errors": None}
    partial = {"errors": ["e1"]}
    result = merge(state, partial)
    assert result["errors"] == ["e1"]


def test_merge_errors_handles_none_in_partial():
    state = {"errors": ["existing"]}
    partial = {"errors": None}
    result = merge(state, partial)
    assert result["errors"] == ["existing"]


def test_merge_does_not_mutate_original():
    state = {"errors": ["a"], "meta": {"name": "x"}}
    merge(state, {"errors": ["b"], "meta": {"name": "y"}})
    assert state["errors"] == ["a"]
    assert state["meta"]["name"] == "x"


def test_merge_adds_new_key():
    state = {"errors": []}
    partial = {"meta": {"name": "ARM"}, "errors": []}
    result = merge(state, partial)
    assert "meta" in result


# ---------------------------------------------------------------------------
# check_prereqs
# ---------------------------------------------------------------------------

def _minimal_state(**kwargs) -> dict:
    base = {"db_path": "dbs/test_chroma", "errors": []}
    base.update(kwargs)
    return base


def test_check_prereqs_meta_passes_with_db_path():
    check_prereqs("meta", _minimal_state())  # no exception


def test_check_prereqs_instructions_raises_if_no_mnemonics():
    state = _minimal_state()  # missing mnemonics
    with pytest.raises(ValueError, match="mnemonics"):
        check_prereqs("instructions", state)


def test_check_prereqs_instructions_raises_if_empty_mnemonics():
    state = _minimal_state(mnemonics=[])
    with pytest.raises(ValueError, match="mnemonics"):
        check_prereqs("instructions", state)


def test_check_prereqs_instructions_passes_with_mnemonics():
    state = _minimal_state(mnemonics=["ADD", "SUB"])
    check_prereqs("instructions", state)  # no exception


def test_check_prereqs_generate_raises_if_no_meta():
    state = _minimal_state(processor_name="ARM", out_dir="./out")
    with pytest.raises(ValueError, match="meta"):
        check_prereqs("generate", state)


def test_check_prereqs_validate_raises_if_no_lang_dir():
    state = _minimal_state(ghidra_home="/tools/ghidra")
    with pytest.raises(ValueError, match="lang_dir"):
        check_prereqs("validate", state)


def test_check_prereqs_evaluate_raises_if_no_reference():
    state = _minimal_state(lang_dir="./output/ARM/data/languages")
    with pytest.raises(ValueError, match="reference_slaspec"):
        check_prereqs("evaluate", state)


def test_check_prereqs_unknown_stage_not_called_directly():
    """check_prereqs only checks registry entries; unknown stages handled by run_stage."""
    pass


# ---------------------------------------------------------------------------
# checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    state = {"db_path": "dbs/x", "errors": ["e1"], "meta": {"name": "ARM"}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)
    assert loaded == state


def test_load_checkpoint_returns_empty_dict_if_missing():
    path = Path("/tmp/nonexistent_rosetta_checkpoint_xyz.json")
    assert load_checkpoint(path) == {}


# ---------------------------------------------------------------------------
# run_stage dispatches to node and merges
# ---------------------------------------------------------------------------

def _fake_meta_result(_state):
    return {
        "meta": {"name": "FakeISA", "endian": "little", "word_size_bits": 32,
                 "alignment": 4, "instruction_sizes_bits": [32]},
        "errors": [],
    }


def test_run_stage_meta_merges_result():
    state = _minimal_state()
    # _import_meta() does "from rosetta_meta.node import meta_node; return meta_node"
    # so patch at the source module so the lazy import picks it up.
    with patch("rosetta_meta.node.meta_node", side_effect=_fake_meta_result):
        result = run_stage(state, "meta")
    assert result["meta"]["name"] == "FakeISA"
    assert result["errors"] == []


def test_run_stage_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage({}, "nonexistent")


def test_run_stage_missing_prereq_raises():
    state = _minimal_state()  # no mnemonics
    with pytest.raises(ValueError, match="mnemonics"):
        run_stage(state, "instructions")


def test_run_stage_preserves_prior_errors():
    state = _minimal_state(errors=["prior_error"])

    def _node_ok(s):
        return {"meta": {"name": "X", "endian": "little", "word_size_bits": 32,
                         "alignment": 4, "instruction_sizes_bits": [32]},
                "errors": []}

    with patch("rosetta_meta.node.meta_node", side_effect=_node_ok):
        result = run_stage(state, "meta")
    assert "prior_error" in result["errors"]


def test_run_stage_extends_errors_from_node():
    state = _minimal_state()

    def _node_err(s):
        return {"meta": {"name": "Unknown", "endian": "little", "word_size_bits": 32,
                         "alignment": 4, "instruction_sizes_bits": [32]},
                "errors": ["meta_node: LLM down"]}

    with patch("rosetta_meta.node.meta_node", side_effect=_node_err):
        result = run_stage(state, "meta")
    assert any("LLM down" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# build_initial_state
# ---------------------------------------------------------------------------

def test_build_initial_state_singleton_concurrency():
    state = build_initial_state(
        db_path="dbs/x_chroma",
        processor_name="TestISA",
        out_dir="./output",
        settings_dict={},
        ghidra_home="/tools/ghidra",
        reference_slaspec=None,
        source_path=None,
        inter_chunk_sleep=2.0,
        max_instructions=None,
        max_pcode=None,
        memory_warn_gb=2.0,
    )
    assert state["max_concurrent"] == 1
    assert state["chunk_size"] == 1
    assert state["inter_chunk_sleep"] == 2.0


def test_build_initial_state_sets_ghidra_home():
    state = build_initial_state(
        db_path="dbs/x", processor_name="X", out_dir="./o",
        settings_dict={}, ghidra_home="/tools/ghidra",
        reference_slaspec=None, source_path=None,
        inter_chunk_sleep=0.0, max_instructions=None,
        max_pcode=None, memory_warn_gb=2.0,
    )
    assert state["ghidra_home"] == "/tools/ghidra"


def test_build_initial_state_includes_reference_and_source():
    state = build_initial_state(
        db_path="dbs/x", processor_name="X", out_dir="./o",
        settings_dict={}, ghidra_home="/g",
        reference_slaspec="/ref/ARM6_le.slaspec",
        source_path="/manuals/arm.pdf",
        inter_chunk_sleep=0.0, max_instructions=None,
        max_pcode=None, memory_warn_gb=2.0,
    )
    assert state["reference_slaspec"] == "/ref/ARM6_le.slaspec"
    assert state["source_path"] == "/manuals/arm.pdf"


def test_build_initial_state_omits_source_when_none():
    state = build_initial_state(
        db_path="dbs/x", processor_name="X", out_dir="./o",
        settings_dict={}, ghidra_home="/g",
        reference_slaspec=None, source_path=None,
        inter_chunk_sleep=0.0, max_instructions=None,
        max_pcode=None, memory_warn_gb=2.0,
    )
    assert "source_path" not in state


# ---------------------------------------------------------------------------
# stage order
# ---------------------------------------------------------------------------

def test_stage_order_covers_all_pipeline_nodes():
    expected = ["ingest", "meta", "registers", "mnemonics", "instructions",
                "pcode", "generate", "validate", "evaluate"]
    assert STAGE_ORDER == expected


# ---------------------------------------------------------------------------
# summarize_and_warn — spot checks (no exceptions, correct log level)
# ---------------------------------------------------------------------------

def test_summarize_meta_fallback_does_not_raise():
    state = {"meta": {"name": "Unknown"}, "errors": []}
    summarize_and_warn("meta", state)  # should not raise


def test_summarize_evaluate_does_not_raise():
    state = {
        "instruction_coverage": 0.85,
        "register_overlap": 0.9,
        "semantic_similarity": 0.7,
        "errors": [],
    }
    summarize_and_warn("evaluate", state)  # should not raise


def test_summarize_empty_instructions_does_not_raise():
    state = {"instructions": [], "mnemonics": ["ADD"], "errors": []}
    summarize_and_warn("instructions", state)
