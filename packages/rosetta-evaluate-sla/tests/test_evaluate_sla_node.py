"""Tests for rosetta_evaluate_sla.node and similarity — no Ollama required."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from rosetta_evaluate_sla.node import evaluate_sla_node
from rosetta_evaluate_sla.sla.similarity import SimilarityReport, compare, _cosine, _chunk_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kwargs):
    base = {
        "lang_dir": None,
        "reference_slaspec": None,
        "settings_dict": {},
        "errors": [],
    }
    base.update(kwargs)
    return base


def _report(**kwargs):
    defaults = dict(
        semantic_similarity=0.85,
        instruction_coverage=0.60,
        register_overlap=0.70,
        generated_mnemonic_count=10,
        reference_mnemonic_count=15,
        common_mnemonics=9,
        generated_register_count=8,
        reference_register_count=10,
        common_registers=7,
    )
    defaults.update(kwargs)
    return SimilarityReport(**defaults)


# ---------------------------------------------------------------------------
# evaluate_sla_node
# ---------------------------------------------------------------------------

def test_evaluate_sla_node_missing_lang_dir():
    result = evaluate_sla_node(_state(reference_slaspec="/ref.slaspec"))
    assert result["semantic_similarity"] is None
    assert any("required" in e for e in result["errors"])


def test_evaluate_sla_node_missing_reference():
    result = evaluate_sla_node(_state(lang_dir="/some/dir"))
    assert result["semantic_similarity"] is None
    assert any("required" in e for e in result["errors"])


def test_evaluate_sla_node_no_slaspec_in_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = evaluate_sla_node(_state(lang_dir=str(empty), reference_slaspec="/ref.slaspec"))
    assert result["semantic_similarity"] is None
    assert any("no .slaspec" in e for e in result["errors"])


def test_evaluate_sla_node_success(tmp_path):
    lang_dir = tmp_path / "languages"
    lang_dir.mkdir()
    (lang_dir / "test.slaspec").write_text("# generated slaspec")

    fake_report = _report()

    with patch("rosetta_evaluate_sla.sla.similarity.compare", return_value=fake_report):
        result = evaluate_sla_node(_state(
            lang_dir=str(lang_dir),
            reference_slaspec="/ref.slaspec",
        ))

    assert result["errors"] == []
    assert result["semantic_similarity"] == pytest.approx(0.85)
    assert result["instruction_coverage"] == pytest.approx(0.60)
    assert result["register_overlap"] == pytest.approx(0.70)


def test_evaluate_sla_node_compare_exception(tmp_path):
    lang_dir = tmp_path / "languages"
    lang_dir.mkdir()
    (lang_dir / "test.slaspec").write_text("# fake")

    with patch("rosetta_evaluate_sla.sla.similarity.compare", side_effect=RuntimeError("embedding failed")):
        result = evaluate_sla_node(_state(
            lang_dir=str(lang_dir),
            reference_slaspec="/ref.slaspec",
        ))

    assert result["semantic_similarity"] is None
    assert any("embedding failed" in e for e in result["errors"])


def test_evaluate_sla_node_returns_no_new_errors_on_success(tmp_path):
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    lang_dir = tmp_path / "languages"
    lang_dir.mkdir()
    (lang_dir / "test.slaspec").write_text("# fake")

    with patch("rosetta_evaluate_sla.sla.similarity.compare", return_value=_report()):
        result = evaluate_sla_node(_state(
            lang_dir=str(lang_dir),
            reference_slaspec="/ref.slaspec",
            errors=["prior"],
        ))

    assert result["errors"] == []


# ---------------------------------------------------------------------------
# SimilarityReport.summary()
# ---------------------------------------------------------------------------

def test_similarity_report_summary():
    report = _report(
        semantic_similarity=0.752,
        instruction_coverage=0.600,
        register_overlap=0.800,
        common_mnemonics=9,
        reference_mnemonic_count=15,
        common_registers=8,
        reference_register_count=10,
    )
    summary = report.summary()
    assert "0.752" in summary
    assert "0.600" in summary
    assert "9/15" in summary
    assert "8/10" in summary


# ---------------------------------------------------------------------------
# _cosine and _chunk_text (pure functions)
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)


def test_chunk_text_single_chunk():
    text = "short text"
    chunks = _chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_chunks():
    text = "A" * 3000
    chunks = _chunk_text(text, chunk_size=1000)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1000


# ---------------------------------------------------------------------------
# Instruction coverage formula
# ---------------------------------------------------------------------------

def test_instruction_coverage_formula():
    """Coverage = |gen ∩ ref| / |ref|"""
    from rosetta_evaluate_sla.sla.spec_loader import extract_mnemonics

    gen_text = ":ADD\n:SUB\n:MOV\n:NOP\n"
    ref_text = ":ADD\n:SUB\n:MOV\n:MUL\n:DIV\n"

    gen_mnemonics = extract_mnemonics(gen_text)
    ref_mnemonics = extract_mnemonics(ref_text)

    common = gen_mnemonics & ref_mnemonics
    coverage = len(common) / len(ref_mnemonics) if ref_mnemonics else 0.0

    assert coverage == pytest.approx(3 / 5)
