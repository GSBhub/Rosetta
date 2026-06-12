"""Tests for rosetta_evaluate_sla.node and similarity — no Ollama required.

The evaluator is *structural*: it compares a generated .slaspec against a
reference on instruction coverage (|gen ∩ ref| / |ref|) and register overlap
(Jaccard). There is no embedding/semantic step.
"""

from unittest.mock import patch

import pytest

from rosetta_evaluate_sla.node import evaluate_sla_node
from rosetta_evaluate_sla.sla.similarity import SimilarityReport, compare


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
    assert result["instruction_coverage"] is None
    assert result["register_overlap"] is None
    assert any("required" in e for e in result["errors"])


def test_evaluate_sla_node_missing_reference():
    result = evaluate_sla_node(_state(lang_dir="/some/dir"))
    assert result["instruction_coverage"] is None
    assert any("required" in e for e in result["errors"])


def test_evaluate_sla_node_success(tmp_path):
    lang_dir = tmp_path / "languages"
    lang_dir.mkdir()
    (lang_dir / "test.slaspec").write_text("# generated slaspec")

    with patch("rosetta_evaluate_sla.sla.similarity.compare", return_value=_report()):
        result = evaluate_sla_node(_state(
            lang_dir=str(lang_dir),
            reference_slaspec="/ref.slaspec",
        ))

    assert result["errors"] == []
    assert result["instruction_coverage"] == pytest.approx(0.60)
    assert result["register_overlap"] == pytest.approx(0.70)


def test_evaluate_sla_node_compare_exception(tmp_path):
    lang_dir = tmp_path / "languages"
    lang_dir.mkdir()
    (lang_dir / "test.slaspec").write_text("# fake")

    with patch("rosetta_evaluate_sla.sla.similarity.compare", side_effect=RuntimeError("loader failed")):
        result = evaluate_sla_node(_state(
            lang_dir=str(lang_dir),
            reference_slaspec="/ref.slaspec",
        ))

    assert result["instruction_coverage"] is None
    assert any("loader failed" in e for e in result["errors"])


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
        instruction_coverage=0.600,
        register_overlap=0.800,
        common_mnemonics=9,
        reference_mnemonic_count=15,
        common_registers=8,
        reference_register_count=10,
    )
    summary = report.summary()
    assert "0.600" in summary
    assert "0.800" in summary
    assert "9/15" in summary
    assert "8/10" in summary


# ---------------------------------------------------------------------------
# compare() end-to-end on tiny slaspec files
# ---------------------------------------------------------------------------

def test_compare_structural(tmp_path):
    gen = tmp_path / "gen.slaspec"
    ref = tmp_path / "ref.slaspec"
    gen.write_text(":ADD\n:SUB\n:MOV\n:NOP\n")
    ref.write_text(":ADD\n:SUB\n:MOV\n:MUL\n:DIV\n")

    report = compare(gen, ref)
    # coverage = |{ADD,SUB,MOV}| / |{ADD,SUB,MOV,MUL,DIV}| = 3/5
    assert report.instruction_coverage == pytest.approx(3 / 5)
    assert report.common_mnemonics == 3
    assert report.reference_mnemonic_count == 5


# ---------------------------------------------------------------------------
# Instruction coverage formula (via spec_loader)
# ---------------------------------------------------------------------------

def test_instruction_coverage_formula():
    """Coverage = |gen ∩ ref| / |ref|"""
    from rosetta_evaluate_sla.sla.spec_loader import extract_mnemonics

    gen_mnemonics = extract_mnemonics(":ADD\n:SUB\n:MOV\n:NOP\n")
    ref_mnemonics = extract_mnemonics(":ADD\n:SUB\n:MOV\n:MUL\n:DIV\n")

    common = gen_mnemonics & ref_mnemonics
    coverage = len(common) / len(ref_mnemonics) if ref_mnemonics else 0.0
    assert coverage == pytest.approx(3 / 5)
