"""Node 8: SLEIGH similarity evaluation (SLA backend).

Reads from state:  lang_dir, reference_slaspec, settings_dict
Returns to state:  semantic_similarity, instruction_coverage, register_overlap, errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosetta_schemas.state import PipelineState

log = logging.getLogger(__name__)


def _find_slaspec(lang_dir: Path) -> Path | None:
    specs = list(lang_dir.glob("*.slaspec"))
    return specs[0] if specs else None


def evaluate_sla_node(state: PipelineState) -> dict[str, Any]:
    """Compute semantic + structural similarity against a reference .slaspec."""
    from docquery.config import Settings
    from rosetta_evaluate_sla.sla.similarity import compare

    errors: list[str] = []
    lang_dir_str = state.get("lang_dir")
    reference_str = state.get("reference_slaspec")

    if not lang_dir_str or not reference_str:
        errors.append("evaluate_sla_node: lang_dir and reference_slaspec are required")
        return {
            "semantic_similarity": None,
            "instruction_coverage": None,
            "register_overlap": None,
            "errors": errors,
        }

    generated = _find_slaspec(Path(lang_dir_str))
    if not generated:
        errors.append(f"evaluate_sla_node: no .slaspec found in {lang_dir_str}")
        return {
            "semantic_similarity": None,
            "instruction_coverage": None,
            "register_overlap": None,
            "errors": errors,
        }

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        report = compare(generated, Path(reference_str), settings)
        log.info("evaluate_sla_node: coverage=%.3f sem=%.3f", report.instruction_coverage, report.semantic_similarity)
        return {
            "semantic_similarity": report.semantic_similarity,
            "instruction_coverage": report.instruction_coverage,
            "register_overlap": report.register_overlap,
            "errors": errors,
        }
    except Exception as exc:
        log.exception("evaluate_sla_node failed")
        errors.append(f"evaluate_sla_node: {exc}")
        return {
            "semantic_similarity": None,
            "instruction_coverage": None,
            "register_overlap": None,
            "errors": errors,
        }
