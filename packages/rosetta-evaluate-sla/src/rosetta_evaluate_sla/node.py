"""Node 8: SLEIGH structural evaluation (SLA backend).

Reads from state:  lang_dir, reference_slaspec
Returns to state:  instruction_coverage, register_overlap, errors
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
    """Compute structural similarity against a reference .slaspec."""
    from rosetta_evaluate_sla.sla.similarity import compare

    errors: list[str] = []
    lang_dir_str = state.get("lang_dir")
    reference_str = state.get("reference_slaspec")

    if not lang_dir_str or not reference_str:
        errors.append("evaluate_sla_node: lang_dir and reference_slaspec are required")
        return {"instruction_coverage": None, "register_overlap": None, "errors": errors}

    generated = _find_slaspec(Path(lang_dir_str))
    if not generated:
        errors.append(f"evaluate_sla_node: no .slaspec found in {lang_dir_str}")
        return {"instruction_coverage": None, "register_overlap": None, "errors": errors}

    try:
        report = compare(generated, Path(reference_str))
        log.info("evaluate_sla_node: coverage=%.3f reg=%.3f", report.instruction_coverage, report.register_overlap)
        return {
            "instruction_coverage": report.instruction_coverage,
            "register_overlap": report.register_overlap,
            "errors": errors,
        }
    except Exception as exc:
        log.exception("evaluate_sla_node failed")
        errors.append(f"evaluate_sla_node: {exc}")
        return {"instruction_coverage": None, "register_overlap": None, "errors": errors}
