"""Node 7: SLEIGH compilation validation (SLA backend).

Reads from state:  lang_dir, ghidra_home
Returns to state:  compile_ok, compile_errors, errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langsmith import traceable
from rosetta_schemas.state import PipelineState
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)


@traceable(run_type="chain", name="stage:validate_sla", process_inputs=state_summary)
def validate_sla_node(state: PipelineState) -> dict[str, Any]:
    """Run the Ghidra SLEIGH compiler against the generated .slaspec."""
    from rosetta_validate_sla.sla.sleigh_compiler import compile_slaspec

    errors: list[str] = []
    lang_dir_str = state.get("lang_dir")
    ghidra_home_str = state.get("ghidra_home")

    if not lang_dir_str:
        errors.append("validate_sla_node: lang_dir not set in state")
        return {"compile_ok": False, "compile_errors": [], "errors": errors}

    if not ghidra_home_str:
        errors.append("validate_sla_node: ghidra_home not set in state")
        return {"compile_ok": False, "compile_errors": [], "errors": errors}

    lang_dir = Path(lang_dir_str)
    ghidra_home = Path(ghidra_home_str)
    specs = list(lang_dir.glob("*.slaspec"))

    if not specs:
        errors.append(f"validate_sla_node: no .slaspec found in {lang_dir}")
        return {"compile_ok": False, "compile_errors": [], "errors": errors}

    all_ok = True
    compile_errors: list[str] = []

    for slaspec in specs:
        try:
            result = compile_slaspec(slaspec, ghidra_home)
            if not result.ok:
                all_ok = False
                compile_errors.extend(result.errors)
        except Exception as exc:
            all_ok = False
            compile_errors.append(str(exc))
            errors.append(f"validate_sla_node: {exc}")

    log.info("validate_sla_node: compile_ok=%s errors=%d", all_ok, len(compile_errors))
    return {"compile_ok": all_ok, "compile_errors": compile_errors, "errors": errors}
