"""Node 6: SLEIGH module generation (SLA backend).

Reads from state:  meta, registers, instructions, processor_name, out_dir
Returns to state:  lang_dir, errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langsmith import traceable
from rosetta_schemas.state import PipelineState, get_isa_spec
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)


@traceable(run_type="chain", name="stage:generate_sla", process_inputs=state_summary)
def generate_sla_node(state: PipelineState) -> dict[str, Any]:
    """Render ISASpec → Ghidra SLEIGH files via Jinja2 templates.

    If decode_node already wrote the slaspec (lang_dir is set and the directory
    exists), this node is a no-op — it returns the existing lang_dir unchanged.
    """
    from rosetta_generate_sla.sla.module_generator import ModuleGenerator

    errors: list[str] = []

    # Idempotent: if the decode loop already streamed the output, skip re-generation.
    existing_lang_dir = state.get("lang_dir")
    if existing_lang_dir and Path(existing_lang_dir).exists():
        log.info("generate_sla_node: lang_dir already set by decode node — skipping")
        return {}  # no-op: do not re-write lang_dir (avoid same-step conflict)

    spec = get_isa_spec(state)
    if spec is None:
        errors.append("generate_sla_node: no ISAMeta in state")
        return {"lang_dir": None, "errors": errors}

    processor_name = state.get("processor_name", "Unknown")
    out_dir = Path(state.get("out_dir", "./output"))

    try:
        generator = ModuleGenerator()
        lang_dir = generator.generate(spec, processor_name, out_dir)
        log.info("generate_sla_node: output at %s", lang_dir)
        return {"lang_dir": str(lang_dir), "errors": errors}
    except Exception as exc:
        log.exception("generate_sla_node failed")
        errors.append(f"generate_sla_node: {exc}")
        return {"lang_dir": None, "errors": errors}
