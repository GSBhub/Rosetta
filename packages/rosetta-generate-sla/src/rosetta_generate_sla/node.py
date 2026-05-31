"""Node 6: SLEIGH module generation (SLA backend).

Reads from state:  meta, registers, instructions, processor_name, out_dir
Returns to state:  lang_dir, errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosetta_schemas.state import PipelineState, get_isa_spec

log = logging.getLogger(__name__)


def generate_sla_node(state: PipelineState) -> dict[str, Any]:
    """Render ISASpec → Ghidra SLEIGH files via Jinja2 templates."""
    from rosetta_generate_sla.sla.module_generator import ModuleGenerator

    errors: list[str] = []

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
