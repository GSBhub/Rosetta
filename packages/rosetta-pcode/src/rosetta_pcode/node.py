"""Node 5: P-code hint generation (Pass 5).

Reads from state:  instructions, settings_dict, max_pcode
Returns to state:  instructions (updated with pcode_hint), errors
"""

from __future__ import annotations

import logging
from typing import Any

from rosetta_schemas.models import InstructionDef
from rosetta_schemas.state import PipelineState, get_instructions

log = logging.getLogger(__name__)


def pcode_node(state: PipelineState) -> dict[str, Any]:
    """Generate SLEIGH P-code hints for each instruction via direct LLM call."""
    from docquery.config import Settings
    from rosetta_pcode.generator import generate_pcode

    errors: list[str] = []
    instructions = get_instructions(state)
    max_pcode = state.get("max_pcode")

    if not instructions:
        return {"instructions": [], "errors": errors}

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        targets = instructions[:max_pcode] if max_pcode else instructions

        if max_pcode:
            log.info("pcode_node: limiting to first %d instructions", max_pcode)

        for instr in targets:
            if not instr.pcode_hint:
                instr.pcode_hint = generate_pcode(instr, settings)

        log.info("pcode_node: generated hints for %d instructions", len(targets))
        return {"instructions": [i.model_dump() for i in instructions], "errors": errors}

    except Exception as exc:
        log.exception("pcode_node failed")
        errors.append(f"pcode_node: {exc}")
        return {"instructions": [i.model_dump() for i in instructions], "errors": errors}
