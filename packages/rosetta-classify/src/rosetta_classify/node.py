"""Node 1b: Classify ISA encoding style (Pass 1b) — runs after meta.

Reads from state:  meta, db_path, settings_dict
Returns to state:  meta (with encoding_style set)
"""

from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field

from rosetta_schemas.models import EncodingStyle
from rosetta_schemas.state import PipelineState, get_meta
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Classify the instruction encoding style of this ISA. "
    "Return only JSON matching the schema."
)

_QUERY = (
    "How are instructions encoded in this ISA? "
    "Choose one of:\n"
    "  'opcode_table' — each instruction starts with a single opcode byte (0x00–0xFF), "
    "followed by variable-length operand bytes. The manual contains an opcode map or "
    "opcode table grid. Examples: 6502, Z80, 8051, Motorola 68HC11, Mitsubishi M7700/MELPS-7700.\n"
    "  'fixed_word' — every instruction is a fixed-width word (16 or 32 bits) where "
    "sub-fields within the word encode the opcode, registers, and immediate values. "
    "Examples: ARM32, MIPS, RISC-V, PowerPC.\n"
    "  'variable_prefix' — variable-length encoding with multi-byte prefix sequences "
    "before the opcode. Examples: x86, x86-64.\n"
    "Also briefly explain your reasoning."
)


class _ClassifyResult(BaseModel):
    encoding_style: EncodingStyle = Field(
        description="ISA encoding family: 'opcode_table', 'fixed_word', or 'variable_prefix'"
    )
    reasoning: str = Field(
        default="",
        description="One-sentence explanation of the classification",
    )


@traceable(run_type="chain", name="stage:classify", process_inputs=state_summary)
def classify_node(state: PipelineState) -> dict[str, Any]:
    """Classify the ISA encoding style and store it in meta.encoding_style."""
    import docquery
    from docquery.config import Settings
    from rosetta_utils.chroma import get_chroma_wrapper

    meta = get_meta(state)
    if meta is None:
        log.warning("classify_node: no meta in state — defaulting to fixed_word")
        return {}

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_wrapper(settings.db_path, settings)

        result = docquery.query(
            _QUERY,
            schema=_ClassifyResult,
            system_prompt=_SYSTEM_PROMPT,
            settings=settings,
        )

        if isinstance(result, _ClassifyResult):
            log.info(
                "classify_node: encoding_style=%r  reason=%r",
                result.encoding_style,
                result.reasoning,
            )
            updated = meta.model_dump()
            updated["encoding_style"] = result.encoding_style
            return {"meta": updated, "errors": []}

        log.warning("classify_node: unexpected result type %s", type(result))

    except Exception as exc:
        log.warning("classify_node failed: %s — encoding_style unchanged", exc)
        return {"errors": [f"classify_node: {exc}"]}

    return {}
