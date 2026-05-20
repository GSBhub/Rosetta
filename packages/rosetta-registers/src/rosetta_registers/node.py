"""Node 2: Extract register file (Pass 2) from a docquery RAG database.

Reads from state:  db_path, settings_dict
Returns to state:  registers, errors
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from rosetta_schemas.models import RegisterDef
from rosetta_schemas.state import PipelineState

log = logging.getLogger(__name__)


class _RegisterList(BaseModel):
    registers: list[RegisterDef]


_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. List all programmer-visible registers. "
    "Return only JSON matching the schema."
)

_QUERY = (
    "List every programmer-visible register: canonical name, any aliases, "
    "size in bits, and purpose (e.g. general purpose, stack pointer, program counter)."
)


def registers_node(state: PipelineState) -> dict[str, Any]:
    """Extract list[RegisterDef] via RAG ExtractionPipeline."""
    import docquery
    from docquery.config import Settings
    from rosetta_utils.chroma import get_chroma_collection

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_collection(settings.db_path)
        result = docquery.query(_QUERY, schema=_RegisterList, system_prompt=_SYSTEM_PROMPT, settings=settings)
        if isinstance(result, _RegisterList):
            log.info("registers_node: found %d registers", len(result.registers))
            return {"registers": [r.model_dump() for r in result.registers], "errors": []}

        log.warning("registers_node: unexpected result type %s", type(result))
    except Exception as exc:
        log.warning("registers_node failed: %s", exc)
        return {"registers": [], "errors": [f"registers_node: {exc}"]}

    return {"registers": [], "errors": []}
