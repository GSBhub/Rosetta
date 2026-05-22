"""Node 1: Extract ISAMeta (Pass 1) from a docquery RAG database.

Reads from state:  db_path, settings_dict
Returns to state:  meta, errors
"""

from __future__ import annotations

import logging
from typing import Any

from rosetta_schemas.models import ISAMeta
from rosetta_schemas.state import PipelineState

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Extract the ISA metadata. "
    "Return only JSON matching the schema."
)

_QUERY = (
    "What is the endianness (little or big), native word size in bits, "
    "minimum instruction alignment in bytes, and all possible instruction "
    "widths in bits for this ISA? Also provide a short ISA name and a "
    "short version/variant identifier (e.g. 'v7', 'v8', 'v8A', 'Cortex-M', "
    "'default' if unknown). Use only alphanumeric characters and hyphens."
)


def meta_node(state: PipelineState) -> dict[str, Any]:
    """Extract ISAMeta via RAG ExtractionPipeline."""
    import docquery
    from docquery.config import Settings
    from rosetta_utils.chroma import get_chroma_wrapper

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_wrapper(settings.db_path, settings)
        result = docquery.query(_QUERY, schema=ISAMeta, system_prompt=_SYSTEM_PROMPT, settings=settings)
        if isinstance(result, ISAMeta):
            log.info("meta_node: extracted ISAMeta name=%r endian=%s", result.name, result.endian)
            return {"meta": result.model_dump(), "errors": []}

        log.warning("meta_node: unexpected result type %s, using fallback", type(result))
    except Exception as exc:
        log.warning("meta_node failed: %s — using fallback ISAMeta", exc)
        return {"meta": _fallback().model_dump(), "errors": [f"meta_node: {exc}"]}

    return {"meta": _fallback().model_dump(), "errors": []}


def _fallback() -> ISAMeta:
    return ISAMeta(
        name="Unknown",
        endian="little",
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[32],
        variant="default",
    )
