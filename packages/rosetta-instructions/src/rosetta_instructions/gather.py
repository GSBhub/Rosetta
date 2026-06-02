"""Gathers full encoding details for a single instruction from ChromaDB."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langsmith import traceable
from rosetta_schemas.models import InstructionDef

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?})\s*```', re.DOTALL)

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "If a field is not present in the context, leave it empty/zero. "
    "Do NOT use prior knowledge of any instruction set; do NOT invent bit fields, "
    "opcodes, or semantics. "
    "For bit_fields provide 'high:low' notation. "
    "For bit_constraints provide the required binary value. "
    "Return ONLY raw JSON matching the schema. "
    "Do NOT wrap the JSON in markdown code fences or backticks of any kind."
)

_SCHEMA_JSON = json.dumps(InstructionDef.model_json_schema(), indent=2)


def _extract_instruction(query: str, settings: Any) -> InstructionDef | None:
    """Extraction pipeline with markdown fence stripping.

    Mirrors docquery's ExtractionPipeline but strips ```json...``` fences
    before JSON validation so models that wrap responses still parse correctly.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from docquery.embeddings.llm import get_llm
    from docquery.tools.retrieval_tools import make_similarity_tool

    similarity_tool = make_similarity_tool(settings.vs, settings)
    llm = get_llm(settings)
    context = similarity_tool.invoke(query)

    validation_errors: list[str] = []
    for attempt in range(3):
        msgs = [
            SystemMessage(content=(
                f"{_SYSTEM_PROMPT}\n\n"
                f"Output ONLY valid JSON matching this schema:\n{_SCHEMA_JSON}\n"
                "Do not include markdown code fences or explanation."
            )),
            HumanMessage(content=(
                f"Context:\n{context}\n\nQuery: {query}"
                + (
                    f"\n\nPrevious attempt validation errors:\n" + "\n".join(validation_errors)
                    if validation_errors else ""
                )
            )),
        ]
        response = llm.invoke(msgs)
        raw = response.content.strip()

        # Strip markdown fences that some models add despite instructions.
        m = _FENCE_RE.search(raw)
        if m:
            raw = m.group(1).strip()

        try:
            return InstructionDef.model_validate_json(raw)
        except Exception as exc:
            validation_errors = [str(exc)]
            log.debug("_extract_instruction attempt %d/%d failed: %s", attempt + 1, 3, exc)

    return None


@traceable(
    run_type="chain",
    name="gather_instruction",
    process_inputs=lambda kw: {"current": kw.get("current"), "next": kw.get("next_mnemonic")},
)
def gather_instruction(
    current: str,
    next_mnemonic: str | None,
    settings: Any,
) -> InstructionDef:
    """Extract full InstructionDef for *current* from the vector store.

    *next_mnemonic* is passed as context so the LLM does not bleed fields
    from the following instruction into the current one.
    """
    next_clause = (
        f" The following instruction in the manual is {next_mnemonic!r} — "
        f"do not include its fields in the output."
        if next_mnemonic
        else ""
    )
    query = (
        f"From the manual context only, extract the encoding for the {current!r} instruction."
        f"{next_clause}"
        f" Provide: all assembly syntax variants, the encoding width in bits, "
        f"all bit field names with their high:low bit positions, any required bit values, "
        f"operand names, and a full description of the operation semantics."
    )

    try:
        result = _extract_instruction(query, settings)
        if result is not None:
            if not result.mnemonic or result.mnemonic.upper() == "UNKNOWN":
                result.mnemonic = current
            return result
    except Exception as exc:
        log.warning("gather failed for %s: %s", current, exc)

    log.warning("gather failed for %s: all extraction attempts returned None", current)
    return InstructionDef(
        mnemonic=current,
        encoding_bits=32,
        semantics=f"Stub for {current} — extraction failed.",
    )


@traceable(
    run_type="tool",
    name="enrich_pcode",
    process_inputs=lambda kw: {"mnemonic": kw["instr"].mnemonic if hasattr(kw.get("instr"), "mnemonic") else str(kw.get("instr"))},
)
def enrich_pcode(instr: InstructionDef, settings: Any) -> InstructionDef:
    """Fill instr.pcode_hint via a direct LLM call (no RAG needed)."""
    if instr.pcode_hint:
        return instr
    try:
        from rosetta_pcode.generator import generate_pcode
        instr.pcode_hint = generate_pcode(instr, settings)
    except Exception as exc:
        log.warning("pcode generation failed for %s: %s", instr.mnemonic, exc)
    return instr
