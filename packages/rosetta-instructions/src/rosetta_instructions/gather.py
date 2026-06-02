"""Gathers full encoding details for a single instruction from ChromaDB."""

from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable
from rosetta_schemas.models import InstructionDef

log = logging.getLogger(__name__)

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
    from docquery._extractor import ExtractionPipeline

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
        pipeline = ExtractionPipeline(
            output_model=InstructionDef,
            system_prompt=_SYSTEM_PROMPT,
            settings=settings,
        )
        result = pipeline.run(query)
        if isinstance(result, InstructionDef):
            # Ensure the mnemonic matches what we asked for (LLM sometimes drifts).
            if not result.mnemonic or result.mnemonic.upper() == "UNKNOWN":
                result.mnemonic = current
            return result
    except Exception as exc:
        log.warning("gather failed for %s: %s", current, exc)

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
