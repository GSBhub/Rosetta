"""Per-opcode cursor extraction for CISC/opcode_table ISAs.

Iterates the bounded opcode space (0x00–0xFF per table) one entry at a time,
calling the LLM for each opcode byte independently so every extraction appears
as its own named LangSmith span.
"""

from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable
from rosetta_schemas.models import OpcodeDef

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst reading a processor reference manual. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "Do NOT use prior knowledge of any instruction set. "
    "If the opcode is not documented in the context, return mnemonic='UNK' "
    "and operand_bytes=0. "
    "Return only JSON matching the schema."
)


@traceable(
    run_type="chain",
    name="gather_opcode",
    process_inputs=lambda kw: {
        "opcode": hex(kw.get("opcode_byte", 0)),
        "prefix": hex(kw["prefix"]) if kw.get("prefix") is not None else None,
    },
)
def gather_opcode_def(
    opcode_byte: int,
    prefix: int | None,
    settings: Any,
) -> OpcodeDef | None:
    """Query ChromaDB for what a single opcode byte does, return an OpcodeDef.

    Returns None only on hard extraction failure; unknown opcodes come back
    as mnemonic='UNK'.
    """
    from docquery._extractor import ExtractionPipeline

    if prefix is not None:
        query = (
            f"From the manual context only, what does opcode 0x{prefix:02X} 0x{opcode_byte:02X} "
            f"(prefix=0x{prefix:02X}, opcode=0x{opcode_byte:02X}) do in this ISA? "
            f"Provide: mnemonic, addressing mode, number of operand bytes after the opcode, "
            f"and a brief description."
        )
    else:
        query = (
            f"From the manual context only, what does opcode 0x{opcode_byte:02X} "
            f"(decimal {opcode_byte}) do in this ISA? "
            f"Provide: mnemonic, addressing mode, number of operand bytes after the opcode, "
            f"and a brief description."
        )

    try:
        pipeline = ExtractionPipeline(
            output_model=OpcodeDef,
            system_prompt=_SYSTEM_PROMPT,
            settings=settings,
        )
        result = pipeline.run(query)
        if isinstance(result, OpcodeDef):
            result.opcode = opcode_byte
            result.prefix = prefix
            log.debug(
                "opcode 0x%s%02X → %s (%s)",
                f"{prefix:02X}/" if prefix is not None else "",
                opcode_byte,
                result.mnemonic,
                result.mode,
            )
            return result
    except Exception as exc:
        log.warning("gather_opcode_def 0x%02X failed: %s", opcode_byte, exc)

    return OpcodeDef(
        opcode=opcode_byte,
        prefix=prefix,
        mnemonic="UNK",
        mode="imp",
        operand_bytes=0,
    )
