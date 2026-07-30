"""Per-instruction async extraction logic (Pass 4)."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from docquery.config import Settings
from docquery._extractor import ExtractionPipeline

from rosetta_schemas.models import InstructionDef

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Extract precise encoding details for the "
    "instruction. For bit_fields provide 'high:low' notation. For bit_constraints "
    "provide the required binary value. If unknown, use empty dict. "
    "Return only JSON matching the schema."
)


def _apply_grounded_encoding(instr: InstructionDef, grounded: dict | None) -> InstructionDef:
    """Overwrite the LLM's encoding with the grounded bit diagram, if present.

    The decode-critical fields (width, bit_fields, bit_constraints, and the
    operand order they imply) come from docquery's deterministic recovery, so
    a hallucinated bit range cannot reach the SLEIGH constructor. Semantics,
    variants, and the p-code hint stay as the LLM produced them.
    """
    if not grounded:
        return instr
    instr.encoding_bits = grounded["encoding_bits"] or instr.encoding_bits
    instr.bit_fields = grounded["bit_fields"]
    instr.bit_constraints = grounded["bit_constraints"]
    if grounded["operands"]:
        instr.operands = grounded["operands"]
    return instr


async def extract_instruction_async(
    mnemonic: str,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    grounded_encoding: dict | None = None,
) -> InstructionDef:
    """Async wrapper: runs sync ExtractionPipeline in a thread-pool executor.

    settings.vs must already be populated via _build_chroma() before calling.
    When *grounded_encoding* is supplied (from docquery's recovered bit
    diagram), it overrides the LLM's bit_fields / bit_constraints so the decode
    pattern is grounded rather than transcribed by the model.
    """
    async with semaphore:
        query = (
            f"For the {mnemonic} instruction: list all assembly syntax variants, "
            f"the encoding width in bits, all bit field names with their bit positions "
            f"(high:low), any required bit values, operand names, and a full description "
            f"of the operation semantics."
        )

        def _run_sync() -> InstructionDef:
            pipeline = ExtractionPipeline(
                output_model=InstructionDef,
                system_prompt=_SYSTEM_PROMPT,
                settings=settings,
            )
            result = pipeline.run(query)
            if not isinstance(result, InstructionDef):
                result = InstructionDef(mnemonic=mnemonic, semantics=str(result), encoding_bits=32)
            return _apply_grounded_encoding(result, grounded_encoding)

        try:
            return await asyncio.get_event_loop().run_in_executor(executor, _run_sync)
        except Exception as exc:
            log.warning("Failed to extract %s: %s", mnemonic, exc)
            stub = InstructionDef(mnemonic=mnemonic, semantics="Unknown", encoding_bits=32)
            return _apply_grounded_encoding(stub, grounded_encoding)
