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


async def extract_instruction_async(
    mnemonic: str,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> InstructionDef:
    """Async wrapper: runs sync ExtractionPipeline in a thread-pool executor.

    settings.vs must already be populated via _build_chroma() before calling.
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
            if isinstance(result, InstructionDef):
                return result
            return InstructionDef(mnemonic=mnemonic, semantics=str(result), encoding_bits=32)

        try:
            return await asyncio.get_event_loop().run_in_executor(executor, _run_sync)
        except Exception as exc:
            log.warning("Failed to extract %s: %s", mnemonic, exc)
            return InstructionDef(mnemonic=mnemonic, semantics="Unknown", encoding_bits=32)
