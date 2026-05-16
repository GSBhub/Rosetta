"""Five-pass ISA extraction using docquery's ExtractionPipeline + ChatAgent."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from docquery.config import Settings as DocSettings
from docquery.embeddings.provider import get_embeddings
from docquery.embeddings.llm import get_llm
from docquery.pipeline.extractor import ExtractionPipeline
from docquery.storage.vector_store import VectorStore
from pydantic import BaseModel

from rosetta.extraction.schemas import (
    ISAMeta,
    ISASpec,
    InstructionDef,
    RegisterDef,
)

log = logging.getLogger(__name__)

_PCODE_SYSTEM = """\
You are an expert in Ghidra's SLEIGH language. Given a natural-language description of an
instruction's semantics, produce a single-line SLEIGH P-code statement that captures the
core operation. Use register names as variables. Examples:
  "Adds Rn and Rm, stores result in Rd" → "Rd = Rn + Rm;"
  "Loads a 32-bit word from memory at address Rn into Rd" → "Rd = *[ram]:4 Rn;"
  "Branches to the address in Rm" → "goto [Rm];"
Return ONLY the P-code statement, no explanation.
"""


class _MnemonicList(BaseModel):
    mnemonics: list[str]


class _RegisterList(BaseModel):
    registers: list[RegisterDef]


def _make_pipeline(db_path: str, output_model: Any, system_prompt: str, settings: DocSettings) -> ExtractionPipeline:
    return ExtractionPipeline(
        db_path=db_path,
        output_model=output_model,
        system_prompt=system_prompt,
        settings=settings,
    )


async def _extract_instruction_async(
    mnemonic: str,
    db_path: str,
    settings: DocSettings,
    semaphore: asyncio.Semaphore,
) -> InstructionDef:
    async with semaphore:
        pipeline = _make_pipeline(
            db_path=db_path,
            output_model=InstructionDef,
            system_prompt=(
                "You are an expert ISA analyst. Extract precise encoding details for the "
                "instruction. For bit_fields provide 'high:low' notation. For bit_constraints "
                "provide the required binary value. If unknown, use empty dict. "
                "Return only JSON matching the schema."
            ),
            settings=settings,
        )
        query = (
            f"For the {mnemonic} instruction: list all assembly syntax variants, "
            f"the encoding width in bits, all bit field names with their bit positions "
            f"(high:low), any required bit values, operand names, and a full description "
            f"of the operation semantics."
        )
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pipeline.run(query)
            )
            if isinstance(result, InstructionDef):
                return result
            # Fallback: build a minimal def if extraction partially succeeded
            return InstructionDef(
                mnemonic=mnemonic,
                semantics=str(result),
                encoding_bits=32,
            )
        except Exception as exc:
            log.warning("Failed to extract %s: %s", mnemonic, exc)
            return InstructionDef(
                mnemonic=mnemonic,
                semantics="Unknown",
                encoding_bits=32,
            )


def _generate_pcode(instruction: InstructionDef, settings: DocSettings) -> str:
    """Call the LLM directly to translate semantics → P-code hint."""
    try:
        llm = get_llm(settings)
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=_PCODE_SYSTEM),
            HumanMessage(content=f"Instruction: {instruction.mnemonic}\nSemantics: {instruction.semantics}"),
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        log.warning("P-code generation failed for %s: %s", instruction.mnemonic, exc)
        return f"# TODO: {instruction.semantics}"


class ISAExtractor:
    """Orchestrates five-pass extraction of an ISASpec from a docquery database."""

    def __init__(self, db_path: str | Path, settings: DocSettings | None = None):
        self.db_path = str(db_path)
        self.settings = settings or DocSettings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, max_concurrent: int = 4) -> ISASpec:
        """Run all five passes and return a complete ISASpec."""
        log.info("Pass 1: extracting ISA metadata")
        meta = self._pass1_meta()

        log.info("Pass 2: extracting register file")
        registers = self._pass2_registers()

        log.info("Pass 3: extracting instruction mnemonic list")
        mnemonics = self._pass3_mnemonics()
        log.info("Found %d mnemonics", len(mnemonics))

        log.info("Pass 4: extracting per-instruction details (concurrency=%d)", max_concurrent)
        instructions = asyncio.run(self._pass4_instructions(mnemonics, max_concurrent))

        log.info("Pass 5: generating P-code hints")
        for instr in instructions:
            if not instr.pcode_hint:
                instr.pcode_hint = _generate_pcode(instr, self.settings)

        return ISASpec(meta=meta, registers=registers, instructions=instructions)

    def save(self, spec: ISASpec, path: str | Path) -> None:
        Path(path).write_text(spec.model_dump_json(indent=2))
        log.info("ISASpec written to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ISASpec:
        return ISASpec.model_validate_json(Path(path).read_text())

    # ------------------------------------------------------------------
    # Extraction passes
    # ------------------------------------------------------------------

    def _pass1_meta(self) -> ISAMeta:
        pipeline = _make_pipeline(
            db_path=self.db_path,
            output_model=ISAMeta,
            system_prompt=(
                "You are an expert ISA analyst. Extract the ISA metadata. "
                "Return only JSON matching the schema."
            ),
            settings=self.settings,
        )
        result = pipeline.run(
            "What is the endianness (little or big), native word size in bits, "
            "minimum instruction alignment in bytes, and all possible instruction "
            "widths in bits for this ISA? Also provide a short ISA name."
        )
        return result if isinstance(result, ISAMeta) else ISAMeta(
            name="Unknown",
            endian="little",
            word_size_bits=32,
            alignment=4,
            instruction_sizes_bits=[32],
        )

    def _pass2_registers(self) -> list[RegisterDef]:
        pipeline = _make_pipeline(
            db_path=self.db_path,
            output_model=_RegisterList,
            system_prompt=(
                "You are an expert ISA analyst. List all programmer-visible registers. "
                "Return only JSON matching the schema."
            ),
            settings=self.settings,
        )
        result = pipeline.run(
            "List every programmer-visible register: canonical name, any aliases, "
            "size in bits, and purpose (e.g. general purpose, stack pointer, program counter)."
        )
        if isinstance(result, _RegisterList):
            return result.registers
        return []

    def _pass3_mnemonics(self) -> list[str]:
        pipeline = _make_pipeline(
            db_path=self.db_path,
            output_model=_MnemonicList,
            system_prompt=(
                "You are an expert ISA analyst. Extract the complete list of instruction "
                "mnemonics defined in this manual. Return only JSON matching the schema."
            ),
            settings=self.settings,
        )
        result = pipeline.run(
            "List every instruction mnemonic defined in this ISA manual. "
            "Include all base mnemonics (e.g. ADD, SUB, LDR) but not condition-code "
            "suffixes as separate entries."
        )
        if isinstance(result, _MnemonicList):
            return [m.upper() for m in result.mnemonics]
        return []

    async def _pass4_instructions(
        self, mnemonics: list[str], max_concurrent: int
    ) -> list[InstructionDef]:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            _extract_instruction_async(m, self.db_path, self.settings, semaphore)
            for m in mnemonics
        ]
        results = await asyncio.gather(*tasks)
        return list(results)
