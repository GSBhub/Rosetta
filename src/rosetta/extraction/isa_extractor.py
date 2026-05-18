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


def _probe_embedding_dim(settings: DocSettings) -> int:
    """Return the actual embedding dimension for the configured model."""
    embeddings = get_embeddings(settings)
    return len(embeddings.embed_query("probe"))


def _make_pipeline(db_path: str, output_model: Any, system_prompt: str, settings: DocSettings, embedding_dim: int | None = None) -> ExtractionPipeline:
    dim = embedding_dim if embedding_dim is not None else _probe_embedding_dim(settings)
    return ExtractionPipeline(
        db_path=db_path,
        output_model=output_model,
        system_prompt=system_prompt,
        embedding_dim=dim,
        settings=settings,
    )


async def _extract_instruction_async(
    mnemonic: str,
    db_path: str,
    settings: DocSettings,
    semaphore: asyncio.Semaphore,
    embedding_dim: int | None = None,
) -> InstructionDef:
    async with semaphore:
        query = (
            f"For the {mnemonic} instruction: list all assembly syntax variants, "
            f"the encoding width in bits, all bit field names with their bit positions "
            f"(high:low), any required bit values, operand names, and a full description "
            f"of the operation semantics."
        )

        def _run_sync() -> InstructionDef:
            # Pipeline (and its SQLite connection) must be created inside the
            # executor thread — SQLite connections cannot be shared across threads.
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
                embedding_dim=embedding_dim,
            )
            result = pipeline.run(query)
            if isinstance(result, InstructionDef):
                return result
            return InstructionDef(
                mnemonic=mnemonic,
                semantics=str(result),
                encoding_bits=32,
            )

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _run_sync)
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
        self._embedding_dim: int | None = None  # probed once on first use

    def _get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._embedding_dim = _probe_embedding_dim(self.settings)
            log.info("Embedding dimension: %d", self._embedding_dim)
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        max_concurrent: int = 2,
        max_instructions: int | None = None,
        max_pcode: int | None = None,
        stop_after: str | None = None,
        filter_mnemonics: str | None = None,
        memory_warn_gb: float = 2.0,
        debug_save_dir: Path | None = None,
        debug_prefix: str = "debug",
    ) -> ISASpec:
        """Run extraction passes and return an ISASpec.

        stop_after: stop early after "meta", "registers", "mnemonics", or "instructions".
        filter_mnemonics: comma-separated glob patterns to keep (e.g. "MOV*,ADD,SUB*").
        max_pcode: generate P-code for only the first N instructions in pass 5.
        debug_save_dir: write per-pass JSON snapshots here for post-mortem inspection.
        """
        from rosetta.utils.memory_guard import check_memory_headroom, log_memory

        log_memory("extract-start")
        check_memory_headroom(min_free_gb=memory_warn_gb)

        log.info("Pass 1: extracting ISA metadata")
        meta = self._pass1_meta()
        log_memory("after-pass1")
        if debug_save_dir:
            self._save_debug(ISASpec(meta=meta), debug_save_dir / f"{debug_prefix}_debug_pass1_meta.json")
        if stop_after == "meta":
            return ISASpec(meta=meta)

        log.info("Pass 2: extracting register file")
        registers = self._pass2_registers()
        log_memory("after-pass2")
        if debug_save_dir:
            self._save_debug(ISASpec(meta=meta, registers=registers), debug_save_dir / f"{debug_prefix}_debug_pass2_registers.json")
        if stop_after == "registers":
            return ISASpec(meta=meta, registers=registers)

        log.info("Pass 3: extracting instruction mnemonic list")
        mnemonics = self._pass3_mnemonics()
        log.info("Found %d mnemonics", len(mnemonics))
        log_memory("after-pass3")

        if filter_mnemonics:
            import fnmatch
            patterns = [p.strip().upper() for p in filter_mnemonics.split(",")]
            before = len(mnemonics)
            mnemonics = [m for m in mnemonics if any(fnmatch.fnmatch(m, p) for p in patterns)]
            log.info("Filter %r: %d → %d mnemonics", filter_mnemonics, before, len(mnemonics))

        if debug_save_dir:
            self._save_debug(mnemonics, debug_save_dir / f"{debug_prefix}_debug_pass3_mnemonics.json")
        if stop_after == "mnemonics":
            return ISASpec(meta=meta, registers=registers)

        if max_instructions and len(mnemonics) > max_instructions:
            log.info("Capping at %d instructions (--max-instructions)", max_instructions)
            mnemonics = mnemonics[:max_instructions]

        chunk_save = (debug_save_dir / f"{debug_prefix}_debug_pass4_partial.json") if debug_save_dir else None
        log.info("Pass 4: extracting per-instruction details (concurrency=%d)", max_concurrent)
        instructions = asyncio.run(
            self._pass4_instructions(mnemonics, max_concurrent, memory_warn_gb, chunk_save_path=chunk_save)
        )
        log_memory("after-pass4")
        if debug_save_dir:
            self._save_debug(
                ISASpec(meta=meta, registers=registers, instructions=instructions),
                debug_save_dir / f"{debug_prefix}_debug_pass4_instructions.json",
            )
        if stop_after == "instructions":
            return ISASpec(meta=meta, registers=registers, instructions=instructions)

        log.info("Pass 5: generating P-code hints")
        pcode_targets = instructions[:max_pcode] if max_pcode else instructions
        if max_pcode:
            log.info("Limiting P-code generation to first %d instructions", max_pcode)
        for instr in pcode_targets:
            if not instr.pcode_hint:
                instr.pcode_hint = _generate_pcode(instr, self.settings)
        log_memory("after-pass5")

        return ISASpec(meta=meta, registers=registers, instructions=instructions)

    def _save_debug(self, obj: Any, path: Path) -> None:
        if isinstance(obj, ISASpec):
            path.write_text(obj.model_dump_json(indent=2))
        else:
            path.write_text(json.dumps(obj, indent=2))
        log.info("Debug save → %s", path)

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
            embedding_dim=self._get_embedding_dim(),
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
            embedding_dim=self._get_embedding_dim(),
        )
        result = pipeline.run(
            "List every programmer-visible register: canonical name, any aliases, "
            "size in bits, and purpose (e.g. general purpose, stack pointer, program counter)."
        )
        if isinstance(result, _RegisterList):
            return result.registers
        return []

    def _pass3_mnemonics(self) -> list[str]:
        from rosetta.extraction.mnemonic_discovery import discover_mnemonics
        return discover_mnemonics(self.db_path, self.settings, self._get_embedding_dim())

    async def _pass4_instructions(
        self,
        mnemonics: list[str],
        max_concurrent: int,
        memory_warn_gb: float = 2.0,
        chunk_size: int = 20,
        chunk_save_path: Path | None = None,
    ) -> list[InstructionDef]:
        import gc
        from rosetta.utils.memory_guard import check_memory_headroom, log_memory

        semaphore = asyncio.Semaphore(max_concurrent)
        dim = self._get_embedding_dim()
        results: list[InstructionDef] = []

        for i in range(0, len(mnemonics), chunk_size):
            chunk = mnemonics[i : i + chunk_size]
            log.info(
                "Pass 4: instructions %d–%d / %d",
                i + 1,
                i + len(chunk),
                len(mnemonics),
            )
            check_memory_headroom(min_free_gb=memory_warn_gb)
            tasks = [
                _extract_instruction_async(m, self.db_path, self.settings, semaphore, dim)
                for m in chunk
            ]
            chunk_results = await asyncio.gather(*tasks)
            results.extend(chunk_results)
            gc.collect()
            log_memory(f"pass4-chunk-{i // chunk_size}")
            if chunk_save_path:
                chunk_save_path.write_text(json.dumps([r.model_dump() for r in results], indent=2))

        return results
