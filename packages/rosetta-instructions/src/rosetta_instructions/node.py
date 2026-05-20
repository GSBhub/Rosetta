"""Node 4: Per-instruction extraction (Pass 4).

Reads from state:  mnemonics, db_path, embedding_dim, settings_dict,
                   max_concurrent, chunk_size, memory_warn_gb,
                   inter_chunk_sleep, resume, debug_save_dir, max_instructions
Returns to state:  instructions, errors
"""

from __future__ import annotations

import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosetta_schemas.models import InstructionDef
from rosetta_schemas.state import PipelineState

log = logging.getLogger(__name__)


def instructions_node(state: PipelineState) -> dict[str, Any]:
    """Concurrent asyncio-based per-instruction RAG extraction."""
    return asyncio.run(_instructions_async(state))


async def _instructions_async(state: PipelineState) -> dict[str, Any]:
    from docquery.config import Settings
    from rosetta_utils.memory_guard import check_memory_headroom, log_memory
    from rosetta_instructions.extractor import extract_instruction_async

    errors: list[str] = []

    if state.get("stop_after") in ("meta", "registers", "mnemonics"):
        return {"instructions": [], "errors": errors}

    mnemonics: list[str] = list(state.get("mnemonics", []))

    if state.get("stop_after") == "stubs":
        stubs = [
            InstructionDef(mnemonic=m, encoding_bits=32, semantics=f"Auto-generated stub for {m}.")
            for m in mnemonics
        ]
        log.info("instructions_node: created %d stubs (stop_after=stubs)", len(stubs))
        return {"instructions": [s.model_dump() for s in stubs], "errors": errors}
    max_concurrent: int = state.get("max_concurrent", 2)
    max_instructions = state.get("max_instructions")
    chunk_size = state.get("chunk_size") or max_concurrent
    memory_warn_gb: float = state.get("memory_warn_gb", 2.0)
    inter_chunk_sleep: float = state.get("inter_chunk_sleep", 0.0)
    resume: bool = state.get("resume", False)
    debug_save_dir = state.get("debug_save_dir")

    chunk_save_path: Path | None = None
    if debug_save_dir:
        chunk_save_path = Path(debug_save_dir) / "pass4_partial.jsonl"

    try:
        from rosetta_utils.chroma import get_chroma_wrapper
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_wrapper(settings.db_path, settings)

        results: list[InstructionDef] = []

        # Resume: load already-extracted instructions.
        resume_from = chunk_save_path if resume else None
        if resume_from and resume_from.exists():
            seen: dict[str, InstructionDef] = {}
            for line in resume_from.read_text().splitlines():
                line = line.strip()
                if line:
                    instr = InstructionDef.model_validate_json(line)
                    seen[instr.mnemonic.upper()] = instr
            results = list(seen.values())
            before = len(mnemonics)
            mnemonics = [m for m in mnemonics if m.upper() not in seen]
            log.info("Resume: %d done, %d remaining", before - len(mnemonics), len(mnemonics))
        elif chunk_save_path:
            chunk_save_path.write_text("")

        if max_instructions and len(mnemonics) > max_instructions:
            mnemonics = mnemonics[:max_instructions]
            log.info("Capped at %d instructions", max_instructions)

        semaphore = asyncio.Semaphore(max_concurrent)

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            for i in range(0, len(mnemonics), chunk_size):
                chunk = mnemonics[i : i + chunk_size]
                log.info("Pass 4: %d–%d / %d", i + 1, i + len(chunk), len(mnemonics))
                check_memory_headroom(min_free_gb=memory_warn_gb)
                tasks = [
                    extract_instruction_async(m, settings, semaphore, executor)
                    for m in chunk
                ]
                chunk_results = await asyncio.gather(*tasks)
                results.extend(chunk_results)
                gc.collect()
                log_memory(f"pass4-chunk-{i // chunk_size}")
                if chunk_save_path:
                    with chunk_save_path.open("a") as f:
                        for r in chunk_results:
                            f.write(r.model_dump_json() + "\n")
                if inter_chunk_sleep > 0:
                    await asyncio.sleep(inter_chunk_sleep)

        log.info("instructions_node: %d instructions extracted", len(results))
        return {"instructions": [r.model_dump() for r in results], "errors": errors}

    except Exception as exc:
        log.exception("instructions_node failed")
        errors.append(f"instructions_node: {exc}")
        return {"instructions": [], "errors": errors}
