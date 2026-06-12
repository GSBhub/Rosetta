"""Cursor-based register discovery: asks ChromaDB for current + next register."""

from __future__ import annotations

import logging
import re
from typing import Any

import docquery
from langsmith import traceable
from pydantic import BaseModel

from rosetta_schemas.models import RegisterDef

log = logging.getLogger(__name__)

_STALL_LIMIT = 5

# The entity-rule name used to tag register-relevant chunks at ingest time.
# Must match the `--entity <name>=<regex>` name passed to `rosetta ingest`.
REGISTER_ENTITY = "register"

_DISCOVERY_SYSTEM = (
    "You are an expert ISA analyst reading a processor reference manual. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "Do NOT use prior knowledge of any instruction set; do NOT invent registers. "
    "Only consider PROGRAMMER-VISIBLE, GENERAL-PURPOSE registers: "
    "integer registers (e.g. R0–R15, X0–X30), the program counter (PC), "
    "stack pointer (SP), link register (LR), flags/status register (CPSR/APSR/PSTATE), "
    "and floating-point/SIMD registers. "
    "EXCLUDE all of the following: system registers, debug registers, "
    "implementation-defined registers, coprocessor registers, "
    "and any register whose name starts with ED, ID, CTR, SCTLR, TTBR, "
    "VBAR, MPIDR, MIDR, PIDR, CIDR, or similar system/debug prefixes. "
    "Return only JSON matching the schema."
)

_GATHER_SYSTEM = (
    "You are an expert ISA analyst. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "If a field is not present in the context, leave it empty/zero. "
    "Do NOT use prior knowledge of any instruction set. "
    "Return ONLY raw JSON matching the schema. "
    "Do NOT wrap the JSON in markdown code fences or backticks of any kind."
)

_SEEN_SAMPLE = 8


class _NextRegister(BaseModel):
    current: str | None = None
    next: str | None = None


@traceable(
    run_type="retriever",
    name="discover_next_register",
    process_inputs=lambda kw: {"last": kw.get("last"), "seen_count": len(kw.get("seen") or [])},
)
def discover_next_register(
    last: str | None,
    seen: list[str],
    settings: Any,
) -> tuple[str | None, str | None]:
    """Ask ChromaDB for the register immediately after *last* and the one after that."""
    seen_sample = seen[-_SEEN_SAMPLE:] if seen else []
    seen_clause = (
        f" Do not return any of these already-processed registers: {seen_sample}."
        if seen_sample else ""
    )

    _prog_note = (
        " Only consider general-purpose integer registers (e.g. R0-R15, X0-X30), "
        "PC, SP, LR, and status/flags registers (CPSR, APSR, PSTATE). "
        "Skip system, debug, coprocessor, and implementation-defined registers."
    )

    if last:
        query = (
            f"From the manual context only, identify the general-purpose or status register "
            f"documented immediately after {last!r} (return its name as 'current') and "
            f"the next such register (return its name as 'next')."
            f"{_prog_note}{seen_clause}"
            f" If no further programmer-visible registers remain, return null for both."
        )
    else:
        query = (
            "From the manual context only, identify the FIRST general-purpose integer "
            "or status register documented in this ISA manual — such as R0, X0, or CPSR "
            "(return its name as 'current') and the next such register (return its name as 'next')."
            f"{_prog_note}{seen_clause}"
            " If the manual has no programmer-visible registers, return null for both."
        )

    try:
        result = docquery.query(
            query,
            schema=_NextRegister,
            system_prompt=_DISCOVERY_SYSTEM,
            settings=settings,
        )
    except Exception as exc:
        log.warning("discover_next_register failed: %s", exc)
        return None, None

    if not isinstance(result, _NextRegister):
        return None, None

    current = _clean_name(result.current)
    next_ = _clean_name(result.next)
    log.debug("discover_next_register: last=%r → current=%r next=%r", last, current, next_)
    return current, next_


@traceable(
    run_type="chain",
    name="gather_register",
    process_inputs=lambda kw: {"name": kw.get("name")},
)
def gather_register(name: str, settings: Any) -> RegisterDef:
    """Extract full RegisterDef for *name* from the vector store."""
    from docquery._extractor import ExtractionPipeline

    query = (
        f"From the manual context only, describe the {name!r} register: "
        f"its canonical name, any aliases, size in bits, and its purpose or role "
        f"(e.g. accumulator, index, stack pointer, program counter, status register)."
    )

    try:
        pipeline = ExtractionPipeline(
            output_model=RegisterDef,
            system_prompt=_GATHER_SYSTEM,
            settings=settings,
        )
        result = pipeline.run(query)
        if isinstance(result, RegisterDef):
            if not result.name:
                result.name = name
            return result
    except Exception as exc:
        log.warning("gather_register failed for %r: %s", name, exc)

    return RegisterDef(name=name, size_bits=0, description=f"Stub for {name} — extraction failed.")


def _clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    # Strip assembly-noise artefacts (e.g. "R0 (general purpose)" → "R0")
    s = re.split(r"[\s(,]", s)[0]
    s = re.sub(r"[^A-Za-z0-9_.]", "", s)
    return s if s else None
