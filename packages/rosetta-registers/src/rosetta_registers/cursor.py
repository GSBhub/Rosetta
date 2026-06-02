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

_DISCOVERY_SYSTEM = (
    "You are an expert ISA analyst reading a processor reference manual. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "Do NOT use prior knowledge of any instruction set; do NOT invent registers. "
    "Return only JSON matching the schema."
)

_GATHER_SYSTEM = (
    "You are an expert ISA analyst. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "If a field is not present in the context, leave it empty/zero. "
    "Do NOT use prior knowledge of any instruction set. "
    "Return only JSON matching the schema."
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

    if last:
        query = (
            f"From the manual context only, identify the programmer-visible register "
            f"documented immediately after {last!r} (return its name as 'current') and "
            f"the register documented after that (return its name as 'next')."
            f"{seen_clause}"
            f" If no further registers remain, return null for both."
        )
    else:
        query = (
            "From the manual context only, identify the first programmer-visible register "
            "documented in this manual (return its name as 'current') and the register "
            "documented after that (return its name as 'next')."
            f"{seen_clause}"
            " If the manual has no registers, return null for both."
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
