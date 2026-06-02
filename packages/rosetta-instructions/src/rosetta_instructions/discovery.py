"""Cursor-based instruction discovery: asks ChromaDB for current + next instruction."""

from __future__ import annotations

import logging
import re
from typing import Any

import docquery
from langsmith import traceable
from pydantic import BaseModel

log = logging.getLogger(__name__)

_VALID_MNEMONIC = re.compile(r"^[A-Z][A-Z0-9]{0,15}(?:\.[A-Z0-9\.]{0,10})?$")

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst reading an instruction-set manual. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "Do NOT use prior knowledge of any instruction set; do NOT invent mnemonics. "
    "Return only JSON matching the schema."
)

# How many recently-seen mnemonics to include in the prompt to help the LLM advance.
_SEEN_SAMPLE_SIZE = 10
# How many consecutive iterations returning a mnemonic already in seen triggers stall.
_STALL_LIMIT = 5


class _CursorResult(BaseModel):
    current: str | None = None
    next: str | None = None


@traceable(
    run_type="retriever",
    name="discover_next",
    process_inputs=lambda kw: {"last": kw.get("last"), "seen_count": len(kw.get("seen") or [])},
)
def discover_next(
    last: str | None,
    seen: list[str],
    settings: Any,
) -> tuple[str | None, str | None]:
    """Ask ChromaDB for the instruction immediately after *last* and the one after that.

    Returns (current, next) where both may be None at end-of-manual.
    """
    seen_sample = seen[-_SEEN_SAMPLE_SIZE:] if seen else []
    seen_clause = (
        f" Do not return any of these already-processed mnemonics: {seen_sample}."
        if seen_sample
        else ""
    )

    if last:
        query = (
            f"From the manual context only, identify the instruction documented "
            f"immediately after {last!r} (return its mnemonic as 'current') and "
            f"the instruction documented after that (return its mnemonic as 'next')."
            f"{seen_clause}"
            f" If no further instructions remain in the manual, return null for both."
        )
    else:
        # First iteration: find the very first instruction in the manual.
        query = (
            "From the manual context only, identify the first instruction documented "
            "in this manual (return its mnemonic as 'current') and the instruction "
            "documented after that (return its mnemonic as 'next')."
            f"{seen_clause}"
            " If the manual has no instructions, return null for both."
        )

    try:
        result = docquery.query(
            query,
            schema=_CursorResult,
            system_prompt=_SYSTEM_PROMPT,
            settings=settings,
        )
    except Exception as exc:
        log.warning("discovery failed: %s", exc)
        return None, None

    if not isinstance(result, _CursorResult):
        return None, None

    current = _clean(result.current)
    next_ = _clean(result.next)
    log.debug("discover_next: last=%r → current=%r next=%r", last, current, next_)
    return current, next_


def _clean(raw: str | None) -> str | None:
    if not raw:
        return None
    m = raw.strip().upper()
    # Strip assembly suffix noise (e.g. "ADD Rd, Rn" → "ADD")
    m = m.split()[0] if m else m
    # Strip punctuation artefacts
    m = re.sub(r"[^A-Z0-9._]", "", m)
    if _VALID_MNEMONIC.match(m):
        return m
    return None
