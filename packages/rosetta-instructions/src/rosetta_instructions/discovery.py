"""Instruction discovery: enumerate instruction mnemonics from ChromaDB.

The primary entry point is `scan_db_for_mnemonics`. It prefers docquery's structural
entity tags (`entity_instruction` metadata, written at ingest time when the manual is
ingested with an `instruction` entity rule) — the deterministic, complete equivalent of
docquery's `cursor_enumerate("instruction")`. When a DB has no such tags (ingested before
tagging, or without `--entity`), it falls back to the legacy frequency-regex token scan.
Either way it builds a complete mnemonic queue for the decode loop with no LLM calls.

`discover_next` (LLM-based cursor) is retained for backward-compatibility with tests
and as an optional fallback.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from docquery.config import ENTITY_PREFIX

log = logging.getLogger(__name__)

# The entity-rule name used to tag instruction-relevant chunks at ingest time.
# Must match the `--entity <name>=<regex>` name passed to `rosetta ingest`.
INSTRUCTION_ENTITY = "instruction"

_VALID_MNEMONIC = re.compile(r"^[A-Z][A-Z0-9]{0,15}(?:\.[A-Z0-9\.]{0,10})?$")
_TOKEN_RE = re.compile(r'\b([A-Z][A-Z0-9]{1,9})\b')

# Tokens that are definitely not instruction mnemonics.
# Keep this list SMALL — false positives are harmless for the coverage metric.
_EXCLUDE_TOKENS = frozenset({
    # Architecture/brand names
    "ARM", "THUMB", "AARCH",
    # Common document-structure words
    "NOTE", "SEE", "ALSO", "PAGE", "TABLE", "FIGURE",
    # Exception-level suffixes
    "EL0", "EL1", "EL2", "EL3",
    # Very common English words that appear in uppercase headings
    "THE", "FOR", "ARE", "ALL", "ITS", "WHEN", "WITH",
    "THIS", "EACH", "THAT", "INTO", "FROM", "HAVE",
    "ONLY", "MUST", "THEN", "TYPE", "USED",
    # Common technical acronyms that are not instructions
    "ISA", "ABI", "API", "CPU", "MMU", "TLB", "GIC",
    "MSB", "LSB",
})


def read_tagged_mnemonics(
    settings: Any,
    entity_type: str = INSTRUCTION_ENTITY,
) -> list[str] | None:
    """Return distinct instruction mnemonics from docquery entity tags, page-ordered.

    Thin wrapper over the shared `rosetta_utils.entities.read_tagged_entities` — the
    deterministic equivalent of `cursor_enumerate(entity_type)`. Returns ``None``
    when the store has no tags of this type so the caller can fall back.
    """
    from rosetta_utils.entities import read_tagged_entities

    return read_tagged_entities(settings, entity_type)


def scan_db_for_mnemonics(
    settings: Any,
    min_freq: int = 3,
    reference_filter: "set[str] | None" = None,
) -> list[str]:
    """Enumerate instruction mnemonics for the decode queue.

    Tag mode (preferred): if the DB was ingested with an ``instruction`` entity rule,
        returns the distinct tagged mnemonics in page order. With *reference_filter*,
        restricts to those also defined by the Ghidra reference (sorted).

    Frequency mode (fallback, when no tags exist): scans raw document text for
        uppercase mnemonic-like tokens.
        - *reference_filter* set: returns reference tokens found at least once, sorted.
        - otherwise: returns tokens meeting *min_freq*, by descending frequency.
    """
    tagged = read_tagged_mnemonics(settings)
    if tagged is not None:
        if reference_filter:
            ref = {r for r in reference_filter}
            found = sorted(m for m in set(tagged) if m in ref)
            log.info(
                "scan_db_for_mnemonics: %d of %d reference mnemonics tagged in DB",
                len(found), len(reference_filter),
            )
            return found
        log.info("scan_db_for_mnemonics: %d distinct tagged instruction entities", len(tagged))
        return tagged

    log.warning(
        "scan_db_for_mnemonics: no '%s%s' tags found — falling back to frequency scan. "
        "Re-ingest with `--entity %s=<regex>` for deterministic enumeration.",
        ENTITY_PREFIX, INSTRUCTION_ENTITY, INSTRUCTION_ENTITY,
    )
    return _frequency_scan(settings, min_freq, reference_filter)


def _frequency_scan(
    settings: Any,
    min_freq: int,
    reference_filter: "set[str] | None",
) -> list[str]:
    """Legacy regex token-frequency scan over all document text."""
    try:
        collection = settings.vs._collection
        total = collection.count()
        if total == 0:
            log.warning("_frequency_scan: collection is empty")
            return []
        results = collection.get(include=["documents"], limit=total)
        docs: list[str] = results.get("documents") or []
    except Exception as exc:
        log.warning("_frequency_scan: DB access failed: %s", exc)
        return []

    counts: Counter[str] = Counter()

    if reference_filter:
        # Reference-filter mode: tally only tokens that appear in the reference set.
        for doc in docs:
            if not doc:
                continue
            for match in _TOKEN_RE.finditer(doc):
                token = match.group(1)
                if token in reference_filter:
                    counts[token] += 1
        found = sorted(tok for tok in reference_filter if counts.get(tok, 0) >= 1)
        log.info(
            "scan_db_for_mnemonics: %d of %d reference mnemonics found in DB",
            len(found),
            len(reference_filter),
        )
        return found

    # Frequency mode: extract all candidate tokens above the frequency threshold.
    for doc in docs:
        if not doc:
            continue
        for match in _TOKEN_RE.finditer(doc):
            token = match.group(1)
            if (
                2 <= len(token) <= 10
                and token not in _EXCLUDE_TOKENS
                and _VALID_MNEMONIC.match(token)
            ):
                counts[token] += 1

    mnemonics = [tok for tok, cnt in counts.most_common() if cnt >= min_freq]
    log.info(
        "scan_db_for_mnemonics: %d candidate mnemonics (min_freq=%d) from %d docs",
        len(mnemonics),
        min_freq,
        len(docs),
    )
    return mnemonics


# ---------------------------------------------------------------------------
# Legacy LLM-based cursor (kept for tests and optional fallback)
# ---------------------------------------------------------------------------

from langsmith import traceable
from pydantic import BaseModel

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst reading an instruction-set manual. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "Do NOT use prior knowledge of any instruction set; do NOT invent mnemonics. "
    "Return only JSON matching the schema."
)

_SEEN_SAMPLE_SIZE = 10


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
    This LLM-based cursor is used as a fallback when scan_db_for_mnemonics fails.
    """
    import docquery

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
    m = m.split()[0] if m else m
    m = re.sub(r"[^A-Z0-9._]", "", m)
    if _VALID_MNEMONIC.match(m):
        return m
    return None
