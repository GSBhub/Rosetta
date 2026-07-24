"""Grounded instruction encodings from docquery's bit-diagram recovery.

Reading bit positions out of an encoding diagram by eye is the highest-
hallucination task in the pipeline (a wrong bit range silently produces a
wrong SLEIGH decode pattern). docquery already recovers those diagrams
deterministically at ingest as ``encoding_grid`` structure documents — machine
``ENCODING`` lines with exact ``name[hi:lo]`` fields and fixed ``bits[hi:lo]=
value`` constraints. This module turns them into the ``bit_fields`` /
``bit_constraints`` an ``InstructionDef`` needs, keyed by mnemonic, so the LLM
extraction of the encoding can be overridden with grounded geometry.

An instruction is matched to its encoding by page: the ``instruction`` entity
and the ``encoding_grid`` document share the manual page the diagram sits on.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _encodings_by_page(vs: Any) -> dict[Any, list[dict]]:
    """{page: [parsed ENCODING records]} from the store's encoding_grid docs."""
    from docquery._bitgrid import parse_encoding_line

    try:
        raw = vs._collection.get(  # type: ignore[attr-defined]
            where={"kind": "encoding_grid"}, include=["documents", "metadatas"])
    except Exception as exc:  # pragma: no cover - store shape
        # Degrading here is silent in effect: the build still succeeds, just with
        # stub constructors instead of real decode patterns. Log at error level
        # so the cause is visible rather than inferred from a poor decode score.
        log.error("could not read encoding_grid docs (%s); "
                  "grounding disabled, constructors will be stubs", exc)
        return {}

    by_page: dict[Any, list[dict]] = {}
    for doc, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        page = (meta or {}).get("page")
        for line in (doc or "").splitlines():
            if line.strip().startswith("ENCODING "):
                rec = parse_encoding_line(line.strip())
                if rec and rec.get("segments"):
                    by_page.setdefault(page, []).append(rec)
    return by_page


def _record_to_fields(rec: dict) -> dict | None:
    """One ENCODING record -> {encoding_bits, bit_fields, bit_constraints, operands}."""
    bit_fields: dict[str, str] = {}
    bit_constraints: dict[str, str] = {}
    operands: list[str] = []
    for seg in rec["segments"]:
        rng = f"{seg['hi']}:{seg['lo']}"
        if seg.get("value") is not None:
            # Fixed opcode bits — the decode pattern. Always name these by
            # position, never by the manual's label: a label is not
            # position-stable across a manual (the same name appears at
            # different ranges), and a constrained field that collides with a
            # differently-ranged one elsewhere would decode from wrong bits.
            name = f"op_{seg['hi']}_{seg['lo']}"
            bit_fields[name] = rng
            bit_constraints[name] = seg["value"]
        elif seg.get("name"):
            bit_fields[seg["name"]] = rng
            operands.append(seg["name"])
    if not bit_fields:
        return None
    return {
        "encoding_bits": int(rec.get("width") or 0),
        "bit_fields": bit_fields,
        "bit_constraints": bit_constraints,
        "operands": operands,
    }


# Max pages an instruction description may span when collecting its diagrams —
# bounds the page range so we never grab the next instruction's encodings.
_MAX_PAGE_SPAN = 6


def build_encoding_index(settings: Any) -> dict[str, list[dict]]:
    """MNEMONIC(upper) -> list of grounded encodings, for tagged instructions.

    An instruction description spans a page *range*: the entity (its ``Syntax``
    line) up to the next instruction's. Diagrams often sit a page or more after
    the syntax (and each functional unit — .L/.S/.D — is its own page), so we
    collect every distinct recovered encoding in that range, not just the
    entity's exact page. A C6x mnemonic thus maps to many encodings (one per
    Opfield opcode value across its unit pages). Empty when no ``instruction``
    entities are tagged or no bit diagrams were recovered.
    """
    from docquery._enumerate import enumerate_entities

    vs = getattr(settings, "vs", None)
    if vs is None:
        return {}
    instructions = enumerate_entities(vs, "instruction")
    if not instructions:
        return {}

    by_page = _encodings_by_page(vs)
    # Page range per instruction: [entity page, next instruction's page).
    pages = sorted({it.page for it in instructions if it.page is not None})
    next_page = {p: pages[i + 1] for i, p in enumerate(pages[:-1])}

    index: dict[str, list[dict]] = {}
    for it in instructions:
        mnem = it.name.strip().upper()
        if mnem in index or it.page is None:
            continue
        start = it.page
        stop = min(next_page.get(start, start + _MAX_PAGE_SPAN), start + _MAX_PAGE_SPAN)
        encs: list[dict] = []
        seen: set = set()
        for page in range(start, max(stop, start + 1)):
            for rec in by_page.get(page, []):
                fields = _record_to_fields(rec)
                if not fields:
                    continue
                sig = (fields["encoding_bits"], tuple(sorted(fields["bit_constraints"].items())))
                if sig in seen:
                    continue
                seen.add(sig)
                encs.append(fields)
        if encs:
            index[mnem] = encs
    total = sum(len(v) for v in index.values())
    log.info("Grounded encodings: %d encodings for %d of %d instructions",
             total, len(index), len(instructions))
    return index
