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


def _encoding_docs(vs: Any) -> list[tuple[str, dict]]:
    """[(document_text, metadata)] for the store's encoding_grid docs."""
    try:
        raw = vs._collection.get(  # type: ignore[attr-defined]
            where={"kind": "encoding_grid"}, include=["documents", "metadatas"])
    except Exception as exc:  # pragma: no cover - store shape
        # Degrading here is silent in effect: the build still succeeds, just with
        # stub constructors instead of real decode patterns. Log at error level
        # so the cause is visible rather than inferred from a poor decode score.
        log.error("could not read encoding_grid docs (%s); "
                  "grounding disabled, constructors will be stubs", exc)
        return []
    return list(zip(raw.get("documents") or [], raw.get("metadatas") or []))


def _parse_records(doc: str) -> list[dict]:
    """Parsed ENCODING records from one encoding_grid document."""
    from docquery._bitgrid import parse_encoding_line

    out: list[dict] = []
    for line in (doc or "").splitlines():
        if line.strip().startswith("ENCODING "):
            rec = parse_encoding_line(line.strip())
            if rec and rec.get("segments"):
                out.append(rec)
    return out


def _encodings_by_page(vs: Any) -> dict[Any, list[dict]]:
    """{page: [parsed ENCODING records]} from the store's encoding_grid docs."""
    by_page: dict[Any, list[dict]] = {}
    for doc, meta in _encoding_docs(vs):
        page = (meta or {}).get("page")
        by_page.setdefault(page, []).extend(_parse_records(doc))
    return by_page


def _encodings_by_owner(vs: Any) -> dict[str, list[dict]]:
    """{MNEMONIC(upper): [parsed ENCODING records]} from owner-tagged docs.

    docquery attributes each recovered diagram to the instruction that owns it by
    reading-order geometry and tags it as ``entity_instruction`` metadata. Keying
    off that name is exact, unlike inferring ownership from the page — manuals
    pack several instructions per page and split descriptions across page breaks.
    Empty when the store predates owner attribution (callers fall back to pages).
    """
    by_owner: dict[str, list[dict]] = {}
    for doc, meta in _encoding_docs(vs):
        raw = (meta or {}).get("entity_instruction")
        if not raw:
            continue
        recs = _parse_records(doc)
        if not recs:
            continue
        for name in str(raw).split(";"):
            name = name.strip().upper()
            if name:
                by_owner.setdefault(name, []).extend(recs)
    return by_owner


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

    def _collect(recs: list[dict]) -> list[dict]:
        """Distinct grounded encodings from parsed records."""
        encs: list[dict] = []
        seen: set = set()
        for rec in recs:
            fields = _record_to_fields(rec)
            if not fields:
                continue
            sig = (fields["encoding_bits"], tuple(sorted(fields["bit_constraints"].items())))
            if sig in seen:
                continue
            seen.add(sig)
            encs.append(fields)
        return encs

    # Preferred: docquery attributed each diagram to its owning instruction by
    # geometry, so match exactly by name.
    by_owner = _encodings_by_owner(vs)
    if by_owner:
        index = {
            mnem: encs
            for mnem in {it.name.strip().upper() for it in instructions}
            if (encs := _collect(by_owner.get(mnem, [])))
        }
        log.info("Grounded encodings (owner-matched): %d encodings for %d of %d instructions",
                 sum(len(v) for v in index.values()), len(index), len(instructions))
        return index

    # Fallback for stores without owner tags: page range per instruction,
    # [entity page, next instruction's page).
    by_page = _encodings_by_page(vs)
    pages = sorted({it.page for it in instructions if it.page is not None})
    next_page = {p: pages[i + 1] for i, p in enumerate(pages[:-1])}

    index: dict[str, list[dict]] = {}
    for it in instructions:
        mnem = it.name.strip().upper()
        if mnem in index or it.page is None:
            continue
        start = it.page
        stop = min(next_page.get(start, start + _MAX_PAGE_SPAN), start + _MAX_PAGE_SPAN)
        recs: list[dict] = []
        for page in range(start, max(stop, start + 1)):
            recs.extend(by_page.get(page, []))
        if encs := _collect(recs):
            index[mnem] = encs
    total = sum(len(v) for v in index.values())
    log.info("Grounded encodings: %d encodings for %d of %d instructions",
             total, len(index), len(instructions))
    return index
