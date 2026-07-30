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


def build_encoding_index(settings: Any) -> dict[str, dict]:
    """MNEMONIC(upper) -> grounded encoding, for instructions tagged in the store.

    Empty when no ``instruction`` entities are tagged (no entity rule at
    ingest) or no bit diagrams were recovered — the caller then keeps the LLM
    encoding.
    """
    from docquery._enumerate import enumerate_entities

    vs = getattr(settings, "vs", None)
    if vs is None:
        return {}
    instructions = enumerate_entities(vs, "instruction")
    if not instructions:
        return {}

    by_page = _encodings_by_page(vs)
    index: dict[str, dict] = {}
    for it in instructions:
        recs = by_page.get(it.page)
        if not recs:
            continue
        # prefer the widest recovered encoding on the page (T2 over T1, etc.)
        rec = max(recs, key=lambda r: int(r.get("width") or 0))
        fields = _record_to_fields(rec)
        if fields:
            index.setdefault(it.name.strip().upper(), fields)
    log.info("Grounded encodings: %d of %d instructions have a recovered bit diagram",
             len(index), len(instructions))
    return index
