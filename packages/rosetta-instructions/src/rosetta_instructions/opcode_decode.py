"""Opcode-table decode strategy for CISC/opcode_table ISAs (6502, Z80, M7700, …).

The RISC strategy (decode_graph) walks a queue of *mnemonics*; CISC instead walks
the bounded opcode space — every byte 0x00–0xFF, once per opcode-table prefix —
extracting one OpcodeDef per entry via the LLM. Rows are streamed to the writer's
``write_opcode_table`` seam, which renders the CISC SLEIGH template at close().

Selected by ``meta.encoding_style == "opcode_table"`` in ``decode_node``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def _opcode_space(meta: Any, seen: set[tuple[int | None, int]]) -> list[tuple[int | None, int]]:
    """Every (prefix, opcode) pair to scan, in deterministic order, minus *seen*.

    ``meta.opcode_prefixes`` lists the prefix bytes for multi-byte tables (e.g.
    M7700's 0x89/0x42 groups); an empty list means a single un-prefixed table.
    """
    prefixes: list[int | None] = list(getattr(meta, "opcode_prefixes", None) or []) or [None]
    return [(p, op) for p in prefixes for op in range(0x100) if (p, op) not in seen]


def run_opcode_scan(
    writer: Any,
    settings: Any,
    meta: Any,
    *,
    max_iterations: int | None = None,
    inter_chunk_sleep: float = 0.0,
    seen: set[tuple[int | None, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Scan the opcode table, emit it via the writer, return (opcode_map, errors).

    One ``gather_opcode_def`` call per (prefix, opcode); ``max_iterations`` caps
    the number of extractions (useful for quick tests). Each call is its own
    LangSmith span. ``inter_chunk_sleep`` paces local Ollama (KV-cache GC).
    """
    from rosetta_instructions.opcode_cursor import gather_opcode_def

    items = _opcode_space(meta, seen or set())
    if max_iterations is not None:
        items = items[:max_iterations]

    rows = []
    errors: list[str] = []
    for prefix, opcode in items:
        try:
            od = gather_opcode_def(opcode, prefix, settings)
            if od is not None:
                rows.append(od)
        except Exception as exc:  # gather_opcode_def is defensive, but never trust it
            label = f"0x{prefix:02X}/0x{opcode:02X}" if prefix is not None else f"0x{opcode:02X}"
            log.warning("run_opcode_scan: opcode %s failed: %s", label, exc)
            errors.append(f"opcode({label}): {exc}")
        if inter_chunk_sleep:
            time.sleep(inter_chunk_sleep)

    known = [r for r in rows if r.mnemonic != "UNK"]
    log.info("run_opcode_scan: scanned %d entries, %d documented (non-UNK)", len(rows), len(known))
    writer.write_opcode_table(rows)
    return [r.model_dump() for r in rows], errors
