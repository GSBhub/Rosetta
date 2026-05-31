"""LangGraph subgraph: complete opcode-table discovery via chunk scanning.

Replaces the fixed 16-row similarity-search loop with a three-phase pipeline:

  chunk_discovery  — enumerate all chunks; apply heuristic to find opcode tables
        ↓  (Send fan-out, one per candidate chunk)
  extract_chunk    — direct LLM extraction from chunk text (no retrieval)
        ↓  (fan-in; scan_entries accumulated by operator.add reducer)
  merge            — deduplicate (prefix, opcode) pairs; prefer non-UNK
        ↓
  coverage_check   — identify (prefix, row) gaps in the opcode space
        ↓  (conditional: gaps found and iterations < MAX → targeted_fill; else END)
  targeted_fill    — RAG similarity search for specific missing rows (docquery)
        ↓  (loops back to coverage_check)

The subgraph is built as a compiled StateGraph and invoked from opcode_map_node.
Its output state contains `opcode_map` as a list of OpcodeDef dicts.
"""

from __future__ import annotations

import logging
import operator
import time
from collections import defaultdict
from typing import Annotated, Any

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger(__name__)

_MAX_FILL_ITERATIONS = 2
# If fewer than this fraction of the opcode space is covered, the manual
# probably doesn't have the table in a retrievable form — skip gap fill.
_MIN_COVERAGE_FOR_FILL = 0.10


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ScannerState(TypedDict):
    # Inputs passed in from the parent pipeline state
    meta: dict
    settings_dict: dict
    db_path: str
    inter_chunk_sleep: float

    # Phase 1: chunk discovery
    candidate_chunks: list[dict]

    # Phase 2: fan-out accumulator (operator.add so all extract_chunk results merge)
    scan_entries: Annotated[list[dict], operator.add]

    # Phase 3+: deduplicated result and gap tracking
    opcode_map: list[dict]
    gaps: list[dict]
    fill_iterations: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(state: dict) -> Any:
    from docquery.config import Settings
    from rosetta_utils.chroma import get_chroma_wrapper
    settings = Settings(**(state.get("settings_dict") or {}))
    settings.db_path = state["db_path"]
    settings.vs = get_chroma_wrapper(settings.db_path, settings)
    return settings


def _dedup(entries: list[dict]) -> list[dict]:
    """Deduplicate by (prefix, opcode), preferring non-UNK over UNK."""
    merged: dict[tuple, dict] = {}
    for e in entries:
        key = (e.get("prefix"), e.get("opcode", -1))
        existing = merged.get(key)
        if existing is None or (existing.get("mnemonic") == "UNK" and e.get("mnemonic") != "UNK"):
            merged[key] = e
    return list(merged.values())


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def chunk_discovery_node(state: ScannerState) -> dict:
    """Enumerate all chunks; filter to opcode-table candidates."""
    from rosetta_opcode_map.chunk_scanner import get_all_chunks, looks_like_opcode_table

    settings = _make_settings(state)
    all_chunks = get_all_chunks(settings)
    candidates = [c for c in all_chunks if looks_like_opcode_table(c["text"])]

    log.info(
        "scanner: %d / %d chunks pass opcode-table heuristic",
        len(candidates), len(all_chunks),
    )
    return {
        "candidate_chunks": candidates,
        "scan_entries": [],   # initialise accumulator
        "fill_iterations": 0,
    }


def route_to_extract(state: ScannerState) -> list[Send]:
    """Fan-out: one extract_chunk invocation per candidate chunk."""
    return [
        Send("extract_chunk", {
            "chunk":        chunk,
            "settings_dict": state["settings_dict"],
            "db_path":      state["db_path"],
        })
        for chunk in state["candidate_chunks"]
    ]


def extract_chunk_node(state: dict) -> dict:
    """Fan-out worker: extract OpcodeDef entries from one chunk via direct LLM call."""
    from docquery.config import Settings
    from rosetta_opcode_map.chunk_scanner import extract_opcodes_from_chunk

    settings = Settings(**(state.get("settings_dict") or {}))
    chunk = state["chunk"]
    entries = extract_opcodes_from_chunk(chunk["text"], settings, chunk_id=chunk["id"])
    return {"scan_entries": [e.model_dump() for e in entries]}


def merge_node(state: ScannerState) -> dict:
    """Fan-in: deduplicate all scan_entries into opcode_map."""
    result = _dedup(state.get("scan_entries", []))
    non_unk = sum(1 for e in result if e.get("mnemonic") != "UNK")
    log.info("scanner: merge → %d unique entries (%d non-UNK)", len(result), non_unk)
    return {"opcode_map": result}


def coverage_check_node(state: ScannerState) -> dict:
    """Compute which (prefix, row) pairs need targeted gap-fill."""
    opcode_map = state.get("opcode_map", [])
    meta       = state.get("meta", {})
    prefixes   = [None] + list(meta.get("opcode_prefixes") or [])

    covered = {(e.get("prefix"), e["opcode"]) for e in opcode_map}
    total_space = len(prefixes) * 256

    # If we have too little overall coverage the chunk scanner found nothing useful;
    # targeted fill won't help either — bail out early.
    if len(covered) < total_space * _MIN_COVERAGE_FOR_FILL:
        log.info(
            "scanner: coverage %.0f%% below threshold — skipping gap fill",
            100 * len(covered) / total_space,
        )
        return {"gaps": []}

    gap_rows: dict[tuple, set[int]] = defaultdict(set)
    for prefix in prefixes:
        for opcode in range(256):
            if (prefix, opcode) not in covered:
                gap_rows[(prefix, opcode // 16)].add(opcode)

    gaps = [
        {"prefix": pfx, "row": row, "opcodes": sorted(ops)}
        for (pfx, row), ops in sorted(
            gap_rows.items(),
            key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, kv[0][1]),
        )
    ]
    log.info("scanner: %d gap rows across %d prefix(es)", len(gaps), len(prefixes))
    return {"gaps": gaps}


def should_fill(state: ScannerState) -> str:
    if state.get("gaps") and state.get("fill_iterations", 0) < _MAX_FILL_ITERATIONS:
        return "targeted_fill"
    return END


def targeted_fill_node(state: ScannerState) -> dict:
    """RAG similarity search for specific missing rows; merge into opcode_map."""
    from rosetta_opcode_map.extractor import extract_opcode_row

    settings = _make_settings(state)
    sleep    = float(state.get("inter_chunk_sleep") or 0.5)
    gaps     = state.get("gaps", [])

    # Collect new entries from targeted row queries
    new_entries: list[dict] = []
    seen: set[tuple] = set()
    for gap in gaps:
        key = (gap["prefix"], gap["row"])
        if key in seen:
            continue
        seen.add(key)
        prefix, row = gap["prefix"], gap["row"]
        log.info("scanner: targeted fill — prefix=%s row=0x%Xx", prefix, row)
        try:
            rows = extract_opcode_row(settings, row=row, prefix=prefix)
            # Stamp prefix on entries that the LLM omitted it from
            for e in rows:
                if e.prefix is None and prefix is not None:
                    e.prefix = prefix
            new_entries.extend(e.model_dump() for e in rows)
        except Exception as exc:
            log.warning("scanner: targeted fill failed for %s: %s", key, exc)
        if sleep:
            time.sleep(sleep)

    # Merge new entries into the existing opcode_map
    merged = _dedup(list(state.get("opcode_map", [])) + new_entries)
    log.info("scanner: targeted fill added %d entries → %d total", len(new_entries), len(merged))
    return {
        "opcode_map":     merged,
        "fill_iterations": state.get("fill_iterations", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_scanner_graph():
    """Build and compile the opcode-table scanner subgraph."""
    g = StateGraph(ScannerState)

    g.add_node("chunk_discovery", chunk_discovery_node)
    g.add_node("extract_chunk",   extract_chunk_node)
    g.add_node("merge",           merge_node)
    g.add_node("coverage_check",  coverage_check_node)
    g.add_node("targeted_fill",   targeted_fill_node)

    g.add_edge(START, "chunk_discovery")
    g.add_conditional_edges("chunk_discovery", route_to_extract, ["extract_chunk"])
    g.add_edge("extract_chunk", "merge")
    g.add_edge("merge", "coverage_check")
    g.add_conditional_edges("coverage_check", should_fill, ["targeted_fill", END])
    g.add_edge("targeted_fill", "coverage_check")

    return g.compile()
