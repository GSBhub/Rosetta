"""LangGraph-based multi-strategy mnemonic discovery.

A single RAG query (top_k=5) sees ~5000 chars of a ~2 MB manual and
consistently misses most mnemonics.  This module fans out across multiple
targeted query strategies, including NEON/VFP type-suffix variants,
and loops until all strategies are exhausted.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

import docquery

log = logging.getLogger(__name__)

_STRATEGIES: list[str] = [
    "List ALL instruction mnemonics defined in this ISA manual. Include every base mnemonic and every data-type-suffixed variant.",
    "List all data-processing instruction mnemonics defined in this manual: arithmetic operations (addition, subtraction, negation, comparison), logical operations (AND, OR, XOR, NOT), and move/copy operations.",
    "List all memory access instruction mnemonics defined in this manual: load and store operations of all widths, addressing modes, and any acquire/release or exclusive variants.",
    "List all branch and control-flow instruction mnemonics defined in this manual: unconditional jumps, conditional branches, subroutine calls, returns, and any indirect or register-based variants.",
    "List all multiply, divide, and extended arithmetic instruction mnemonics defined in this manual, including any high-word, widening, or accumulate variants.",
    "List all shift, rotate, and bit-manipulation instruction mnemonics defined in this manual: logical shift, arithmetic shift, rotate, bit-field extract, bit-field insert, count-leading-zeros, and byte-reversal variants.",
    "List all SIMD, vector, and floating-point instruction mnemonics defined in this manual, including all data-type-suffixed variants (e.g. integer, floating-point, and element-size suffixes).",
    "List all system, privileged, hint, barrier, and coprocessor instruction mnemonics defined in this manual: supervisor calls, memory barriers, cache operations, and debug hints.",
    "List all remaining instruction mnemonics defined in this manual not covered by data processing, memory, branch, bit manipulation, or SIMD categories — such as saturation, packing, conversion, and cryptographic instructions.",
]

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Extract the complete list of instruction "
    "mnemonics defined in this manual. Return only JSON matching the schema. "
    "Include only base mnemonics (e.g. ADD, not ADDGE or ADDS). "
    "Upper-case all mnemonics."
)

_COUNT_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Answer with a single integer — "
    "the total number of unique instruction mnemonics defined in this ISA manual."
)

_VALID_MNEMONIC = re.compile(r"^[A-Z][A-Z0-9]{0,15}(?:\.[A-Z0-9\.]{0,10})?$")


class _MnemonicList(BaseModel):
    mnemonics: list[str]


class _InstructionCount(BaseModel):
    count: int


class _MnemonicState(TypedDict):
    settings: Any
    total_expected: int
    mnemonics: list[str]
    strategies: list[str]
    last_new_count: int
    done: bool


def _count_node(state: _MnemonicState) -> _MnemonicState:
    try:
        result = docquery.query(
            "How many unique instruction mnemonics are defined in this ISA manual? "
            "Return only a JSON object with a 'count' field containing the integer.",
            schema=_InstructionCount,
            system_prompt=_COUNT_SYSTEM_PROMPT,
            settings=state["settings"],
        )
        total = result.count if isinstance(result, _InstructionCount) else 50
    except Exception as exc:
        log.warning("Count query failed (%s); defaulting to 50", exc)
        total = 50

    log.info("Estimated instruction count: %d", total)
    return {**state, "total_expected": total}


def _fetch_node(state: _MnemonicState) -> _MnemonicState:
    strategies = list(state["strategies"])
    if not strategies:
        return {**state, "done": True, "last_new_count": 0}

    strategy = strategies.pop(0)
    log.info("Mnemonic strategy: %s", strategy[:80])

    existing = set(state["mnemonics"])
    new_mnemonics: list[str] = []

    try:
        result = docquery.query(
            strategy,
            schema=_MnemonicList,
            system_prompt=_SYSTEM_PROMPT,
            settings=state["settings"],
        )
        if isinstance(result, _MnemonicList):
            for m in result.mnemonics:
                cleaned = _clean_mnemonic(m)
                if cleaned and cleaned not in existing:
                    new_mnemonics.append(cleaned)
                    existing.add(cleaned)
    except Exception as exc:
        log.warning("Strategy fetch failed (%s)", exc)

    log.info("Found %d new mnemonic(s) (total: %d)", len(new_mnemonics), len(existing))
    return {
        **state,
        "strategies": strategies,
        "mnemonics": state["mnemonics"] + new_mnemonics,
        "last_new_count": len(new_mnemonics),
    }


def _check_node(state: _MnemonicState) -> _MnemonicState:
    unique = len(set(state["mnemonics"]))
    expected = max(state["total_expected"], 1)
    coverage = unique / expected
    done = len(state["strategies"]) == 0

    if done and coverage < 0.7:
        log.warning(
            "Low mnemonic coverage: found %d, expected ~%d (%.0f%%). "
            "Supplement with: rosetta ingest <supplement.pdf> --db <same-db-path>",
            unique, expected, coverage * 100,
        )
    else:
        log.info(
            "Check: unique=%d expected=%d coverage=%.0f%% strategies_left=%d → done=%s",
            unique, expected, coverage * 100, len(state["strategies"]), done,
        )
    return {**state, "done": done}


def _route(state: _MnemonicState) -> str:
    return END if state["done"] else "fetch"


def _clean_mnemonic(raw: str) -> str:
    m = raw.strip().upper()
    if " " in m or "," in m or "(" in m:
        return ""
    if _VALID_MNEMONIC.match(m):
        return m
    return ""


def discover_mnemonics(
    db_path: str,
    settings: Any,
    strategies: list[str] | None = None,
) -> list[str]:
    """Fan out across multiple RAG query strategies; return deduplicated uppercase mnemonics."""
    settings.db_path = db_path
    # settings.vs must already be set by the caller (mnemonics_node sets it via
    # get_chroma_wrapper before calling here). Do NOT call _build_chroma — that
    # overwrites settings.vs with the old SQLite-backed chroma and hangs.

    graph = StateGraph(_MnemonicState)
    graph.add_node("count", _count_node)
    graph.add_node("fetch", _fetch_node)
    graph.add_node("check", _check_node)

    graph.add_edge(START, "count")
    graph.add_edge("count", "fetch")
    graph.add_edge("fetch", "check")
    graph.add_conditional_edges("check", _route, {"fetch": "fetch", END: END})

    app = graph.compile()

    initial: _MnemonicState = {
        "settings": settings,
        "total_expected": 0,
        "mnemonics": [],
        "strategies": list(strategies if strategies is not None else _STRATEGIES),
        "last_new_count": 0,
        "done": False,
    }

    final = app.invoke(initial)
    result = sorted(set(m.upper() for m in final["mnemonics"] if m))
    log.info("Mnemonic discovery complete: %d unique mnemonics", len(result))
    return result
