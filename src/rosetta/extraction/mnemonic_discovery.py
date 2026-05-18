"""LangGraph-based multi-strategy mnemonic discovery.

A single RAG query (top_k=5) sees ~5000 chars of a ~2 MB manual and
consistently misses most mnemonics.  This module fans out across ten
targeted query strategies and loops until coverage stabilises.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from docquery.pipeline.extractor import ExtractionPipeline

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategies — ordered roughly by expected yield so the early iterations
# capture the bulk of mnemonics quickly.
# ---------------------------------------------------------------------------

_STRATEGIES: list[str] = [
    "List ALL instruction mnemonics defined in this ISA manual. Include every base mnemonic.",
    "List all data-processing instruction mnemonics: arithmetic (ADD, SUB, ADC, SBC, RSB, RSC), logical (AND, ORR, EOR, BIC), move (MOV, MVN), and comparison (CMP, CMN, TST, TEQ).",
    "List all load and store instruction mnemonics (LDR, STR, LDM, STM, PUSH, POP and all their variants).",
    "List all branch and control-flow instruction mnemonics (B, BL, BX, BLX, CBZ, CBNZ, TBB, TBH, IT).",
    "List all multiply and divide instruction mnemonics (MUL, MLA, MLS, UMULL, UMLAL, SMULL, SMLAL, SDIV, UDIV).",
    "List instruction mnemonics beginning with letters A through F (e.g. ADC, ADD, AND, ASR, B, BFC, BFI, BIC, BKP, BL, BLX, BX, CDP, CLZ, CMN, CMP, EOR, F-prefix instructions).",
    "List instruction mnemonics beginning with letters G through N (e.g. LDR, LDM, LSL, LSR, MLA, MOV, MRC, MRS, MSR, MUL, MVN).",
    "List instruction mnemonics beginning with letters O through Z (e.g. ORR, PLD, PLI, POP, PUSH, REV, ROR, RRX, RSB, RSC, SBC, SEV, SMLAL, SMULL, STM, STR, SUB, SVC, SWP, TEQ, TST, UMLAL, UMULL, WFE, WFI, YIELD).",
    "List all SIMD, VFP, NEON, and floating-point instruction mnemonics (VADD, VSUB, VMUL, VMOV, VLDM, VSTM, VCMP, etc.).",
    "List all system, privileged, coprocessor, hint, and barrier instruction mnemonics (MRC, MCR, SVC, SMC, HVC, ISB, DSB, DMB, PLD, WFE, WFI, SEV, YIELD, NOP, DBG).",
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


# ---------------------------------------------------------------------------
# Pydantic schemas used by ExtractionPipeline
# ---------------------------------------------------------------------------

class _MnemonicList(BaseModel):
    mnemonics: list[str]


class _InstructionCount(BaseModel):
    count: int


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class _MnemonicState(TypedDict):
    db_path: str
    settings: Any
    embedding_dim: int
    total_expected: int
    mnemonics: list[str]       # accumulated, may contain duplicates across iterations
    strategies: list[str]
    last_new_count: int
    done: bool


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _count_node(state: _MnemonicState) -> _MnemonicState:
    """Query for a rough total instruction count to know when we're done."""
    try:
        pipeline = ExtractionPipeline(
            db_path=state["db_path"],
            output_model=_InstructionCount,
            system_prompt=_COUNT_SYSTEM_PROMPT,
            embedding_dim=state["embedding_dim"],
            settings=state["settings"],
        )
        result = pipeline.run(
            "How many unique instruction mnemonics are defined in this ISA manual? "
            "Return only a JSON object with a 'count' field containing the integer."
        )
        total = result.count if isinstance(result, _InstructionCount) else 50
    except Exception as exc:
        log.warning("Count query failed (%s); defaulting to 50", exc)
        total = 50

    log.info("Estimated instruction count: %d", total)
    return {**state, "total_expected": total}


def _fetch_node(state: _MnemonicState) -> _MnemonicState:
    """Pop the next strategy and run it, accumulating results."""
    strategies = list(state["strategies"])
    if not strategies:
        return {**state, "done": True, "last_new_count": 0}

    strategy = strategies.pop(0)
    log.info("Mnemonic discovery strategy: %s", strategy[:80])

    existing = set(state["mnemonics"])
    new_mnemonics: list[str] = []

    try:
        pipeline = ExtractionPipeline(
            db_path=state["db_path"],
            output_model=_MnemonicList,
            system_prompt=_SYSTEM_PROMPT,
            embedding_dim=state["embedding_dim"],
            settings=state["settings"],
        )
        result = pipeline.run(strategy)
        if isinstance(result, _MnemonicList):
            for m in result.mnemonics:
                cleaned = _clean_mnemonic(m)
                if cleaned and cleaned not in existing:
                    new_mnemonics.append(cleaned)
                    existing.add(cleaned)
    except Exception as exc:
        log.warning("Strategy fetch failed (%s)", exc)

    log.info("Strategy found %d new mnemonic(s) (total so far: %d)",
             len(new_mnemonics), len(existing))

    return {
        **state,
        "strategies": strategies,
        "mnemonics": state["mnemonics"] + new_mnemonics,
        "last_new_count": len(new_mnemonics),
    }


def _check_node(state: _MnemonicState) -> _MnemonicState:
    """Decide whether to keep fetching or stop."""
    unique = len(set(state["mnemonics"]))
    expected = max(state["total_expected"], 1)
    coverage = unique / expected

    no_strategies_left = len(state["strategies"]) == 0

    # Only stop when all strategies are exhausted — coverage-based early exit is
    # unreliable because the LLM's instruction-count estimate is often far too low,
    # which causes the loop to terminate after just one strategy.
    done = no_strategies_left
    log.info(
        "Check: unique=%d expected=%d coverage=%.0f%% strategies_left=%d → done=%s",
        unique, expected, coverage * 100, len(state["strategies"]), done,
    )
    return {**state, "done": done}


def _route(state: _MnemonicState) -> str:
    return END if state["done"] else "fetch"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover_mnemonics(db_path: str, settings: Any, embedding_dim: int) -> list[str]:
    """Fan out across multiple RAG query strategies; return deduplicated uppercase mnemonics."""
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
        "db_path": db_path,
        "settings": settings,
        "embedding_dim": embedding_dim,
        "total_expected": 0,
        "mnemonics": [],
        "strategies": list(_STRATEGIES),
        "last_new_count": 0,
        "done": False,
    }

    final = app.invoke(initial)
    result = sorted(set(m.upper() for m in final["mnemonics"] if m))
    log.info("Mnemonic discovery complete: %d unique mnemonics found", len(result))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_MNEMONIC = re.compile(r"^[A-Z][A-Z0-9]{0,9}$")


def _clean_mnemonic(raw: str) -> str:
    """Strip whitespace, upper-case, and reject non-mnemonic strings."""
    m = raw.strip().upper()
    # Drop anything with spaces, punctuation beyond letters/digits, or too long
    if _VALID_MNEMONIC.match(m):
        return m
    return ""
