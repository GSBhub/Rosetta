"""LangGraph decode subgraph: cursor-driven per-instruction loop for all ISA families.

Flow:
  START → discover → [fill | END] → fill → validate → decode → advance → discover (loop)

Termination triggers (checked in _route_cursor after discover):
  - current is None (manual exhausted)
  - current is already in seen _STALL_LIMIT consecutive times
  - iterations >= max_iterations
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger(__name__)

_STALL_LIMIT = 5


# ---------------------------------------------------------------------------
# Internal subgraph state  (NOT part of PipelineState — stays in the closure)
# ---------------------------------------------------------------------------

class DecodeState(TypedDict, total=False):
    # Inputs threaded from parent PipelineState
    settings: Any
    meta: dict[str, Any]
    registers: list[dict[str, Any]]
    out_dir: str
    processor_name: str
    max_iterations: int | None
    inter_chunk_sleep: float
    debug_save_dir: str | None
    resume: bool

    # Cursor state
    last: str | None             # last successfully emitted mnemonic
    seen: list[str]              # all mnemonics processed so far
    current: str | None          # mnemonic being decoded this iteration
    next: str | None             # lookahead mnemonic (disambiguation context for fill)
    iterations: int
    stall_count: int
    mnemonic_queue: list[str]    # remaining mnemonics to process (from DB scan)

    # Per-iteration scratch
    current_def: dict[str, Any] | None  # InstructionDef.model_dump()

    # Accumulators
    written: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_cursor(state: DecodeState) -> str:
    """Return 'fill' to continue the loop or END to stop."""
    current = state.get("current")
    if not current:
        log.info("decode_graph: cursor exhausted (current=None) — stopping")
        return END

    seen = state.get("seen") or []
    if current.upper() in (s.upper() for s in seen):
        stall = (state.get("stall_count") or 0) + 1
        log.debug("decode_graph: stall %d/%d on %r", stall, _STALL_LIMIT, current)
        if stall >= _STALL_LIMIT:
            log.warning("decode_graph: stall limit reached — stopping")
            return END

    max_iter = state.get("max_iterations")
    iterations = state.get("iterations", 0)
    if max_iter is not None and iterations >= max_iter:
        log.info("decode_graph: max_iterations=%d reached — stopping", max_iter)
        return END

    return "fill"


# ---------------------------------------------------------------------------
# Node factory (writer bound via closure so state stays JSON-serializable)
# ---------------------------------------------------------------------------

def _make_nodes(writer: Any) -> dict[str, Callable]:

    def discover_node(state: DecodeState) -> dict[str, Any]:
        """Pop the next mnemonic from the queue; build the queue on first call via DB scan."""
        from rosetta_instructions.discovery import scan_db_for_mnemonics

        queue = list(state.get("mnemonic_queue") or [])
        seen_upper = {s.upper() for s in (state.get("seen") or [])}

        if not queue:
            if state.get("last") is None:
                # First call — scan the entire DB for mnemonic tokens.
                queue = scan_db_for_mnemonics(state["settings"])
            if not queue:
                return {"current": None, "next": None, "mnemonic_queue": []}

        # Skip items already processed in a previous session (resume support).
        while queue and queue[0].upper() in seen_upper:
            queue.pop(0)

        if not queue:
            return {"current": None, "next": None, "mnemonic_queue": []}

        current = queue[0]
        next_ = queue[1] if len(queue) > 1 else None
        # Pop current; advance_node will add it to seen after decode.
        return {"current": current, "next": next_, "mnemonic_queue": queue[1:]}

    def fill_node(state: DecodeState) -> dict[str, Any]:
        """Extract the full InstructionDef for *current* from the vector store."""
        from rosetta_instructions.gather import enrich_pcode, gather_instruction

        current = state.get("current") or ""
        next_ = state.get("next")

        instr = gather_instruction(current, next_, state["settings"])
        instr = enrich_pcode(instr, state["settings"])
        return {"current_def": instr.model_dump()}

    def validate_node(state: DecodeState) -> dict[str, Any]:
        """Validate and fix the InstructionDef; always produces a safe stub on failure."""
        from rosetta_instructions.validate import validate_and_fix
        from rosetta_schemas.models import InstructionDef

        raw = state.get("current_def") or {}
        try:
            instr = InstructionDef.model_validate(raw)
        except Exception as exc:
            mnemonic = raw.get("mnemonic") or state.get("current") or "UNKNOWN"
            log.warning("validate: Pydantic error for %r: %s — building stub", mnemonic, exc)
            instr = InstructionDef(mnemonic=mnemonic, encoding_bits=32, semantics="Extraction failed.")

        instr, issues = validate_and_fix(instr)
        errors = [f"validate({instr.mnemonic}): {iss}" for iss in issues]
        return {"current_def": instr.model_dump(), "errors": errors}

    def decode_node_fn(state: DecodeState) -> dict[str, Any]:
        """Decode the validated InstructionDef into the output format (SLEIGH/PCode)."""
        from rosetta_schemas.models import InstructionDef

        raw = state.get("current_def") or {}
        try:
            instr = InstructionDef.model_validate(raw)
            writer.write_instruction(instr)
            return {"written": [instr.mnemonic]}
        except Exception as exc:
            mnemonic = raw.get("mnemonic") or state.get("current") or "UNKNOWN"
            log.warning("decode: emit failed for %r: %s", mnemonic, exc)
            return {"errors": [f"decode({mnemonic}): {exc}"]}

    def advance_node(state: DecodeState) -> dict[str, Any]:
        """Advance the cursor: record current as seen, increment iteration counter."""
        current = (state.get("current") or "").upper()
        seen = list(state.get("seen") or [])
        stall = state.get("stall_count", 0)

        if current and current not in (s.upper() for s in seen):
            seen.append(current)
            stall = 0
        else:
            stall += 1

        return {
            "last": state.get("current"),
            "seen": seen,
            "iterations": (state.get("iterations") or 0) + 1,
            "stall_count": stall,
            "current": None,
            "current_def": None,
        }

    return {
        "discover": discover_node,
        "fill":     fill_node,
        "validate": validate_node,
        "decode":   decode_node_fn,
        "advance":  advance_node,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_decode_graph(writer: Any):
    """Compile and return the decode subgraph with *writer* bound via closure."""
    nodes = _make_nodes(writer)

    g = StateGraph(DecodeState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    # Entry point → discover; conditional gate before fill
    g.add_edge(START, "discover")
    g.add_conditional_edges(
        "discover",
        _route_cursor,
        {"fill": "fill", END: END},
    )

    # Per-instruction loop: fill → validate → decode → advance → discover
    g.add_edge("fill",     "validate")
    g.add_edge("validate", "decode")
    g.add_edge("decode",   "advance")
    g.add_edge("advance",  "discover")

    return g.compile()
