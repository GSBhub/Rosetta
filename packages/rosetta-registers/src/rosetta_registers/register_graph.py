"""LangGraph cursor subgraph for register discovery.

Pattern mirrors the RISC instruction decode loop in rosetta_instructions:
  START → reg_discover → (reg_gather | END) → reg_gather → reg_emit
        → reg_advance → reg_discover   (loop)
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger(__name__)

_STALL_LIMIT = 5


class RegisterCursorState(TypedDict, total=False):
    settings: Any
    max_iterations: int | None

    # Cursor
    last: str | None
    seen: list[str]
    current: str | None
    next: str | None
    iterations: int
    stall_count: int

    # Tag-based enumeration: when tag_mode is True, registers are popped from
    # register_queue (docquery entity tags); otherwise the legacy LLM cursor runs.
    tag_mode: bool
    register_queue: list[str]

    # Per-iteration scratch
    current_def: dict[str, Any] | None

    # Accumulator — operator.add merges lists from parallel/looping nodes
    registers: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_cursor(state: RegisterCursorState) -> str:
    current = state.get("current")
    if not current:
        log.info("register cursor: exhausted (current=None)")
        return END

    seen = state.get("seen") or []
    if current.upper() in (s.upper() for s in seen):
        stall = (state.get("stall_count") or 0) + 1
        if stall >= _STALL_LIMIT:
            log.warning("register cursor: stall limit reached — stopping")
            return END

    max_iter = state.get("max_iterations")
    if max_iter is not None and (state.get("iterations") or 0) >= max_iter:
        log.info("register cursor: max_iterations=%d reached", max_iter)
        return END

    return "reg_gather"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _discover_node(state: RegisterCursorState) -> dict[str, Any]:
    # Tag mode: pop the next register from the pre-built queue (docquery tags) —
    # the deterministic equivalent of cursor_enumerate("register").
    if state.get("tag_mode"):
        queue = list(state.get("register_queue") or [])
        seen_upper = {s.upper() for s in (state.get("seen") or [])}
        while queue and queue[0].upper() in seen_upper:
            queue.pop(0)
        if not queue:
            return {"current": None, "next": None, "register_queue": []}
        current = queue[0]
        next_ = queue[1] if len(queue) > 1 else None
        return {"current": current, "next": next_, "register_queue": queue[1:]}

    # Fallback: legacy LLM cursor (DB has no register tags).
    from rosetta_registers.cursor import discover_next_register
    current, next_ = discover_next_register(
        state.get("last"),
        list(state.get("seen") or []),
        state["settings"],
    )
    return {"current": current, "next": next_}


def _gather_node(state: RegisterCursorState) -> dict[str, Any]:
    from rosetta_registers.cursor import gather_register
    name = state.get("current") or ""
    reg = gather_register(name, state["settings"])
    return {"current_def": reg.model_dump()}


def _emit_node(state: RegisterCursorState) -> dict[str, Any]:
    raw = state.get("current_def") or {}
    return {"registers": [raw]}


def _advance_node(state: RegisterCursorState) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_register_graph():
    g = StateGraph(RegisterCursorState)

    g.add_node("reg_discover", _discover_node)
    g.add_node("reg_gather",   _gather_node)
    g.add_node("reg_emit",     _emit_node)
    g.add_node("reg_advance",  _advance_node)

    g.add_edge(START, "reg_discover")
    g.add_conditional_edges(
        "reg_discover",
        _route_cursor,
        {"reg_gather": "reg_gather", END: END},
    )
    g.add_edge("reg_gather",  "reg_emit")
    g.add_edge("reg_emit",    "reg_advance")
    g.add_edge("reg_advance", "reg_discover")

    return g.compile()
