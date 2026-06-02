"""Node 2: Extract register file via cursor-driven LangGraph loop.

Reads from state:  db_path, settings_dict, max_iterations
Returns to state:  registers, errors
"""

from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable
from rosetta_schemas.state import PipelineState
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)


@traceable(run_type="chain", name="stage:registers", process_inputs=state_summary)
def registers_node(state: PipelineState) -> dict[str, Any]:
    """Discover registers one-at-a-time via the cursor LangGraph loop."""
    from docquery.config import Settings
    from rosetta_registers.register_graph import (
        RegisterCursorState,
        build_register_graph,
    )
    from rosetta_utils.chroma import get_chroma_wrapper

    errors: list[str] = []

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_wrapper(settings.db_path, settings)
    except Exception as exc:
        log.exception("registers_node: settings/chroma init failed")
        return {"registers": [], "errors": [f"registers_node init: {exc}"]}

    max_iterations: int | None = state.get("max_iterations")

    initial: RegisterCursorState = {
        "settings": settings,
        "max_iterations": max_iterations,
        "last": None,
        "seen": [],
        "current": None,
        "next": None,
        "iterations": 0,
        "stall_count": 0,
        "current_def": None,
        "registers": [],
        "errors": [],
    }

    try:
        app = build_register_graph()
        final: RegisterCursorState = app.invoke(initial)
    except Exception as exc:
        log.exception("registers_node: cursor graph failed")
        return {"registers": [], "errors": [f"registers_node: {exc}"]}

    registers = list(final.get("registers") or [])
    errors.extend(final.get("errors") or [])

    log.info("registers_node: discovered %d registers", len(registers))
    return {"registers": registers, "errors": errors}
