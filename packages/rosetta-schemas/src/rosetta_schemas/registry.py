"""Simple node registry for the Rosetta LangGraph pipeline.

Each backend package calls register_node() in its __init__.py.
The orchestrator's build_graph() iterates get_registry() to wire
the graph without importing packages by name.
"""

from __future__ import annotations

from typing import Any, Callable

from rosetta_schemas.state import PipelineState

NodeFn = Callable[[PipelineState], dict[str, Any]]

# (node_fn, predecessors, successors)
NodeDef = tuple[NodeFn, list[str], list[str]]

_REGISTRY: dict[str, NodeDef] = {}


def register_node(
    name: str,
    fn: NodeFn,
    predecessors: list[str],
    successors: list[str],
) -> None:
    """Register a LangGraph node so the orchestrator can wire it automatically."""
    _REGISTRY[name] = (fn, list(predecessors), list(successors))


def get_registry() -> dict[str, NodeDef]:
    return dict(_REGISTRY)
