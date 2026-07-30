"""The instructions fan-in must be a true barrier.

`instructions` fans in from branches of different lengths (ingest→registers is
2 hops; meta→classify→mnemonics is 4). Without `defer=True` the short branch
schedules it early, so it *also* runs with 0 mnemonics — and `generate_sla`
consumes that empty result, rendering a plausible-looking module with no
constructors while the real pass-4 output is computed afterwards and discarded.

Pinned by tests because it depends on LangGraph scheduling semantics and
`langgraph` is pinned loosely (>=0.2): a dependency bump could silently
reintroduce the empty module.
"""

from __future__ import annotations

import pytest

from rosetta.graph import build_compiled_graph, build_graph

# Modules build_graph() imports its node callables from, in graph order.
_NODE_SOURCES = [
    ("rosetta_ingest.node", "ingest_node"),
    ("rosetta_meta.node", "meta_node"),
    ("rosetta_classify.node", "classify_node"),
    ("rosetta_registers.node", "registers_node"),
    ("rosetta_mnemonics.node", "mnemonics_node"),
    ("rosetta_opcode_map.node", "opcode_map_node"),
    ("rosetta_opcode_map.pcode_node", "opcode_map_pcode_node"),
    ("rosetta_instructions.node", "instructions_node"),
    ("rosetta_pcode.node", "pcode_node"),
    ("rosetta_generate_sla.node", "generate_sla_node"),
    ("rosetta_validate_sla.node", "validate_sla_node"),
    ("rosetta_evaluate_sla.node", "evaluate_sla_node"),
]


def test_instructions_node_is_deferred():
    node = build_graph().nodes["instructions"]
    assert getattr(node, "defer", False), (
        "instructions must be deferred, else the short registers branch fires it "
        "before mnemonics is populated and generate_sla renders an empty module"
    )


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace every node with a trivial stub; return the instructions call log."""
    import importlib

    calls: list[dict] = []

    def instructions(state):
        calls.append({"mnemonics": list(state.get("mnemonics") or [])})
        return {"instructions": []}

    outputs = {
        "meta_node": {"meta": {"name": "T"}},
        "registers_node": {"registers": [{"name": "R0"}]},
        "mnemonics_node": {"mnemonics": ["ADD", "SUB"]},
    }
    for mod_name, attr in _NODE_SOURCES:
        mod = importlib.import_module(mod_name)
        if attr == "instructions_node":
            monkeypatch.setattr(mod, attr, instructions)
        else:
            monkeypatch.setattr(mod, attr, lambda _s, _o=outputs.get(attr, {}): dict(_o))
    return calls


def test_instructions_runs_once_after_every_branch(stub_pipeline):
    """The barrier invariant: one invocation, and it sees all upstream output.

    Without defer the node *also* fires early with 0 mnemonics; asserting only
    on the final state would pass either way, so assert the call count too.
    """
    build_compiled_graph().invoke({"errors": []})

    assert len(stub_pipeline) == 1, (
        f"instructions ran {len(stub_pipeline)}x — the fan-in is not a barrier, "
        f"so an early empty run reached generate_sla. Saw: {stub_pipeline}"
    )
    assert stub_pipeline[0]["mnemonics"] == ["ADD", "SUB"]
