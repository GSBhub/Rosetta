"""LangGraph orchestration: wires all pipeline node packages into a compiled StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rosetta_schemas.state import PipelineState


def build_graph() -> StateGraph:
    from rosetta_evaluate_sla.node import evaluate_sla_node
    from rosetta_generate_sla.node import generate_sla_node
    from rosetta_ingest.node import ingest_node
    from rosetta_instructions.node import instructions_node
    from rosetta_meta.node import meta_node
    from rosetta_mnemonics.node import mnemonics_node
    from rosetta_pcode.node import pcode_node
    from rosetta_registers.node import registers_node
    from rosetta_validate_sla.node import validate_sla_node

    g = StateGraph(PipelineState)

    g.add_node("ingest", ingest_node)
    g.add_node("meta", meta_node)
    g.add_node("registers", registers_node)
    g.add_node("mnemonics", mnemonics_node)
    g.add_node("instructions", instructions_node)
    g.add_node("pcode", pcode_node)
    g.add_node("generate_sla", generate_sla_node)
    g.add_node("validate_sla", validate_sla_node)
    g.add_node("evaluate_sla", evaluate_sla_node)

    # ingest → fan-out to meta/registers/mnemonics in parallel
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "meta")
    g.add_edge("ingest", "registers")
    g.add_edge("ingest", "mnemonics")

    # fan-in: LangGraph waits for all three before firing instructions
    g.add_edge("meta", "instructions")
    g.add_edge("registers", "instructions")
    g.add_edge("mnemonics", "instructions")

    g.add_edge("instructions", "pcode")
    g.add_edge("pcode", "generate_sla")
    g.add_edge("generate_sla", "validate_sla")
    g.add_edge("validate_sla", "evaluate_sla")
    g.add_edge("evaluate_sla", END)

    return g


def build_compiled_graph():
    return build_graph().compile()
