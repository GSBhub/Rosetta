"""LangGraph orchestration: wires all pipeline node packages into a compiled StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rosetta_schemas.state import PipelineState


def build_graph() -> StateGraph:
    from rosetta_classify.node import classify_node
    from rosetta_evaluate_sla.node import evaluate_sla_node
    from rosetta_generate_sla.node import generate_sla_node
    from rosetta_ingest.node import ingest_node
    from rosetta_instructions.node import decode_node
    from rosetta_meta.node import meta_node
    from rosetta_registers.node import registers_node
    from rosetta_validate_sla.node import validate_sla_node

    g = StateGraph(PipelineState)

    g.add_node("ingest",       ingest_node)
    g.add_node("meta",         meta_node)
    g.add_node("classify",     classify_node)
    g.add_node("registers",    registers_node)
    g.add_node("decode",       decode_node)
    # generate_sla kept as an idempotent finaliser: returns lang_dir unchanged if
    # decode already set it, so `run-stage generate` still works standalone.
    g.add_node("generate_sla", generate_sla_node)
    g.add_node("validate_sla", validate_sla_node)
    g.add_node("evaluate_sla", evaluate_sla_node)

    # Serial chain: each stage feeds the next.  A parallel fork into decode caused
    # LangGraph to invoke decode_node twice (once per incoming edge), so we keep
    # a single predecessor for decode.
    g.add_edge(START,       "ingest")
    g.add_edge("ingest",    "meta")
    g.add_edge("meta",      "classify")
    g.add_edge("classify",  "registers")
    g.add_edge("registers", "decode")

    # decode handles both RISC and CISC internally; downstream stages are unchanged.
    g.add_edge("decode",       "generate_sla")
    g.add_edge("generate_sla", "validate_sla")
    g.add_edge("validate_sla", "evaluate_sla")
    g.add_edge("evaluate_sla", END)

    return g


def build_compiled_graph():
    return build_graph().compile()
