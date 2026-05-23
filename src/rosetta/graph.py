"""LangGraph orchestration: wires all pipeline node packages into a compiled StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rosetta_schemas.state import PipelineState


def build_graph() -> StateGraph:
    from rosetta_classify.node import classify_node
    from rosetta_evaluate_sla.node import evaluate_sla_node
    from rosetta_generate_sla.node import generate_sla_node
    from rosetta_ingest.node import ingest_node
    from rosetta_instructions.node import instructions_node
    from rosetta_meta.node import meta_node
    from rosetta_mnemonics.node import mnemonics_node
    from rosetta_opcode_map.node import opcode_map_node
    from rosetta_opcode_map.pcode_node import opcode_map_pcode_node
    from rosetta_pcode.node import pcode_node
    from rosetta_registers.node import registers_node
    from rosetta_validate_sla.node import validate_sla_node

    g = StateGraph(PipelineState)

    g.add_node("ingest",       ingest_node)
    g.add_node("meta",         meta_node)
    g.add_node("classify",     classify_node)
    g.add_node("registers",    registers_node)
    g.add_node("mnemonics",    mnemonics_node)
    g.add_node("opcode_map",       opcode_map_node)
    g.add_node("opcode_map_pcode", opcode_map_pcode_node)
    g.add_node("instructions",     instructions_node)
    g.add_node("pcode",        pcode_node)
    g.add_node("generate_sla", generate_sla_node)
    g.add_node("validate_sla", validate_sla_node)
    g.add_node("evaluate_sla", evaluate_sla_node)

    # ingest → meta (serial: classify needs meta.encoding_style)
    # ingest → registers (parallel: doesn't depend on encoding_style)
    g.add_edge(START,    "ingest")
    g.add_edge("ingest", "meta")
    g.add_edge("ingest", "registers")

    # meta → classify → parallel extraction fan-out
    # Both mnemonics and opcode_map always run; each checks encoding_style
    # and is a no-op when not applicable.
    g.add_edge("meta",     "classify")
    g.add_edge("classify", "mnemonics")
    g.add_edge("classify", "opcode_map")

    # opcode_map_pcode enriches CISC entries with pcode_body before the fan-in.
    # For fixed_word ISAs this node is a no-op (passes opcode_map through unchanged).
    g.add_edge("opcode_map",       "opcode_map_pcode")

    # fan-in at instructions: waits for registers, mnemonics, opcode_map_pcode
    g.add_edge("registers",        "instructions")
    g.add_edge("mnemonics",        "instructions")
    g.add_edge("opcode_map_pcode", "instructions")

    g.add_edge("instructions", "pcode")
    g.add_edge("pcode",        "generate_sla")
    g.add_edge("generate_sla", "validate_sla")
    g.add_edge("validate_sla", "evaluate_sla")
    g.add_edge("evaluate_sla", END)

    return g


def build_compiled_graph():
    return build_graph().compile()
