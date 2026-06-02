"""Node 4b: Extract opcode map for opcode_table ISAs (Pass 4b).

For 'opcode_table' ISAs (6502/65816 family, Z80, M7700, etc.) this node
replaces the mnemonics + instructions passes with a direct opcode-map scan:
it queries each row of the 256-entry opcode table from the manual and
produces a list of OpcodeDef entries covering the full opcode space.

For other encoding styles this node is a no-op.

Reads from state:  meta, db_path, settings_dict
Returns to state:  opcode_map, mnemonics (deduplicated from opcode_map)
"""

from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable
from rosetta_schemas.state import PipelineState, get_meta
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)


@traceable(run_type="chain", name="stage:opcode_map", process_inputs=state_summary)
def opcode_map_node(state: PipelineState) -> dict[str, Any]:
    """Extract the opcode table for opcode_table ISAs; no-op otherwise."""
    meta = get_meta(state)
    if meta is None or meta.encoding_style != "opcode_table":
        log.info("opcode_map_node: encoding_style=%r — skipping",
                 meta.encoding_style if meta else "none")
        return {"opcode_map": [], "errors": []}

    from rosetta_opcode_map.scanner_graph import build_scanner_graph

    try:
        scanner = build_scanner_graph()
        result = scanner.invoke({
            "meta":             meta.model_dump(),
            "settings_dict":    state.get("settings_dict") or {},
            "db_path":          state["db_path"],
            "inter_chunk_sleep": float(state.get("inter_chunk_sleep") or 1.0),
            # initialise accumulator so the reducer has a baseline
            "scan_entries":    [],
            "candidate_chunks": [],
            "opcode_map":      [],
            "gaps":            [],
            "fill_iterations": 0,
        })

        opcode_map: list[dict] = result.get("opcode_map", [])
        mnemonics = sorted({
            e["mnemonic"] for e in opcode_map if e.get("mnemonic") != "UNK"
        })

        log.info(
            "opcode_map_node: %d entries, %d unique mnemonics",
            len(opcode_map), len(mnemonics),
        )
        return {"opcode_map": opcode_map, "mnemonics": mnemonics, "errors": []}

    except Exception as exc:
        log.error("opcode_map_node failed: %s", exc)
        return {"opcode_map": [], "errors": [f"opcode_map_node: {exc}"]}
