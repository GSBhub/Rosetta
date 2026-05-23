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

from rosetta_schemas.state import PipelineState, get_meta

log = logging.getLogger(__name__)


def opcode_map_node(state: PipelineState) -> dict[str, Any]:
    """Extract the opcode table for opcode_table ISAs; no-op otherwise."""
    meta = get_meta(state)
    if meta is None or meta.encoding_style != "opcode_table":
        log.info("opcode_map_node: encoding_style=%r — skipping",
                 meta.encoding_style if meta else "none")
        return {"opcode_map": [], "errors": []}

    from docquery.config import Settings
    from rosetta_utils.chroma import get_chroma_wrapper

    from rosetta_opcode_map.extractor import extract_opcode_map

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_wrapper(settings.db_path, settings)

        sleep = float(state.get("inter_chunk_sleep") or 1.0)
        entries = extract_opcode_map(settings, prefix=None, inter_row_sleep=sleep)

        # Derive mnemonics list as a side effect so downstream nodes that
        # expect 'mnemonics' still see a populated list.
        mnemonics = sorted({e.mnemonic for e in entries if e.mnemonic != "UNK"})

        log.info(
            "opcode_map_node: %d opcode entries, %d unique mnemonics",
            len(entries),
            len(mnemonics),
        )
        return {
            "opcode_map": [e.model_dump() for e in entries],
            "mnemonics": mnemonics,
            "errors": [],
        }

    except Exception as exc:
        log.error("opcode_map_node failed: %s", exc)
        return {"opcode_map": [], "errors": [f"opcode_map_node: {exc}"]}
