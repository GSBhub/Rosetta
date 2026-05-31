"""Node: opcode_map_pcode — populate pcode_body on each OpcodeDef for CISC ISAs.

Reads from state:  opcode_map, meta, registers, settings_dict
Returns to state:  opcode_map (entries updated with pcode_body), errors

No-op when meta.encoding_style != 'opcode_table'.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def opcode_map_pcode_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate SLEIGH pcode bodies for all opcode_map entries."""
    meta = state.get("meta") or {}
    encoding_style = meta.get("encoding_style", "fixed_word")

    if encoding_style != "opcode_table":
        log.info("opcode_map_pcode_node: skip (encoding_style=%r)", encoding_style)
        return {"opcode_map": state.get("opcode_map", []), "errors": []}

    from docquery.config import Settings
    from rosetta_opcode_map.pcode_generator import generate_pcode_bodies

    errors: list[str] = []
    opcode_map: list[dict[str, Any]] = list(state.get("opcode_map") or [])

    if not opcode_map:
        log.warning("opcode_map_pcode_node: opcode_map is empty — nothing to do")
        return {"opcode_map": opcode_map, "errors": errors}

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        registers = state.get("registers") or []
        reg_names = [r.get("name", "") for r in registers if r.get("name")]
        word_size_bits = meta.get("word_size_bits", 16)
        isa_name = meta.get("name", "unknown ISA")
        sleep = state.get("inter_chunk_sleep", 0.5)

        pcode_map = generate_pcode_bodies(
            opcode_map,
            isa_name=isa_name,
            register_names=reg_names,
            word_size_bits=word_size_bits,
            settings=settings,
            inter_entry_sleep=sleep,
        )

        from rosetta_opcode_map.pcode_generator import _normalize_mnemonic

        hits = 0
        for entry in opcode_map:
            raw_mn = entry.get("mnemonic", "UNK")
            mn = _normalize_mnemonic(raw_mn)
            if mn == "UNK":
                continue
            key = (mn, entry.get("mode", "imp"))
            body = pcode_map.get(key, "")
            if body:
                entry["pcode_body"] = body
                hits += 1

        log.info("opcode_map_pcode_node: %d entries received pcode_body", hits)

    except Exception as exc:
        log.exception("opcode_map_pcode_node failed")
        errors.append(f"opcode_map_pcode_node: {exc}")

    return {"opcode_map": opcode_map, "errors": errors}
