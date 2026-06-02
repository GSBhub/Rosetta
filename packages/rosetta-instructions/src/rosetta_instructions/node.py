"""Decode node: cursor-driven per-instruction extraction with pluggable output.

Reads from state:  meta, registers, db_path, settings_dict, out_dir,
                   processor_name, output_format, max_iterations,
                   inter_chunk_sleep, resume, debug_save_dir
Returns to state:  instructions, opcode_map, lang_dir, errors
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langsmith import traceable
from rosetta_schemas.state import PipelineState
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)


@traceable(run_type="chain", name="stage:decode", process_inputs=state_summary)
def decode_node(state: PipelineState) -> dict[str, Any]:
    """Cursor-driven decode loop with a pluggable output writer."""
    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Build Settings + Chroma wrapper
    # ------------------------------------------------------------------
    try:
        from docquery.config import Settings
        from rosetta_utils.chroma import get_chroma_wrapper

        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = state["db_path"]
        settings.vs = get_chroma_wrapper(settings.db_path, settings)
    except Exception as exc:
        log.exception("decode_node: failed to initialise settings/chroma")
        return {"instructions": [], "opcode_map": [], "lang_dir": None, "errors": [f"decode_node init: {exc}"]}

    # ------------------------------------------------------------------
    # 2. Resolve writer
    # ------------------------------------------------------------------
    output_format: str = state.get("output_format", "sla") or "sla"
    processor_name: str = state.get("processor_name", "Unknown") or "Unknown"
    out_dir = Path(state.get("out_dir", "./output") or "./output")

    from rosetta_generate_sla.writers.base import get_writer
    from rosetta_schemas.state import get_meta, get_registers

    meta = get_meta(state)
    registers = get_registers(state)

    if not meta:
        return {"instructions": [], "opcode_map": [], "lang_dir": None, "errors": ["decode_node: no meta in state"]}

    writer = get_writer(output_format)

    # ------------------------------------------------------------------
    # 3. Resume: seed seen-set from existing slaspec
    # ------------------------------------------------------------------
    seen: list[str] = []
    if state.get("resume"):
        seen = _seen_from_slaspec(out_dir, processor_name, meta.endian == "bi")
        if seen:
            log.info("decode_node resume: %d mnemonics already emitted", len(seen))

    # ------------------------------------------------------------------
    # 4. Open writer (writes header + aux files)
    # ------------------------------------------------------------------
    try:
        writer.open(
            meta=meta,
            registers=registers,
            processor_name=processor_name,
            out_dir=out_dir,
        )
    except Exception as exc:
        log.exception("decode_node: writer.open failed")
        return {"instructions": [], "opcode_map": [], "lang_dir": None, "errors": [f"decode_node writer.open: {exc}"]}

    # ------------------------------------------------------------------
    # 5. Build and run decode subgraph
    # ------------------------------------------------------------------
    from rosetta_instructions.decode_graph import DecodeState, build_decode_graph

    max_iterations: int | None = state.get("max_iterations")
    debug_save_dir = state.get("debug_save_dir")
    debug_path: Path | None = Path(debug_save_dir) / "decode_partial.jsonl" if debug_save_dir else None
    if debug_path and not state.get("resume"):
        debug_path.write_text("")

    initial: DecodeState = {
        "settings": settings,
        "meta": meta.model_dump(),
        "registers": [r.model_dump() for r in registers],
        "out_dir": str(out_dir),
        "processor_name": processor_name,
        "max_iterations": max_iterations,
        "inter_chunk_sleep": state.get("inter_chunk_sleep", 0.0),
        "debug_save_dir": debug_save_dir,
        "resume": state.get("resume", False),
        "last": seen[-1] if seen else None,
        "seen": seen,
        "current": None,
        "next": None,
        "iterations": len(seen),
        "stall_count": 0,
        "current_def": None,
        "opcode_map_rows": [],
        "written": [],
        "errors": [],
    }

    try:
        app = build_decode_graph(writer)
        final: DecodeState = app.invoke(initial)
    except Exception as exc:
        log.exception("decode_node: subgraph failed")
        writer.close()
        return {"instructions": [], "opcode_map": [], "lang_dir": str(writer.lang_dir) if writer.lang_dir else None, "errors": [f"decode_node subgraph: {exc}"]}

    # ------------------------------------------------------------------
    # 6. Finalise writer + collect results
    # ------------------------------------------------------------------
    try:
        writer.close()
    except Exception as exc:
        log.warning("decode_node: writer.close() error: %s", exc)

    subgraph_errors = list(final.get("errors") or [])
    written_mnemonics = list(final.get("written") or [])
    opcode_rows = list(final.get("opcode_map_rows") or [])

    log.info(
        "decode_node: wrote %d constructors, %d opcode rows, lang_dir=%s",
        len(written_mnemonics), len(opcode_rows), writer.lang_dir,
    )

    # Rebuild InstructionDef dicts for state (for downstream evaluate/validate).
    instructions_out: list[dict[str, Any]] = []
    if debug_path and debug_path.exists():
        for line in debug_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    instructions_out.append(json.loads(line))
                except Exception:
                    pass

    return {
        "instructions": instructions_out,
        "opcode_map": opcode_rows,
        "lang_dir": str(writer.lang_dir) if writer.lang_dir else None,
        "errors": errors + subgraph_errors,
    }


def _seen_from_slaspec(out_dir: Path, processor_name: str, bi_endian: bool) -> list[str]:
    """Parse existing .slaspec(s) for `:MNEMONIC` lines to seed the seen-set."""
    paths = []
    if bi_endian:
        paths = [
            out_dir / processor_name / "data" / "languages" / f"{processor_name}_le.slaspec",
            out_dir / processor_name / "data" / "languages" / f"{processor_name}_be.slaspec",
        ]
    else:
        paths = [out_dir / processor_name / "data" / "languages" / f"{processor_name}.slaspec"]

    seen: list[str] = []
    for path in paths:
        if path.exists():
            for m in re.findall(r"^:([A-Z][A-Z0-9_]*)\b", path.read_text(), re.MULTILINE):
                if m not in seen:
                    seen.append(m)
    return seen


# ---------------------------------------------------------------------------
# Legacy entry point kept for backward-compat / run-stage standalone use
# ---------------------------------------------------------------------------

def instructions_node(state: PipelineState) -> dict[str, Any]:
    """Alias for decode_node — kept so `rosetta run-stage instructions` still works."""
    return decode_node(state)
