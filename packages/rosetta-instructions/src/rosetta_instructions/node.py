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
    # 3b. Pre-build mnemonic queue via Ghidra reference cross-filter
    # ------------------------------------------------------------------
    mnemonic_queue: list[str] = []
    reference_str = state.get("reference_slaspec")
    ghidra_home = state.get("ghidra_home")
    if reference_str and ghidra_home:
        mnemonic_queue = _build_reference_queue(settings, reference_str, ghidra_home)
    # If no reference or build failed, discover_node will scan DB on first call.

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

    max_iterations: int | None = state.get("max_iterations")

    # ------------------------------------------------------------------
    # 5. Dispatch a decode strategy on the ISA family (set by `classify`)
    # ------------------------------------------------------------------
    # opcode_table (CISC: 6502/Z80/M7700) walks the bounded opcode space; the
    # RISC/variable strategies walk a mnemonic cursor. The writer was opened
    # with meta, so it already knows whether to render the CISC template.
    if meta.encoding_style == "opcode_table":
        from rosetta_instructions.opcode_decode import run_opcode_scan

        try:
            opcode_map, scan_errors = run_opcode_scan(
                writer, settings, meta,
                max_iterations=max_iterations,
                inter_chunk_sleep=state.get("inter_chunk_sleep", 0.0),
            )
        except Exception as exc:
            log.exception("decode_node: opcode scan failed")
            scan_errors = [f"decode_node opcode scan: {exc}"]
            opcode_map = []
        try:
            writer.close()
        except Exception as exc:
            log.warning("decode_node: writer.close() error: %s", exc)
        log.info("decode_node: opcode_table scan wrote %d entries, lang_dir=%s",
                 len(opcode_map), writer.lang_dir)
        return {
            "instructions": [],
            "opcode_map": opcode_map,
            "lang_dir": str(writer.lang_dir) if writer.lang_dir else None,
            "errors": errors + scan_errors,
        }

    if meta.encoding_style == "variable_prefix":
        # x86-style multi-byte prefixes aren't modelled yet; the mnemonic cursor
        # still recovers the documented instruction set, just without per-prefix
        # opcode tables. Surface that limitation rather than failing silently.
        errors.append(
            "decode_node: variable_prefix (x86-style) prefix decoding not implemented — "
            "falling back to the mnemonic strategy"
        )

    # ------------------------------------------------------------------
    # 5b. Mnemonic-cursor strategy (fixed_word RISC, and variable_prefix fallback)
    # ------------------------------------------------------------------
    from rosetta_instructions.decode_graph import DecodeState, build_decode_graph

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
        "mnemonic_queue": mnemonic_queue,  # pre-built from reference or populated lazily
        "current_def": None,
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

    log.info(
        "decode_node: wrote %d constructors, lang_dir=%s",
        len(written_mnemonics), writer.lang_dir,
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
        "lang_dir": str(writer.lang_dir) if writer.lang_dir else None,
        "errors": errors + subgraph_errors,
    }


def _build_reference_queue(settings: Any, reference_str: str, ghidra_home: str) -> list[str]:
    """Return mnemonics from the Ghidra reference that also appear in the DB.

    This cross-filter ensures every queued token is a legitimate instruction AND
    is documented in the manual — making every iteration meaningful and eliminating
    false positives from raw text scanning.
    """
    try:
        from pathlib import Path as _Path
        from rosetta_evaluate_sla.sla.spec_loader import (
            extract_mnemonics,
            load_ghidra_reference,
            load_spec_text,
        )
        ref_path = load_ghidra_reference(_Path(ghidra_home), reference_str)
        ref_mnemonics = extract_mnemonics(load_spec_text(ref_path))
    except Exception as exc:
        log.warning("_build_reference_queue: failed to load reference '%s': %s", reference_str, exc)
        return []

    if not ref_mnemonics:
        return []

    from rosetta_instructions.discovery import scan_db_for_mnemonics
    queue = scan_db_for_mnemonics(settings, reference_filter=ref_mnemonics)
    log.info("_build_reference_queue: queued %d mnemonics (reference=%s)", len(queue), reference_str)
    return queue


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
