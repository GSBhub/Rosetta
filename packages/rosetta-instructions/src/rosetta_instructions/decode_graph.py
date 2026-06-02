"""LangGraph decode subgraph: dispatch → RISC cursor loop OR CISC batch parse."""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger(__name__)

# Maximum consecutive stalls (discovery returns a mnemonic already in seen) before giving up.
_STALL_LIMIT = 5


# ---------------------------------------------------------------------------
# Internal subgraph state  (NOT part of PipelineState — stays in the closure)
# ---------------------------------------------------------------------------

class DecodeState(TypedDict, total=False):
    # Inputs threaded from parent PipelineState
    settings: Any                # configured docquery Settings object
    meta: dict[str, Any]
    registers: list[dict[str, Any]]
    out_dir: str
    processor_name: str
    max_iterations: int | None
    inter_chunk_sleep: float
    debug_save_dir: str | None
    resume: bool

    # Cursor state (RISC path)
    last: str | None             # last successfully emitted mnemonic
    seen: list[str]              # all mnemonics processed so far (for stall detection)
    current: str | None          # mnemonic being decoded this iteration
    next: str | None             # lookahead mnemonic (disambiguation context)
    iterations: int
    stall_count: int             # consecutive stalls

    # Per-iteration scratch
    current_def: dict[str, Any] | None  # InstructionDef.model_dump()

    # CISC path scratch
    opcode_map_rows: list[dict[str, Any]]

    # Accumulators
    written: Annotated[list[str], operator.add]   # mnemonics successfully emitted
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_family(state: DecodeState) -> str:
    meta = state.get("meta") or {}
    if meta.get("encoding_style") == "opcode_table":
        return "cisc"
    return "risc"


def _route_cursor(state: DecodeState) -> str:
    """Return 'gather' to continue or END to stop the RISC loop."""
    current = state.get("current")
    if not current:
        log.info("decode_graph: cursor exhausted (current=None) — stopping")
        return END

    seen = state.get("seen") or []
    if current.upper() in (s.upper() for s in seen):
        stall = (state.get("stall_count") or 0) + 1
        log.debug("decode_graph: stall %d/%d on %r", stall, _STALL_LIMIT, current)
        if stall >= _STALL_LIMIT:
            log.warning("decode_graph: stall limit reached — stopping")
            return END
    else:
        pass  # stall_count reset happens in advance_node

    max_iter = state.get("max_iterations")
    iterations = state.get("iterations", 0)
    if max_iter is not None and iterations >= max_iter:
        log.info("decode_graph: max_iterations=%d reached — stopping", max_iter)
        return END

    return "risc_gather"


# ---------------------------------------------------------------------------
# Node factories (take the writer via closure)
# ---------------------------------------------------------------------------

def _make_nodes(writer: Any) -> dict[str, Callable]:

    def dispatch_node(state: DecodeState) -> dict[str, Any]:
        # Nothing to do — routing is handled by conditional edges.
        return {}

    def risc_discover_node(state: DecodeState) -> dict[str, Any]:
        from rosetta_instructions.discovery import discover_next

        settings = state["settings"]
        last = state.get("last")
        seen = list(state.get("seen") or [])

        current, next_ = discover_next(last, seen, settings)
        return {"current": current, "next": next_}

    def risc_gather_node(state: DecodeState) -> dict[str, Any]:
        from rosetta_instructions.gather import enrich_pcode, gather_instruction

        settings = state["settings"]
        current = state.get("current") or ""
        next_ = state.get("next")

        instr = gather_instruction(current, next_, settings)
        instr = enrich_pcode(instr, settings)
        return {"current_def": instr.model_dump()}

    def risc_validate_node(state: DecodeState) -> dict[str, Any]:
        from rosetta_instructions.validate import validate_and_fix
        from rosetta_schemas.models import InstructionDef

        raw = state.get("current_def") or {}
        try:
            instr = InstructionDef.model_validate(raw)
        except Exception as exc:
            log.warning("validate: Pydantic error for %r: %s — building stub", raw.get("mnemonic"), exc)
            mnemonic = raw.get("mnemonic") or state.get("current") or "UNKNOWN"
            instr = InstructionDef(mnemonic=mnemonic, encoding_bits=32, semantics="Extraction failed.")

        instr, issues = validate_and_fix(instr)
        errors = [f"validate({instr.mnemonic}): {iss}" for iss in issues]
        return {"current_def": instr.model_dump(), "errors": errors}

    def risc_emit_node(state: DecodeState) -> dict[str, Any]:
        from rosetta_schemas.models import InstructionDef

        raw = state.get("current_def") or {}
        try:
            instr = InstructionDef.model_validate(raw)
            writer.write_instruction(instr)
            mnemonic = instr.mnemonic
        except Exception as exc:
            mnemonic = raw.get("mnemonic") or state.get("current") or "UNKNOWN"
            log.warning("emit failed for %r: %s", mnemonic, exc)
            return {"errors": [f"emit({mnemonic}): {exc}"]}

        return {"written": [mnemonic]}

    def risc_advance_node(state: DecodeState) -> dict[str, Any]:
        current = (state.get("current") or "").upper()
        seen = list(state.get("seen") or [])
        stall = state.get("stall_count", 0)

        if current and current not in (s.upper() for s in seen):
            seen.append(current)
            stall = 0  # reset stall counter on genuine progress
        else:
            stall += 1

        return {
            "last": state.get("current"),
            "seen": seen,
            "iterations": (state.get("iterations") or 0) + 1,
            "stall_count": stall,
            "current": None,
            "current_def": None,
        }

    def cisc_parse_node(state: DecodeState) -> dict[str, Any]:
        """Cursor over the opcode space (0x00–0xFF per table), one entry at a time.

        Iterates the base table then each prefix table declared in
        meta.opcode_prefixes, calling gather_opcode_def() for every byte so
        each extraction appears as its own named LangSmith span.
        """
        from rosetta_instructions.opcode_cursor import gather_opcode_def
        from rosetta_schemas.models import ISAMeta, OpcodeDef

        meta_raw = state.get("meta") or {}
        settings = state["settings"]
        max_iter = state.get("max_iterations")

        try:
            meta = ISAMeta.model_validate(meta_raw)
        except Exception as exc:
            return {"errors": [f"cisc_parse: schema error: {exc}"]}

        # Tables to scan: base (prefix=None) + any declared prefix tables.
        tables: list[int | None] = [None] + list(meta.opcode_prefixes or [])
        opcode_map: list[OpcodeDef] = []
        errors: list[str] = []
        total = 0

        for prefix in tables:
            label = f"0x{prefix:02X}/" if prefix is not None else "base"
            log.info("cisc_parse: scanning %s opcode table", label)
            for byte in range(256):
                if max_iter is not None and total >= max_iter:
                    log.info("cisc_parse: max_iterations=%d reached", max_iter)
                    break
                try:
                    entry = gather_opcode_def(byte, prefix, settings)
                    if entry and entry.mnemonic.upper() != "UNK":
                        opcode_map.append(entry)
                except Exception as exc:
                    errors.append(f"cisc_parse 0x{byte:02X}: {exc}")
                total += 1
            else:
                continue
            break  # inner break propagates out when max_iter hit

        log.info(
            "cisc_parse: %d known entries across %d table(s)",
            len(opcode_map), len(tables),
        )

        try:
            writer.write_opcode_table(opcode_map)
        except Exception as exc:
            errors.append(f"cisc_parse write_opcode_table: {exc}")

        opcode_rows = [e.model_dump() for e in opcode_map]
        written = [f"{e.mnemonic}:0x{e.opcode:02X}" for e in opcode_map]
        return {"opcode_map_rows": opcode_rows, "written": written, "errors": errors}

    return {
        "dispatch": dispatch_node,
        "risc_discover": risc_discover_node,
        "risc_gather": risc_gather_node,
        "risc_validate": risc_validate_node,
        "risc_emit": risc_emit_node,
        "risc_advance": risc_advance_node,
        "cisc_parse": cisc_parse_node,
    }


def _settings_to_dict(settings: Any) -> dict[str, Any]:
    """Serialize a docquery Settings object back to a plain dict for sub-invocations."""
    try:
        import dataclasses
        if dataclasses.is_dataclass(settings):
            d = dataclasses.asdict(settings)
            # vs and db_client are live objects — drop them; the sub-nodes rebuild them.
            d.pop("vs", None)
            d.pop("db_client", None)
            return d
    except Exception:
        pass
    return {}


def _merge(state: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    for k, v in partial.items():
        if k == "errors":
            result["errors"] = list(result.get("errors") or []) + list(v or [])
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_decode_graph(writer: Any):
    """Compile and return the decode subgraph with *writer* bound via closure."""
    nodes = _make_nodes(writer)

    g = StateGraph(DecodeState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "dispatch")
    g.add_conditional_edges(
        "dispatch",
        _route_family,
        {"risc": "risc_discover", "cisc": "cisc_parse"},
    )

    # RISC cursor loop
    g.add_conditional_edges(
        "risc_discover",
        _route_cursor,
        {"risc_gather": "risc_gather", END: END},
    )
    g.add_edge("risc_gather", "risc_validate")
    g.add_edge("risc_validate", "risc_emit")
    g.add_edge("risc_emit", "risc_advance")
    g.add_edge("risc_advance", "risc_discover")  # loop back

    # CISC batch path
    g.add_edge("cisc_parse", END)

    return g.compile()
