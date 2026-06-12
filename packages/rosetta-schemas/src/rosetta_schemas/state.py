from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from rosetta_schemas.models import ISAMeta, ISASpec, InstructionDef, OpcodeDef, RegisterDef


class PipelineState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    source_path: str            # path to PDF or source directory for ingest
    db_path: str                # path to ChromaDB directory (set in settings_dict too)
    settings_dict: dict[str, Any]   # serialized DocSettings fields (including db_path)
    processor_name: str         # e.g. "ARM_v7"
    out_dir: str                # output root for generated files
    ghidra_home: str            # absolute path to Ghidra installation

    # ── Extraction config ─────────────────────────────────────────────────────
    max_concurrent: int
    max_instructions: int | None
    max_iterations: int | None   # cursor guard for decode loop (defaults to auto from count query)
    output_format: str           # "sla" (default); selects the InstructionWriter implementation
    memory_warn_gb: float
    chunk_size: int | None
    inter_chunk_sleep: float
    resume: bool
    debug_save_dir: str | None

    # ── Per-pass outputs (Pydantic models serialized as model_dump() dicts) ───
    meta: dict[str, Any] | None
    registers: list[dict[str, Any]]
    instructions: list[dict[str, Any]]
    opcode_map: list[dict[str, Any]]    # OpcodeDef rows for opcode_table ISAs

    # ── Downstream stage outputs ──────────────────────────────────────────────
    lang_dir: str | None        # path to generated languages/ directory
    compile_ok: bool | None
    compile_errors: list[str]
    headless_ok: bool | None

    # ── Evaluation outputs ────────────────────────────────────────────────────
    reference_slaspec: str | None
    instruction_coverage: float | None
    register_overlap: float | None

    # ── Error accumulation ────────────────────────────────────────────────────
    # Annotated with operator.add so LangGraph merges lists from parallel fan-out nodes.
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Accessor helpers — centralize Pydantic deserialization from state dicts
# ---------------------------------------------------------------------------

def get_meta(state: PipelineState) -> ISAMeta | None:
    raw = state.get("meta")
    return ISAMeta.model_validate(raw) if raw else None


def get_registers(state: PipelineState) -> list[RegisterDef]:
    return [RegisterDef.model_validate(r) for r in state.get("registers", [])]


def get_instructions(state: PipelineState) -> list[InstructionDef]:
    return [InstructionDef.model_validate(i) for i in state.get("instructions", [])]


def get_opcode_map(state: PipelineState) -> list[OpcodeDef]:
    return [OpcodeDef.model_validate(o) for o in state.get("opcode_map", [])]


def get_isa_spec(state: PipelineState) -> ISASpec | None:
    meta = get_meta(state)
    if not meta:
        return None
    return ISASpec(
        meta=meta,
        registers=get_registers(state),
        instructions=get_instructions(state),
        opcode_map=get_opcode_map(state),
    )
