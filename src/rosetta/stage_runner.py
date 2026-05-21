"""Checkpointed per-node pipeline runner for staged verification.

Each stage calls exactly one LangGraph node function in isolation, merges
its output into a persistent checkpoint PipelineState dict, and dumps a
per-stage snapshot for inspection.  Merge semantics match LangGraph:
the 'errors' key extends (operator.add), every other key overwrites.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

# Each entry: (node_import_fn, required_input_keys, summary_fn)
# node_import_fn() → the callable node function (lazy import so CLI starts fast)

def _import_ingest():
    from rosetta_ingest.node import ingest_node
    return ingest_node

def _import_meta():
    from rosetta_meta.node import meta_node
    return meta_node

def _import_registers():
    from rosetta_registers.node import registers_node
    return registers_node

def _import_mnemonics():
    from rosetta_mnemonics.node import mnemonics_node
    return mnemonics_node

def _import_instructions():
    from rosetta_instructions.node import instructions_node
    return instructions_node

def _import_pcode():
    from rosetta_pcode.node import pcode_node
    return pcode_node

def _import_generate():
    from rosetta_generate_sla.node import generate_sla_node
    return generate_sla_node

def _import_validate():
    from rosetta_validate_sla.node import validate_sla_node
    return validate_sla_node

def _import_evaluate():
    from rosetta_evaluate_sla.node import evaluate_sla_node
    return evaluate_sla_node


# (node_import, required_keys_for_prereq_check)
STAGE_REGISTRY: dict[str, tuple[Callable, list[str]]] = {
    "ingest":       (_import_ingest,       ["source_path", "db_path"]),
    "meta":         (_import_meta,         ["db_path"]),
    "registers":    (_import_registers,    ["db_path"]),
    "mnemonics":    (_import_mnemonics,    ["db_path"]),
    "instructions": (_import_instructions, ["mnemonics", "db_path"]),
    "pcode":        (_import_pcode,        ["instructions"]),
    "generate":     (_import_generate,     ["meta", "processor_name", "out_dir"]),
    "validate":     (_import_validate,     ["lang_dir", "ghidra_home"]),
    "evaluate":     (_import_evaluate,     ["lang_dir", "reference_slaspec"]),
}

STAGE_ORDER = list(STAGE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# State merge  (mirrors LangGraph operator.add for errors, overwrite for rest)
# ---------------------------------------------------------------------------

def merge(state: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    for k, v in partial.items():
        if k == "errors":
            result["errors"] = list(result.get("errors") or []) + list(v or [])
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------

def check_prereqs(stage: str, state: dict[str, Any]) -> None:
    """Raise ValueError with a clear message if required keys are absent/empty."""
    _, required = STAGE_REGISTRY[stage]
    for key in required:
        val = state.get(key)
        if val is None or val == [] or val == "":
            # Find the stage that produces this key so we give an actionable hint
            producer = _find_producer(key)
            hint = f" (run stage '{producer}' first)" if producer else ""
            raise ValueError(
                f"Stage '{stage}' requires '{key}' in state but it is missing or empty{hint}."
            )


def _find_producer(key: str) -> str | None:
    """Return the stage name that writes `key`, for error messages."""
    _produces: dict[str, str] = {
        "meta": "meta",
        "registers": "registers",
        "mnemonics": "mnemonics",
        "instructions": "instructions",
        "lang_dir": "generate",
        "ghidra_home": "initial state (pass --checkpoint with ghidra_home set)",
        "reference_slaspec": "initial state (pass --reference)",
        "source_path": "initial state (pass --source)",
        "db_path": "initial state (pass --db)",
        "processor_name": "initial state (pass --name)",
        "out_dir": "initial state (pass --out)",
        "settings_dict": "initial state",
    }
    return _produces.get(key)


# ---------------------------------------------------------------------------
# Summaries and fallback detection
# ---------------------------------------------------------------------------

def summarize_and_warn(stage: str, state: dict[str, Any]) -> None:
    """Print a verification summary; loudly warn on silent fallbacks."""
    errors = state.get("errors") or []
    if errors:
        log.warning("Stage '%s' accumulated errors: %s", stage, errors)

    if stage == "ingest":
        log.info("ingest: ChromaDB populated at %s", state.get("db_path"))

    elif stage == "meta":
        meta = state.get("meta") or {}
        name = meta.get("name", "")
        if name in ("Unknown", "", None):
            log.warning("meta: name='%s' — looks like the LLM fallback fired!", name)
        else:
            log.info("meta: name=%r endian=%s word_size=%s",
                     name, meta.get("endian"), meta.get("word_size_bits"))

    elif stage == "registers":
        regs = state.get("registers") or []
        if not regs:
            log.warning("registers: empty — LLM returned nothing")
        else:
            log.info("registers: %d registers (first: %s)", len(regs), regs[0].get("name"))

    elif stage == "mnemonics":
        mn = state.get("mnemonics") or []
        if not mn:
            log.warning("mnemonics: empty — discovery returned nothing")
        else:
            log.info("mnemonics: %d discovered (sample: %s)", len(mn), mn[:5])

    elif stage == "instructions":
        instrs = state.get("instructions") or []
        mn = state.get("mnemonics") or []
        if not instrs:
            log.warning("instructions: empty — no instructions extracted")
        else:
            log.info("instructions: %d extracted (of %d mnemonics)", len(instrs), len(mn))
            no_semantics = sum(1 for i in instrs if not i.get("semantics"))
            if no_semantics:
                log.warning("instructions: %d entries have empty semantics", no_semantics)

    elif stage == "pcode":
        instrs = state.get("instructions") or []
        with_hint = sum(1 for i in instrs if i.get("pcode_hint"))
        if with_hint == 0:
            log.warning("pcode: no instructions received a pcode_hint")
        else:
            log.info("pcode: %d / %d instructions have pcode_hint", with_hint, len(instrs))

    elif stage == "generate":
        lang_dir = state.get("lang_dir")
        if not lang_dir:
            log.warning("generate: lang_dir not set — generation may have failed")
        else:
            p = Path(lang_dir)
            files = list(p.glob("*")) if p.exists() else []
            log.info("generate: lang_dir=%s (%d files: %s)",
                     lang_dir, len(files), [f.name for f in files])

    elif stage == "validate":
        ok = state.get("compile_ok")
        errs = state.get("compile_errors") or []
        if ok:
            log.info("validate: SLEIGH compile OK")
        else:
            log.warning("validate: compile FAILED — %d errors: %s", len(errs), errs[:3])

    elif stage == "evaluate":
        cov = state.get("instruction_coverage")
        reg = state.get("register_overlap")
        log.info("evaluate: coverage=%.3f  reg_overlap=%.3f", cov or 0.0, reg or 0.0)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_stage(state: dict[str, Any], stage: str) -> dict[str, Any]:
    """Validate prereqs, run the node, merge result, return updated state."""
    if stage not in STAGE_REGISTRY:
        raise ValueError(f"Unknown stage '{stage}'. Valid stages: {STAGE_ORDER}")

    check_prereqs(stage, state)

    import_fn, _ = STAGE_REGISTRY[stage]
    node_fn = import_fn()
    log.info("--- Running stage: %s ---", stage)
    partial = node_fn(state)
    return merge(state, partial)


# ---------------------------------------------------------------------------
# Build initial state
# ---------------------------------------------------------------------------

def build_initial_state(
    db_path: str,
    processor_name: str,
    out_dir: str,
    settings_dict: dict[str, Any],
    ghidra_home: str,
    reference_slaspec: str | None,
    source_path: str | None,
    inter_chunk_sleep: float,
    max_instructions: int | None,
    max_pcode: int | None,
    memory_warn_gb: float,
) -> dict[str, Any]:
    debug_save_dir = str(Path(db_path).parent)
    state: dict[str, Any] = {
        "db_path": db_path,
        "settings_dict": settings_dict,
        "processor_name": processor_name,
        "out_dir": out_dir,
        "ghidra_home": ghidra_home,
        "debug_save_dir": debug_save_dir,
        # Singleton concurrency
        "max_concurrent": 1,
        "chunk_size": 1,
        "inter_chunk_sleep": inter_chunk_sleep,
        "memory_warn_gb": memory_warn_gb,
        "max_instructions": max_instructions,
        "max_pcode": max_pcode,
        "resume": False,
        "stop_after": None,
        "filter_mnemonics": None,
        "errors": [],
    }
    if source_path:
        state["source_path"] = source_path
    if reference_slaspec:
        state["reference_slaspec"] = reference_slaspec
    return state
