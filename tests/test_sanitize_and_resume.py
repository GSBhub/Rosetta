"""Register-name sanitizing and multi-encoding resume.

Both guard silent data corruption: a name collision aliases two architectural
registers onto one varnode, and a resume that keys by mnemonic drops every
encoding but the last.
"""

from __future__ import annotations

import json

from rosetta_schemas.models import InstructionDef, RegisterDef
from rosetta_generate_sla.sla.sanitize import (
    sanitize_ident,
    sanitize_registers,
)


# ── identifiers ─────────────────────────────────────────────────────────────

def test_sanitize_ident_coerces_invalid_characters():
    assert sanitize_ident("CPU ID") == "CPU_ID"       # space (real: TMS320C6x)
    assert sanitize_ident("R0") == "R0"               # already valid
    assert sanitize_ident("2ND") == "_2ND"            # leading digit
    assert sanitize_ident("  ") == "_"                # degenerate


def test_colliding_register_names_stay_distinct():
    """Coercion is many-to-one, so two registers can collapse onto one
    identifier — SLEIGH would alias them to the same varnode."""
    regs = [
        RegisterDef(name="CPU ID", size_bits=32, description="a"),
        RegisterDef(name="CPU-ID", size_bits=32, description="b"),
        RegisterDef(name="CPU.ID", size_bits=32, description="c"),
    ]
    names = [r.name for r in sanitize_registers(regs)]
    assert names == ["CPU_ID", "CPU_ID_2", "CPU_ID_3"]
    assert len(set(names)) == len(names)


def test_sanitize_registers_leaves_distinct_names_alone():
    regs = [RegisterDef(name="R0", size_bits=32, description=""),
            RegisterDef(name="R1", size_bits=32, description="")]
    assert [r.name for r in sanitize_registers(regs)] == ["R0", "R1"]


# ── resume with multiple encodings per mnemonic ─────────────────────────────

def _partial(tmp_path, instrs):
    p = tmp_path / "pass4_partial.jsonl"
    p.write_text("\n".join(i.model_dump_json() for i in instrs) + "\n")
    return p


def test_resume_keeps_every_encoding_of_a_mnemonic(tmp_path, monkeypatch):
    """A mnemonic yields one InstructionDef per grounded encoding; resume must
    restore them all, not just the last line for that mnemonic."""
    from rosetta_instructions import node as node_mod

    instrs = [
        InstructionDef(mnemonic="ADD", encoding_bits=32, semantics="s",
                       bit_fields={"op_11_5": "11:5"},
                       bit_constraints={"op_11_5": "0000011"}),
        InstructionDef(mnemonic="ADD", encoding_bits=32, semantics="s",
                       bit_fields={"op_11_5": "11:5"},
                       bit_constraints={"op_11_5": "0100011"}),
        InstructionDef(mnemonic="SUB", encoding_bits=32, semantics="s",
                       bit_fields={"op_11_5": "11:5"},
                       bit_constraints={"op_11_5": "0000111"}),
    ]
    debug_dir = tmp_path
    _partial(debug_dir, instrs)

    # Stub out everything after the resume load: no store, no LLM.
    monkeypatch.setattr(node_mod, "build_encoding_index", lambda *_a, **_k: {},
                        raising=False)

    state = {
        "mnemonics": ["ADD", "SUB"],      # both already done -> no extraction
        "db_path": str(tmp_path / "db"),
        "settings_dict": {},
        "resume": True,
        "debug_save_dir": str(debug_dir),
        "errors": [],
    }

    class _FakeSettings:
        def __init__(self, **kw): self.db_path = None; self.vs = None

    monkeypatch.setattr("docquery.config.Settings", _FakeSettings)
    monkeypatch.setattr("rosetta_utils.chroma.get_chroma_wrapper",
                        lambda *_a, **_k: None)

    out = node_mod.instructions_node(state)
    restored = out["instructions"]
    assert len(restored) == 3, f"resume dropped encodings: {restored}"
    add_constraints = sorted(
        json.dumps(i["bit_constraints"], sort_keys=True)
        for i in restored if i["mnemonic"] == "ADD"
    )
    assert len(add_constraints) == 2 and len(set(add_constraints)) == 2
