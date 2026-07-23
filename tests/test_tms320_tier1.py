"""Tier 1 structured-decode: schema fields, config overlay, SLEIGH rendering.

Covers the backend-neutral additions (ISA-family variants, parallel bit,
predication, field attachments, functional unit) end-to-end without Ghidra:
the overlay applies config data, and the generator renders multi-language
.ldefs, `attach names`, and mnemonic-first constructors with predication/unit.
Also asserts ISAs *without* the overlay are unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rosetta_schemas.models import ISAMeta, ISASpec, InstructionDef, RegisterDef
from rosetta_schemas.overlay import apply_isa_config
from rosetta_generate_sla.sla.module_generator import ModuleGenerator
from rosetta_generate_sla.sla.render import (
    build_attach_stmts,
    build_field_symbols,
    build_render_instructions,
)


def _meta(**kw) -> ISAMeta:
    base = dict(name="T", endian="little", word_size_bits=32, alignment=4,
                instruction_sizes_bits=[32], encoding_style="fixed_word")
    base.update(kw)
    return ISAMeta(**base)


def _instr(mnem, fields, constraints, **kw) -> InstructionDef:
    return InstructionDef(mnemonic=mnem, encoding_bits=32, bit_fields=fields,
                          bit_constraints=constraints, semantics="s",
                          pcode_hint="local tmp:4 = 0;", **kw)


def _tms_spec() -> ISASpec:
    regs = [RegisterDef(name="A0", size_bits=32, description="gp"),
            RegisterDef(name="pc", size_bits=32, aliases=["PC"], description="program counter")]
    instrs = [
        _instr("ABSDP",
               {"creg": "31:29", "z": "28:28", "op_11_2": "11:2", "s": "1:1", "p": "0:0"},
               {"op_11_2": "1011001000"}),
        _instr("LDW", {}, {}),  # no recovered diagram -> stub, still gets unit letter
    ]
    return ISASpec(meta=_meta(), registers=regs, instructions=instrs)


# ── schema ────────────────────────────────────────────────────────────────

def test_new_fields_default_empty():
    m = _meta()
    assert m.isa_variants == [] and m.parallel_field is None
    assert m.predicate_fields == [] and m.field_attachments == {}
    i = _instr("X", {}, {})
    assert i.isa_variants == [] and i.functional_unit == ""


# ── overlay ─────────────────────────────────────────────────────────────────

def test_apply_isa_config_overlays_decode_table(tmp_path):
    cfg = tmp_path / "isa.toml"
    cfg.write_text(
        'name="T"\ndb="d"\nmanual="m"\n'
        '[decode]\n'
        'endian="little"\n'
        'isa_variants=["C62x","C67x"]\n'
        'parallel_field="p"\n'
        'predicate_fields=["creg","z"]\n'
        '[decode.field_attachments]\n'
        's=["1","2"]\n'
        '[decode.unit_map]\n'
        'LDW="D"\n'
    )
    spec = apply_isa_config(_tms_spec(), cfg)
    assert spec.meta.isa_variants == ["C62x", "C67x"]
    assert spec.meta.parallel_field == "p"
    assert spec.meta.predicate_fields == ["creg", "z"]
    assert spec.meta.field_attachments == {"s": ["1", "2"]}
    ldw = next(i for i in spec.instructions if i.mnemonic == "LDW")
    assert ldw.functional_unit == "D"


def test_apply_isa_config_no_decode_is_noop(tmp_path):
    cfg = tmp_path / "isa.toml"
    cfg.write_text('name="T"\ndb="d"\nmanual="m"\n')
    spec = apply_isa_config(_tms_spec(), cfg)
    assert spec.meta.isa_variants == [] and spec.meta.parallel_field is None


# ── render helpers ──────────────────────────────────────────────────────────

def test_build_attach_stmts_width_suffixed():
    meta = _meta(field_attachments={"s": ["1", "2"]})
    instrs = _tms_spec().instructions
    symbols = build_field_symbols(instrs)
    stmts = build_attach_stmts(instrs, meta, symbols)
    assert stmts == [{"sym": "s32", "names": '"1" "2"'}]


def test_render_predication_unit_and_pbit():
    meta = _meta(parallel_field="p", predicate_fields=["creg", "z"],
                 field_attachments={"s": ["1", "2"]})
    instrs = _tms_spec().instructions
    instrs[1].functional_unit = "D"
    rendered = build_render_instructions(instrs, meta)
    absdp, ldw = rendered
    # mnemonic-first, predication + side shown, opcode bits + bound operands
    assert absdp["display"].split()[0] == "ABSDP"
    assert "creg32" in absdp["display"] and "s32" in absdp["display"]
    assert "op_11_232=0b1011001000" in absdp["pattern"]
    assert "p32" in absdp["pattern"]
    # stub instruction still gets the unit letter fused onto the mnemonic
    assert ldw["display"] == "LDW.D"
    assert ldw["pattern"].startswith("op32stub=")


# ── end-to-end generation ───────────────────────────────────────────────────

def _render(spec) -> str:
    out = Path(tempfile.mkdtemp())
    try:
        ModuleGenerator().generate(spec, "T", out)
        slaspec = (out / "T/data/languages/T.slaspec").read_text()
        ldefs = (out / "T/data/languages/T.ldefs").read_text()
        return slaspec + "\n@@LDEFS@@\n" + ldefs
    finally:
        import shutil
        shutil.rmtree(out)


def test_generate_multi_language_and_attach():
    spec = _tms_spec()
    spec.meta.isa_variants = ["C62x", "C64x", "C67x"]
    spec.meta.parallel_field = "p"
    spec.meta.predicate_fields = ["creg", "z"]
    spec.meta.field_attachments = {"s": ["1", "2"]}
    text = _render(spec)
    slaspec, ldefs = text.split("@@LDEFS@@")
    # one <language> per family
    for fam in ("C62x", "C64x", "C67x"):
        assert f":32:{fam}" in ldefs
    assert 'attach names [ s32 ] [ "1" "2" ];' in slaspec
    # grounded constructor: mnemonic-first with bound predication/side/p
    assert ":ABSDP creg32 z32 s32 is" in slaspec
    assert "op_11_232=0b1011001000" in slaspec


def test_generate_without_overlay_is_plain():
    """No isa_variants/attachments -> single language, no attach, mnemonic-only."""
    spec = _tms_spec()  # defaults: no variants/predicate/attachments
    text = _render(spec)
    slaspec, ldefs = text.split("@@LDEFS@@")
    assert ldefs.count("<language ") == 1
    assert "attach names" not in slaspec
    assert ":ABSDP is op_11_232=0b1011001000" in slaspec  # no creg/z/s operands


# ── token symbols: a field name is not position-stable ──────────────────────

def test_ambiguous_field_name_gets_position_disambiguated_symbols():
    """The same name at different ranges must not share one token symbol.

    Recovery yields e.g. `creg` at both 31:29 and 31:30 in one manual. A token
    is declared once, so reusing the symbol would decode every instruction but
    the first from the wrong bits.
    """
    instrs = [
        _instr("A", {"creg": "31:29", "op_11_2": "11:2"}, {"op_11_2": "1011001000"}),
        _instr("B", {"creg": "31:30", "op_11_2": "11:2"}, {"op_11_2": "1111001000"}),
    ]
    symbols = build_field_symbols(instrs)
    assert symbols[("creg", "31:29", 32)] == "creg_31_29_32"
    assert symbols[("creg", "31:30", 32)] == "creg_31_30_32"
    # unambiguous names keep the plain symbol
    assert symbols[("op_11_2", "11:2", 32)] == "op_11_232"


def test_unambiguous_field_keeps_plain_symbol():
    instrs = [_instr("A", {"creg": "31:29"}, {}), _instr("B", {"creg": "31:29"}, {})]
    assert build_field_symbols(instrs)[("creg", "31:29", 32)] == "creg32"


def test_every_declared_token_is_unique_and_constructors_reference_them():
    """End-to-end: no duplicate token declarations, and every symbol a
    constructor references is declared."""
    import re
    spec = _tms_spec()
    spec.meta.predicate_fields = ["creg", "z"]
    spec.instructions = [
        _instr("A", {"creg": "31:29", "z": "28:28", "op_11_2": "11:2"},
               {"op_11_2": "1011001000"}),
        _instr("B", {"creg": "31:30", "z": "29:27", "op_11_2": "11:2"},
               {"op_11_2": "1111001000"}),
    ]
    slaspec = _render(spec).split("@@LDEFS@@")[0]

    declared = re.findall(r"^\s+(\w+) = \(\d+,\d+\)", slaspec, re.M)
    assert len(declared) == len(set(declared)), f"duplicate token declarations: {declared}"

    referenced = set(re.findall(r"(\w+)=0b[01]+", slaspec))
    referenced |= {s for line in slaspec.splitlines() if line.startswith(":")
                   for s in re.findall(r"& (\w+)", line)}
    undeclared = referenced - set(declared) - {"op32stub", "op16stub"}
    assert not undeclared, f"constructors reference undeclared symbols: {undeclared}"
