"""Overlay per-ISA structured-decode config onto an extracted ISASpec.

Keeps the extraction pipeline ISA-agnostic: family variants, the VLIW p-bit,
predication fields, SLEIGH field attachments, and per-instruction functional
units are supplied as *data* in an ``examples/*.toml`` ``[decode]`` table and
applied here just before rendering. The same neutral fields feed a future
QEMU/TCG backend unchanged.

Example ``[decode]`` table::

    [decode]
    isa_variants    = ["C62x", "C64x", "C67x", "C67x+"]
    parallel_field  = "p"
    predicate_fields = ["creg", "z"]

    [decode.field_attachments]
    s = ["1", "2"]

    [decode.unit_map]
    ABSSP = "S"
    LDW   = "D"
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from rosetta_schemas.models import ISASpec


def apply_isa_config(spec: ISASpec, config_path: str | Path) -> ISASpec:
    """Overlay the ``[decode]`` table of *config_path* onto *spec*.

    Mutates *spec* in place and returns it for convenience — the return value is
    the same object, not a copy. Missing keys leave the spec untouched, so a
    config without a ``[decode]`` table (or no config at all) is a no-op —
    existing ISAs are unaffected.
    """
    with Path(config_path).open("rb") as fh:
        data = tomllib.load(fh)
    dec = data.get("decode")
    if not dec:
        return spec

    meta = spec.meta
    # Correct extraction-metadata the manual states plainly but the LLM may miss.
    if dec.get("endian") in ("little", "big", "bi"):
        meta.endian = dec["endian"]
    if dec.get("variant"):
        meta.variant = str(dec["variant"])
    if dec.get("isa_variants"):
        meta.isa_variants = [str(v) for v in dec["isa_variants"]]
    if dec.get("parallel_field"):
        meta.parallel_field = str(dec["parallel_field"])
    if dec.get("predicate_fields"):
        meta.predicate_fields = [str(v) for v in dec["predicate_fields"]]
    if dec.get("field_attachments"):
        meta.field_attachments = {
            str(k): [str(s) for s in v] for k, v in dec["field_attachments"].items()
        }

    unit_map = dec.get("unit_map") or {}
    if unit_map:
        by_mnem = {str(k).upper(): str(v) for k, v in unit_map.items()}
        for instr in spec.instructions:
            unit = by_mnem.get(instr.mnemonic.upper())
            if unit:
                instr.functional_unit = unit

    return spec
