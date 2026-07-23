"""Build SLEIGH constructor patterns/displays and `attach names` statements.

Tier 1 structured-decode rendering. Kept in Python (not the Jinja template) so
the SLEIGH-validity rules are unit-testable by string inspection. All behaviour
is gated on the backend-neutral ISAMeta fields (isa_variants, parallel_field,
predicate_fields, field_attachments) + InstructionDef.functional_unit — when
those are empty (every ISA except a configured TMS320) output is unchanged.

Token fields are width-suffixed to avoid cross-width collisions (e.g. `s` in a
32-bit token is `s32`), matching the token definitions in processor.slaspec.j2.
"""

from __future__ import annotations

from rosetta_schemas.models import ISAMeta, InstructionDef


def field_sym(name: str, width: int) -> str:
    """Width-suffixed token-field symbol, matching processor.slaspec.j2."""
    return f"{name}{width}"


def build_attach_stmts(instructions: list[InstructionDef], meta: ISAMeta) -> list[dict]:
    """`attach names` statements: one per (width, attachment field) actually defined.

    Renders raw bit fields (register side, condition register, …) as symbolic
    operands. Empty when meta.field_attachments is empty.
    """
    stmts: list[dict] = []
    seen: set[str] = set()
    widths = sorted({i.encoding_bits for i in instructions if i.encoding_bits > 0})
    for width in widths:
        fields_here: set[str] = set()
        for i in instructions:
            if i.encoding_bits == width:
                fields_here |= set(i.bit_fields)
        for field, names in meta.field_attachments.items():
            if field in fields_here:
                sym = field_sym(field, width)
                if sym not in seen:
                    stmts.append({"sym": sym, "names": [str(n) for n in names]})
                    seen.add(sym)
    return stmts


def build_render_instructions(
    instructions: list[InstructionDef], meta: ISAMeta
) -> list[dict]:
    """Precompute {display, pattern, pcode_hint} per instruction for the template.

    The grounded fixed-opcode bits (bit_constraints) stay the decode core; the
    predication prefix, functional-unit qualifier, parallel marker, and attached
    operand fields are layered on as display + bound pattern fields. Instructions
    without recovered opcode bits fall back to a unique opNstub pattern.
    """
    par = meta.parallel_field
    out: list[dict] = []
    stub_id = 0
    for instr in instructions:
        width = instr.encoding_bits
        binds: list[str] = []
        trail: list[str] = []  # display operands after the mnemonic

        # Mnemonic first (SLEIGH convention — mnemonic is the leading literal),
        # with the functional-unit letter fused on (ADDSP -> ADDSP.S).
        mnem = instr.mnemonic
        if instr.functional_unit:
            mnem = f"{mnem}.{instr.functional_unit}"

        # Predication (e.g. creg/z) — bound + shown right after the mnemonic.
        for f in meta.predicate_fields:
            if f in instr.bit_fields:
                s = field_sym(f, width)
                trail.append(s)
                binds.append(s)

        # Attached operand fields (e.g. register side 's'), excluding predication
        # and the parallel bit — shown after predication.
        for f in meta.field_attachments:
            if f in instr.bit_fields and f not in meta.predicate_fields and f != par:
                s = field_sym(f, width)
                trail.append(s)
                binds.append(s)

        # Parallel marker (VLIW p-bit): bind always; display (as trailing operand)
        # only when the config attaches display names to it.
        if par and par in instr.bit_fields:
            psym = field_sym(par, width)
            binds.append(psym)
            if par in meta.field_attachments:
                trail.append(psym)

        display = " ".join([mnem, *trail])

        if instr.bit_constraints:
            core = [f"{field_sym(f, width)}=0b{v}" for f, v in instr.bit_constraints.items()]
            pattern = " & ".join(core + binds)
        else:
            pattern = f"op{width}stub={stub_id}"
            stub_id += 1
            if binds:
                pattern += " & " + " & ".join(binds)

        out.append({"display": display, "pattern": pattern, "pcode_hint": instr.pcode_hint})
    return out
