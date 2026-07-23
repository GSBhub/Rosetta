"""Build SLEIGH token fields, constructor patterns/displays and `attach names`.

Kept in Python (not the Jinja template) so the SLEIGH-validity rules are
unit-testable by string inspection, and so token declarations and the
constructors that reference them are derived from one shared symbol table —
they must agree exactly or the spec decodes from the wrong bits.

Structured-decode rendering is gated on the backend-neutral ISAMeta fields
(parallel_field, predicate_fields, field_attachments) plus
InstructionDef.functional_unit. When those are empty (every ISA except a
configured one) output is unchanged.
"""

from __future__ import annotations

from rosetta_schemas.models import ISAMeta, InstructionDef


def _split_range(rng: str) -> tuple[int, int] | None:
    """'hi:lo' -> (hi, lo), or None when unparseable."""
    hi, sep, lo = str(rng).partition(":")
    if not sep:
        return None
    try:
        return int(hi), int(lo)
    except ValueError:
        return None


def build_field_symbols(instructions: list[InstructionDef]) -> dict[tuple[str, str, int], str]:
    """``(field_name, 'hi:lo', width) -> SLEIGH token symbol``.

    Field names recovered from a manual are **not** position-stable: the same
    name routinely appears at different bit ranges across instructions (``creg``
    at both 31:29 and 31:30, ``x`` at half a dozen). A token symbol is declared
    once, so reusing one symbol for several ranges would silently decode every
    instruction but the first from the wrong bits.

    Names used at a single range keep the plain ``<name><width>`` symbol — the
    common case, and what keeps output stable for well-formed manuals. Ambiguous
    names are disambiguated by position (``creg_31_29_32``).
    """
    ranges: dict[tuple[str, int], set[str]] = {}
    for instr in instructions:
        width = instr.encoding_bits
        for name, rng in instr.bit_fields.items():
            if _split_range(rng):
                ranges.setdefault((name, width), set()).add(str(rng))

    symbols: dict[tuple[str, str, int], str] = {}
    for (name, width), rngs in ranges.items():
        for rng in rngs:
            if len(rngs) == 1:
                symbols[(name, rng, width)] = f"{name}{width}"
            else:
                # Separator before the width: `creg_31_29` + `32` would read as
                # the unparseable `creg_31_2932`.
                hi, lo = _split_range(rng)  # type: ignore[misc]
                symbols[(name, rng, width)] = f"{name}_{hi}_{lo}_{width}"
    return symbols


def symbol_for(
    symbols: dict[tuple[str, str, int], str], instr: InstructionDef, name: str
) -> str | None:
    """The token symbol *instr* uses for its field *name*, or None if it has none."""
    rng = instr.bit_fields.get(name)
    if rng is None:
        return None
    return symbols.get((name, str(rng), instr.encoding_bits))


def build_tokens(
    instructions: list[InstructionDef],
    widths: list[int],
    symbols: dict[tuple[str, str, int], str],
) -> list[dict]:
    """One token block per width: ``[{"width", "fields": [{"sym","lo","hi"}]}]``.

    Built here rather than in the template so every declared symbol comes from
    the same table the constructors reference.
    """
    per_width: dict[int, dict[str, tuple[int, int]]] = {}
    for instr in instructions:
        width = instr.encoding_bits
        for name, rng in instr.bit_fields.items():
            parts = _split_range(rng)
            sym = symbols.get((name, str(rng), width))
            if not parts or not sym:
                continue
            hi, lo = parts
            per_width.setdefault(width, {}).setdefault(sym, (lo, hi))

    return [
        {
            "width": width,
            "fields": [
                {"sym": sym, "lo": lo, "hi": hi}
                for sym, (lo, hi) in (per_width.get(width) or {}).items()
            ],
        }
        for width in widths
    ]


def build_attach_stmts(
    instructions: list[InstructionDef],
    meta: ISAMeta,
    symbols: dict[tuple[str, str, int], str],
) -> list[dict]:
    """`attach names` statements for the configured attachment fields.

    A name that resolved to several symbols (see :func:`build_field_symbols`)
    gets one statement per symbol, so every declared variant is attached.
    """
    stmts: list[dict] = []
    seen: set[str] = set()
    for (name, _rng, _width), sym in sorted(symbols.items()):
        names = meta.field_attachments.get(name)
        if not names or sym in seen:
            continue
        seen.add(sym)
        stmts.append({"sym": sym, "names": _fmt_attach_names(names)})
    return stmts


def _fmt_attach_names(names: list[str]) -> str:
    """Format an `attach names` value list: `_` stays bare, everything else is
    a quoted display string (SLEIGH rejects bare numbers/punctuation).

    Values come from a per-ISA config, so escape the quote and backslash that
    would otherwise terminate the string and produce an unparseable statement.
    """
    out = []
    for n in names:
        s = str(n)
        if s == "_":
            out.append(s)
        else:
            escaped = s.replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'"{escaped}"')
    return " ".join(out)


def build_render_instructions(
    instructions: list[InstructionDef],
    meta: ISAMeta,
    symbols: dict[tuple[str, str, int], str] | None = None,
) -> list[dict]:
    """Precompute {display, pattern, pcode_hint} per instruction for the template.

    The grounded fixed-opcode bits (bit_constraints) stay the decode core; the
    predication prefix, functional-unit qualifier, parallel marker, and attached
    operand fields are layered on as display + bound pattern fields. Instructions
    without recovered opcode bits fall back to a unique opNstub pattern.
    """
    if symbols is None:
        symbols = build_field_symbols(instructions)
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
            if s := symbol_for(symbols, instr, f):
                trail.append(s)
                binds.append(s)

        # Attached operand fields (e.g. register side 's'), excluding predication
        # and the parallel bit — shown after predication.
        for f in meta.field_attachments:
            if f in meta.predicate_fields or f == par:
                continue
            if s := symbol_for(symbols, instr, f):
                trail.append(s)
                binds.append(s)

        # Parallel marker (VLIW p-bit): bind always; display (as trailing operand)
        # only when the config attaches display names to it.
        if par and (psym := symbol_for(symbols, instr, par)):
            binds.append(psym)
            if par in meta.field_attachments:
                trail.append(psym)

        display = " ".join([mnem, *trail])

        if instr.bit_constraints:
            core = []
            for f, v in instr.bit_constraints.items():
                if s := symbol_for(symbols, instr, f):
                    core.append(f"{s}=0b{v}")
            pattern = " & ".join(core + binds)
        else:
            pattern = f"op{width}stub={stub_id}"
            stub_id += 1
            if binds:
                pattern += " & " + " & ".join(binds)

        out.append({"display": display, "pattern": pattern, "pcode_hint": instr.pcode_hint})
    return out
