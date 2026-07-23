"""SLEIGH P-code sanitization and instruction normalization helpers."""

from __future__ import annotations

import copy
import re

from rosetta_schemas.models import InstructionDef, RegisterDef

_BAD_PCODE = re.compile(
    r"Error:|Unknown\b|undefined\b|TODO\b|not extracted|"
    r"[A-Za-z_]\w*\s*\(|"
    r"\bor\b|\band\b|\bnot\b|"
    r"[A-Z][a-z]+(?:_[A-Z][a-z]+)+|"
    r"\bMem(?:ory)?\[|"
    r"#[A-Za-z_]\w*|"
    r"\b\w+\[\d+:\d+\]\s*=|"
    r"\b[A-Z][a-z]\w*\b",
    re.IGNORECASE,
)

_VALID_IDENT = re.compile(r'^[A-Za-z_]\w*$')
_PURE_BINARY = re.compile(r'^[01]+$')
_SINGLE_INT = re.compile(r'^\d+$')
_IDENT_BAD = re.compile(r'[^A-Za-z0-9_]')


def sanitize_ident(name: str) -> str:
    """Coerce a name into a valid SLEIGH identifier ([A-Za-z_]\\w*).

    Register/field names lifted from a manual can contain spaces or punctuation
    (e.g. 'CPU ID'); the SLEIGH compiler rejects those, so runs of invalid
    characters collapse to '_' and a leading digit is prefixed with '_'.
    """
    s = _IDENT_BAD.sub("_", name.strip())
    if not s:
        return "_"
    if s[0].isdigit():
        s = "_" + s
    return s


def sanitize_register(reg: "RegisterDef") -> "RegisterDef":
    """Copy of *reg* with name and aliases coerced to valid SLEIGH identifiers."""
    r = copy.copy(reg)
    r.name = sanitize_ident(reg.name)
    r.aliases = [sanitize_ident(a) for a in reg.aliases]
    return r


def sanitize_pcode(hint: str) -> str:
    """Return hint if it looks like valid SLEIGH P-code, else a safe stub."""
    s = hint.strip() if hint else ""
    if not s:
        return "local tmp:4 = 0;"
    # The rejected hint is kept as a comment, which must be a single line: a
    # SLEIGH '#' comment ends at the newline, so a multi-line hint would spill
    # its own prose into the constructor body as invalid p-code.
    comment = " ".join(s.split())[:80]
    if not s.endswith(";"):
        return f"# {comment}\n    local tmp:4 = 0;"
    if _BAD_PCODE.search(s):
        return f"# {comment}\n    local tmp:4 = 0;"
    return s


def normalize_bit_fields(bit_fields: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, bit_range in bit_fields.items():
        if not _VALID_IDENT.match(name):
            continue
        s = str(bit_range).strip()
        if ' ' in s or s.count(':') > 1:
            continue
        if _SINGLE_INT.match(s):
            s = f"{s}:{s}"
        parts = s.split(':')
        if len(parts) != 2:
            continue
        try:
            int(parts[0]), int(parts[1])
        except ValueError:
            continue
        result[name] = s
    return result



def normalize_bit_constraints(
    bit_constraints: dict[str, str], bit_fields: dict[str, str]
) -> dict[str, str]:
    """Keep only constraints that render as a valid SLEIGH pattern equation.

    A constraint survives when its field is a defined token field (present in
    the already-normalized bit_fields) and its value is pure binary, so the
    template can emit ``<field>=0b<value>``. Grounded opcode fields are named
    ``op_<hi>_<lo>`` (position-unique, collision-safe); this is what turns a
    stub constructor into a real decode pattern.
    """
    result: dict[str, str] = {}
    for name, value in bit_constraints.items():
        if name not in bit_fields:
            continue
        v = str(value).strip()
        if not _PURE_BINARY.match(v):
            continue
        result[name] = v
    return result


def normalize_instruction(instr: InstructionDef) -> InstructionDef:
    ni = copy.copy(instr)
    if ni.encoding_bits <= 0:
        ni.encoding_bits = 32
    elif ni.encoding_bits % 8 != 0:
        ni.encoding_bits = ((ni.encoding_bits + 7) // 8) * 8
    ni.mnemonic = ni.mnemonic.replace(" ", "_")
    ni.bit_fields = normalize_bit_fields(ni.bit_fields)
    # Preserve grounded fixed-opcode bits as the decode pattern (filtered to
    # token-defined fields with pure-binary values). generate() further filters
    # over-wide values and de-duplicates identical patterns for compile-safety.
    ni.bit_constraints = normalize_bit_constraints(ni.bit_constraints, ni.bit_fields)
    ni.semantics = " ".join(ni.semantics.splitlines())
    # Operand display binding is intentionally omitted: binding named register
    # fields in the pattern risks SLEIGH field-name collisions/reserved words
    # that can't be verified without the Ghidra compiler. Display stays mnemonic
    # -only; the pattern still decodes correctly from the fixed opcode bits.
    ni.operands = []
    return ni


def find_register(
    registers: list[RegisterDef],
    *candidate_aliases: str,
    description_keyword: str = "",
) -> str:
    upper_aliases = {a.upper() for a in candidate_aliases}
    for reg in registers:
        names = {reg.name.upper()} | {a.upper() for a in reg.aliases}
        if names & upper_aliases:
            return reg.name
    if description_keyword:
        kw = description_keyword.lower()
        for reg in registers:
            if kw in reg.description.lower():
                return reg.name
    return registers[0].name if registers else candidate_aliases[0] if candidate_aliases else "PC"
