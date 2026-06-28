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


def sanitize_pcode(hint: str) -> str:
    """Return hint if it looks like valid SLEIGH P-code, else a safe stub.

    Used as a *predicate* by the decode validator: a return value equal to the
    input means the hint looked acceptable. For rendering a guaranteed-compilable
    constructor body, use :func:`render_pcode_body` instead.
    """
    s = hint.strip() if hint else ""
    if not s:
        return "local tmp:4 = 0;"
    if not s.endswith(";"):
        return f"# {s[:80]}\n    local tmp:4 = 0;"
    if _BAD_PCODE.search(s):
        return f"# {s[:80]}\n    local tmp:4 = 0;"
    return s


def render_pcode_body(hint: str) -> str:
    """Return a guaranteed-compilable SLEIGH semantic body for *hint*.

    The extracted ``pcode_hint`` is LLM-authored prose that almost always
    references identifiers (``rd``, ``ra``, ``rb`` …) not declared as registers
    or operands in this constructor — emitting it verbatim produces "undefined
    symbol" errors, and partial fragments produce parse errors. So we never treat
    the hint as executable code: it is preserved as a single-line comment and the
    body is a no-op that references nothing undefined, so the constructor always
    compiles. P-code fidelity is a separate concern requiring decoded operands.
    """
    s = " ".join((hint or "").split())  # collapse newlines/runs of whitespace
    if not s:
        return "local tmp:4 = 0;"
    return f"# {s[:120]}\n    local tmp:4 = 0;"


def dedup_registers(registers: list[RegisterDef]) -> list[RegisterDef]:
    """Drop duplicate register names (case-insensitive), keeping first occurrence.

    Extraction frequently emits the same register twice (e.g. ``R15`` as both PC
    and a banked alias). SLEIGH rejects duplicate symbol names in a ``define
    register`` block, so we keep only the first definition of each name.
    """
    seen: set[str] = set()
    result: list[RegisterDef] = []
    for reg in registers:
        key = reg.name.upper()
        if key in seen:
            continue
        seen.add(key)
        result.append(reg)
    return result


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



def normalize_instruction(instr: InstructionDef) -> InstructionDef:
    ni = copy.copy(instr)
    if ni.encoding_bits <= 0:
        ni.encoding_bits = 32
    elif ni.encoding_bits % 8 != 0:
        ni.encoding_bits = ((ni.encoding_bits + 7) // 8) * 8
    ni.mnemonic = ni.mnemonic.replace(" ", "_")
    ni.bit_fields = normalize_bit_fields(ni.bit_fields)
    ni.bit_constraints = {}
    ni.semantics = " ".join(ni.semantics.splitlines())
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
