"""Per-instruction structural validation and safe-stub downgrade."""

from __future__ import annotations

import re
import logging

from rosetta_schemas.models import InstructionDef

log = logging.getLogger(__name__)

_VALID_MNEMONIC = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,19}$")


def validate_and_fix(instr: InstructionDef) -> tuple[InstructionDef, list[str]]:
    """Validate *instr*, downgrade any bad fields to safe stubs, return (instr, issues).

    Never returns an InstructionDef that would cause a hard downstream error.
    Issues list is non-empty when a field was corrected.
    """
    from rosetta_generate_sla.sla.sanitize import sanitize_pcode

    issues: list[str] = []

    # 1. encoding_bits must be positive and a multiple of 8.
    if instr.encoding_bits <= 0:
        instr = instr.model_copy(update={"encoding_bits": 32})
        issues.append(f"{instr.mnemonic}: encoding_bits was <=0, defaulted to 32")
    elif instr.encoding_bits % 8 != 0:
        fixed = ((instr.encoding_bits + 7) // 8) * 8
        instr = instr.model_copy(update={"encoding_bits": fixed})
        issues.append(f"{instr.mnemonic}: encoding_bits rounded up to {fixed}")

    # 2. Mnemonic must be a valid identifier.
    if not _VALID_MNEMONIC.match(instr.mnemonic):
        issues.append(f"invalid mnemonic {instr.mnemonic!r} — kept as-is, will use stub pattern")

    # 3. pcode_hint: sanitize; if the sanitizer returns a stub that's fine.
    cleaned = sanitize_pcode(instr.pcode_hint)
    if cleaned != instr.pcode_hint:
        instr = instr.model_copy(update={"pcode_hint": cleaned})
        if instr.pcode_hint:
            issues.append(f"{instr.mnemonic}: pcode_hint sanitized")

    # 4. bit_constraints values must be binary strings; prune invalid ones.
    bad_constraints = {
        k: v for k, v in instr.bit_constraints.items()
        if not re.match(r"^[01]+$", str(v))
    }
    if bad_constraints:
        clean_bc = {k: v for k, v in instr.bit_constraints.items() if k not in bad_constraints}
        instr = instr.model_copy(update={"bit_constraints": clean_bc})
        issues.append(f"{instr.mnemonic}: pruned non-binary bit_constraints {list(bad_constraints)}")

    return instr, issues
