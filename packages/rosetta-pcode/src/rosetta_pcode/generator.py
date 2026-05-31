"""P-code hint generation via direct LLM call (Pass 5)."""

from __future__ import annotations

import logging

from rosetta_schemas.models import InstructionDef

log = logging.getLogger(__name__)

_PCODE_SYSTEM = """\
You are an expert in Ghidra's SLEIGH language. Given a natural-language description of an
instruction's semantics, produce a single-line SLEIGH P-code statement that captures the
core operation. Use lowercase register/variable names (e.g. rd, rs, ra, tmp). Examples:
  "Adds two registers, stores result in destination" → "rd = ra + rb;"
  "Loads a 32-bit word from memory at base address into destination" → "rd = *[ram]:4 ra;"
  "Branches unconditionally to the address in a register" → "goto [ra];"
Return ONLY the P-code statement, no explanation.
"""


def generate_pcode(instruction: InstructionDef, settings: object) -> str:
    """Call the LLM directly to translate instruction semantics → P-code hint."""
    try:
        from rosetta_utils.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm(settings)
        messages = [
            SystemMessage(content=_PCODE_SYSTEM),
            HumanMessage(content=f"Instruction: {instruction.mnemonic}\nSemantics: {instruction.semantics}"),
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        log.warning("P-code generation failed for %s: %s", instruction.mnemonic, exc)
        return f"# TODO: {instruction.semantics}"
