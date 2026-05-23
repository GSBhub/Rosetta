"""Opcode-table extraction: query the manual for 16-opcode rows of the opcode map."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field, model_validator

from rosetta_schemas.models import OpcodeDef

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst reading a processor reference manual. "
    "Extract the opcode table entries exactly as documented. "
    "Return only JSON matching the schema — no prose."
)


class _OpcodeRow(BaseModel):
    entries: list[OpcodeDef] = Field(
        description="The opcode table entries for this row of 16 opcodes"
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_entries(cls, data: Any) -> Any:
        """Accept any top-level list or dict-with-a-single-list-value the LLM returns."""
        if isinstance(data, list):
            return {"entries": data}
        if isinstance(data, dict) and "entries" not in data:
            # Case 1: wrong key name with a list value → {"opcode_row": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return {"entries": v}
            # Case 2: dict-of-dicts keyed by opcode → {"70": {"mnemonic": ...}, ...}
            dicts = [v for v in data.values() if isinstance(v, dict)]
            if dicts:
                return {"entries": dicts}
        return data


def _row_query(row: int, prefix: int | None) -> str:
    lo = row * 16
    hi = lo + 15
    hex_range = f"0x{lo:02X}–0x{hi:02X}"
    if prefix is not None:
        prefix_str = f"after prefix byte 0x{prefix:02X}, "
    else:
        prefix_str = ""
    return (
        f"From the instruction set opcode table, list all instructions {prefix_str}"
        f"with opcode byte values {hex_range} (row {row:X}h of the opcode map). "
        f"For each opcode in this range, provide: the exact opcode byte value as an integer, "
        f"the mnemonic (or 'UNK' if undefined/reserved), the addressing mode using standard "
        f"abbreviations (imp=implied, acc=accumulator, imm=immediate, dp=direct page, "
        f"dp,X=direct page+X, dp,Y=direct page+Y, (dp)=indirect dp, (dp,X)=dp indexed indirect, "
        f"(dp),Y=dp indirect indexed, abs=absolute, abs,X=absolute+X, abs,Y=absolute+Y, "
        f"(abs)=absolute indirect, (abs,X)=absolute indirect+X, rel=relative branch, "
        f"rel16=16-bit relative, long=absolute long 24-bit, long,X=long+X, "
        f"sr=stack relative, (sr),Y=stack relative indirect+Y), "
        f"and the number of operand bytes (0 for implied/acc, 1 for dp/rel/imm8/sr, "
        f"2 for abs/rel16/imm16, 3 for long)."
    )


def extract_opcode_row(
    settings: Any,
    row: int,
    prefix: int | None = None,
) -> list[OpcodeDef]:
    """Extract exactly one row (16 opcodes) via RAG similarity search."""
    import docquery

    query = _row_query(row, prefix)
    try:
        result = docquery.query(
            query,
            schema=_OpcodeRow,
            system_prompt=_SYSTEM_PROMPT,
            settings=settings,
        )
        if isinstance(result, _OpcodeRow):
            return result.entries
    except Exception as exc:
        log.warning("opcode_map: row 0x%Xx failed: %s", row, exc)
    return []


def extract_opcode_map(
    settings: Any,
    prefix: int | None = None,
    inter_row_sleep: float = 1.0,
) -> list[OpcodeDef]:
    """Extract all 256 opcodes by querying 16 rows of 16 opcodes each."""
    import docquery

    all_entries: list[OpcodeDef] = []
    seen_opcodes: set[tuple[int | None, int]] = set()

    for row in range(16):
        query = _row_query(row, prefix)
        log.info("opcode_map: extracting row 0x%X0–0x%XF (prefix=%s)", row, row,
                 f"0x{prefix:02X}" if prefix is not None else "none")
        try:
            result = docquery.query(
                query,
                schema=_OpcodeRow,
                system_prompt=_SYSTEM_PROMPT,
                settings=settings,
            )
            if isinstance(result, _OpcodeRow):
                new = 0
                for entry in result.entries:
                    key = (entry.prefix, entry.opcode)
                    if key not in seen_opcodes:
                        seen_opcodes.add(key)
                        all_entries.append(entry)
                        new += 1
                log.info("opcode_map: row 0x%Xx — %d entries (%d new)", row, len(result.entries), new)
            else:
                log.warning("opcode_map: row 0x%Xx — unexpected result type %s", row, type(result))
        except Exception as exc:
            log.warning("opcode_map: row 0x%Xx failed: %s", row, exc)

        if row < 15 and inter_row_sleep > 0:
            time.sleep(inter_row_sleep)

    log.info("opcode_map: extracted %d total entries", len(all_entries))
    return all_entries
