"""Chunk-level opcode table heuristic and direct-LLM extraction.

Used by scanner_graph.py to bypass similarity search for the initial scan pass:
every chunk in the vector store is inspected locally (heuristic) and those that
look like opcode tables are passed directly to the LLM for structured extraction.
No retrieval step is involved — the chunk text IS the context.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from rosetta_schemas.models import OpcodeDef

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic: does this chunk look like an opcode table?
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(
    r"\b[0-9A-Fa-f]{2}[Hh]\b"   # 4AH style
    r"|0x[0-9A-Fa-f]{2}\b"       # 0x4A style
    r"|\$[0-9A-Fa-f]{2}\b"       # $4A style
    r"|\b[0-9A-Fa-f]{2}\b",      # bare 4A (must appear several times to matter)
)

_MNEMONIC_RE = re.compile(
    r"\b("
    r"LDA|STA|LDX|STX|LDY|STY|STZ|"
    r"ADC|SBC|AND|ORA|EOR|CMP|CPX|CPY|BIT|TRB|TSB|"
    r"INC|DEC|INX|DEX|INY|DEY|"
    r"ASL|LSR|ROL|ROR|"
    r"JMP|JSR|JSL|JML|RTS|RTI|RTL|"
    r"BEQ|BNE|BCC|BCS|BMI|BPL|BVC|BVS|BRA|BRL|"
    r"PHA|PHP|PLA|PLP|PHX|PHY|PLX|PLY|PHB|PHD|PHK|PLB|PLD|"
    r"TAX|TXA|TAY|TYA|TSX|TXS|TXY|TYX|TCD|TCS|TDC|TSC|XBA|XCE|"
    r"REP|SEP|MVN|MVP|COP|WDM|STP|WAI|NOP|"
    r"SEC|CLC|SEI|CLI|CLV|SED|CLD|"
    # M37700-specific
    r"ADDC|SUBC|ADDW|SUBW|MPY|DIV|NEG|LINK|UNLK|TAS|TAD|"
    r"CLRP|SETP|LDPL|STPL"
    r")\b",
    re.IGNORECASE,
)

_MIN_HEX = 6
_MIN_MNEMONICS = 2


def looks_like_opcode_table(text: str) -> bool:
    """Return True if *text* likely contains rows of an opcode table."""
    return (
        len(_HEX_RE.findall(text)) >= _MIN_HEX
        and len(_MNEMONIC_RE.findall(text)) >= _MIN_MNEMONICS
    )


# ---------------------------------------------------------------------------
# Chunk retrieval from ChromaDB
# ---------------------------------------------------------------------------

def get_all_chunks(settings: Any) -> list[dict[str, str]]:
    """Return every chunk in the vector store as {id, text} dicts."""
    try:
        # settings.vs is a langchain_chroma.Chroma; _collection is the raw ChromaDB client
        result = settings.vs._collection.get(include=["documents", "ids"])
        chunks = [
            {"id": id_, "text": doc}
            for id_, doc in zip(result["ids"], result["documents"])
            if doc and doc.strip()
        ]
        log.info("chunk_scanner: %d chunks in collection", len(chunks))
        return chunks
    except Exception as exc:
        log.error("chunk_scanner: failed to enumerate chunks: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Direct-LLM extraction (no retrieval step)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are an expert ISA analyst extracting opcode table data from processor reference manual text.

For EVERY opcode table entry you find in the provided text, return:
- opcode: integer byte value 0-255
- prefix: integer prefix byte if this is a prefixed instruction table (e.g. 0xCE), or null
- mnemonic: the instruction mnemonic (e.g. 'LDA') or 'UNK' if undefined/reserved
- mode: addressing mode abbreviation (imp, acc, imm, dp, dp,X, dp,Y, (dp), (dp,X), (dp),Y,
        abs, abs,X, abs,Y, (abs), (abs,X), (abs),Y, rel, rel16, long, long,X, long,Y,
        (long,X), (long),Y, sr, (sr),Y)
- operand_bytes: 0=implied/acc, 1=byte operand, 2=word operand, 3=long operand

If the text shows a section header like "CE Prefix Instructions" or "Extended Instruction Set"
set prefix on all entries from that table accordingly.

Return an empty entries list if the text contains no opcode table data.
Return ONLY JSON matching the schema.
"""


class _ChunkOpcodes(BaseModel):
    entries: list[OpcodeDef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"entries": data}
        if isinstance(data, dict) and "entries" not in data:
            for v in data.values():
                if isinstance(v, list):
                    return {"entries": v}
        return data


def _build_llm(settings: Any) -> Any:
    """Build a LangChain chat model from docquery LLM settings."""
    provider = (getattr(settings, "llm_provider", None) or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model    = getattr(settings, "llm_model", None)    or os.getenv("LLM_MODEL", "llama3")
    base_url = getattr(settings, "llm_base_url", None) or os.getenv("LLM_BASE_URL", "http://localhost:11434")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    if provider in ("openai", "azure"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    # fallback
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def extract_opcodes_from_chunk(
    chunk_text: str,
    settings: Any,
    chunk_id: str = "",
) -> list[OpcodeDef]:
    """Extract OpcodeDef entries from *chunk_text* via a direct LLM call.

    No similarity search is performed — the chunk text is the entire context.
    Uses langchain structured output (tool_calls / JSON mode) to get Pydantic
    validation and automatic coercion consistent with docquery's extraction.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm(settings)
    try:
        structured = llm.with_structured_output(_ChunkOpcodes)
        result = structured.invoke([
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=(
                "Extract all opcode table entries from the following text.\n\n"
                f"{chunk_text[:3000]}"
            )),
        ])
        if isinstance(result, _ChunkOpcodes):
            valid = [e for e in result.entries if 0 <= e.opcode <= 255]
            if valid:
                log.debug("chunk_scanner: %s → %d entries", chunk_id or "chunk", len(valid))
            return valid
    except Exception as exc:
        log.warning("chunk_scanner: extraction failed for %s: %s", chunk_id or "chunk", exc)
    return []
