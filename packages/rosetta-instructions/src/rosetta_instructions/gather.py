"""Gathers full encoding details for a single instruction from ChromaDB."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langsmith import traceable
from rosetta_schemas.models import InstructionDef

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?})\s*```', re.DOTALL)

# LLMs often emit camelCase or alternative names for schema fields.
_KEY_MAP: dict[str, str] = {
    # mnemonic
    "instruction":      "mnemonic",
    "name":             "mnemonic",
    "opcode":           "mnemonic",
    # encoding_bits
    "encodingBits":     "encoding_bits",
    "encodingWidth":    "encoding_bits",
    "encoding_width":   "encoding_bits",
    "instructionWidth": "encoding_bits",
    "width":            "encoding_bits",
    "size":             "encoding_bits",
    "encoding":         "encoding_bits",
    # bit_fields
    "bitFields":        "bit_fields",
    "fields":           "bit_fields",
    "format":           "bit_fields",
    "encodingFormat":   "bit_fields",
    # bit_constraints
    "bitConstraints":   "bit_constraints",
    "requiredBits":     "bit_constraints",
    "required_bits":    "bit_constraints",
    "fixedBits":        "bit_constraints",
    # semantics (many synonyms)
    "description":      "semantics",
    "operation":        "semantics",
    "meaning":          "semantics",
    "behavior":         "semantics",
    "behaviour":        "semantics",
    "explanation":      "semantics",
    "summary":          "semantics",
    "detail":           "semantics",
    "details":          "semantics",
    # pcode_hint
    "pcodeHint":        "pcode_hint",
    "pcode":            "pcode_hint",
    "pseudocode":       "pcode_hint",
    # assembly variants
    "assembly":         "variants",
    "syntax":           "variants",
    "assemblySyntax":   "variants",
    "encodings":        "variants",
}


def _normalize(d: dict) -> dict:
    """Remap LLM key variations → snake_case Pydantic fields; coerce nested values."""
    out = {_KEY_MAP.get(k, k): v for k, v in d.items()}

    # operands: list[str] — LLMs sometimes return list[{name, type, ...}]
    if "operands" in out and out["operands"]:
        out["operands"] = [
            op["name"] if isinstance(op, dict) else str(op)
            for op in out["operands"]
        ]

    # variants: must be list[str]
    if "variants" in out:
        v = out["variants"]
        if isinstance(v, str):
            out["variants"] = [v]
        elif isinstance(v, list):
            out["variants"] = [str(x) for x in v]

    # bit_fields: dict[str, str] ("high:low") — LLMs sometimes return dict[str, {high, low}]
    if "bit_fields" in out and isinstance(out["bit_fields"], dict):
        bf: dict[str, str] = {}
        for k, v in out["bit_fields"].items():
            if isinstance(v, dict):
                h = v.get("high", v.get("msb", v.get("start", 0)))
                lo = v.get("low", v.get("lsb", v.get("end", 0)))
                bf[k] = f"{h}:{lo}"
            else:
                bf[k] = str(v)
        out["bit_fields"] = bf

    # bit_constraints: dict[str, str] — coerce any non-string values
    if "bit_constraints" in out and isinstance(out["bit_constraints"], dict):
        out["bit_constraints"] = {
            k: str(v.get("value", v.get("val", next(iter(v.values()), "")))
               if isinstance(v, dict) else v)
            for k, v in out["bit_constraints"].items()
        }

    # encoding_bits: must be int; sometimes arrives as string "32"
    if "encoding_bits" in out:
        try:
            out["encoding_bits"] = int(out["encoding_bits"])
        except (TypeError, ValueError):
            out["encoding_bits"] = 32

    # semantics fallback: use any remaining text-valued key if semantics is missing
    if "semantics" not in out:
        for fallback in ("notes", "note", "comment", "text"):
            if fallback in out and out[fallback]:
                out["semantics"] = str(out[fallback])
                break

    # Required-field defaults: if the LLM omitted them, provide safe values so
    # model_validate succeeds rather than failing all 3 attempts and returning None.
    if "encoding_bits" not in out:
        out["encoding_bits"] = 32
    if "semantics" not in out:
        mnemonic = out.get("mnemonic", "")
        out["semantics"] = f"ARM {mnemonic} instruction." if mnemonic else ""

    return out

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. "
    "Use ONLY the retrieved manual excerpts provided as context. "
    "If a field is not present in the context, leave it empty/zero. "
    "Do NOT use prior knowledge of any instruction set; do NOT invent bit fields, "
    "opcodes, or semantics. "
    "For bit_fields provide 'high:low' notation. "
    "For bit_constraints provide the required binary value. "
    "Return ONLY raw JSON matching the schema. "
    "Do NOT wrap the JSON in markdown code fences or backticks of any kind."
)

_SCHEMA_JSON = json.dumps(InstructionDef.model_json_schema(), indent=2)


def _extract_instruction(
    mnemonic: str,
    next_mnemonic: str | None,
    settings: Any,
) -> InstructionDef | None:
    """Extraction pipeline with fence stripping and key normalisation.

    Keeps the HumanMessage short ("Extract: {mnemonic}") so that models
    which switch to documentation mode on long queries stay in JSON mode.
    Disambiguation is a one-line note in the SystemMessage.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from docquery.embeddings.llm import get_llm
    from docquery.tools.retrieval_tools import make_similarity_tool

    rag_query = (
        f"From the manual context only, extract the encoding for the {mnemonic!r} instruction."
        f" Provide: encoding width in bits, bit fields (name → high:low), operands, semantics."
    )

    similarity_tool = make_similarity_tool(settings.vs, settings)
    llm = get_llm(settings)
    context = similarity_tool.invoke(rag_query)

    sys_content = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Output ONLY valid JSON matching this exact schema:\n{_SCHEMA_JSON}\n"
        "Do not include markdown code fences. Do not write any explanation."
    )
    if next_mnemonic:
        sys_content += f" Extract only '{mnemonic}', not '{next_mnemonic}'."

    validation_errors: list[str] = []
    for attempt in range(3):
        msgs = [
            SystemMessage(content=sys_content),
            HumanMessage(content=(
                f"Context:\n{context}\n\nExtract: {mnemonic}"
                + (
                    f"\n\nPrevious validation errors:\n" + "\n".join(validation_errors)
                    if validation_errors else ""
                )
            )),
        ]
        response = llm.invoke(msgs)
        raw = response.content.strip()

        # Strip markdown fences that some models add despite instructions.
        m = _FENCE_RE.search(raw)
        if m:
            raw = m.group(1).strip()

        try:
            # Try direct JSON parse first; fall back to key-normalised dict parse.
            try:
                result = InstructionDef.model_validate_json(raw)
            except Exception:
                d = _normalize(json.loads(raw))
                # Always use the known mnemonic — LLMs often return "ADD, ADDS (immediate)"
                # or other variants that would fail the required-field check.
                d["mnemonic"] = mnemonic
                result = InstructionDef.model_validate(d)
            return result
        except Exception as exc:
            validation_errors = [str(exc)]
            log.debug("_extract_instruction attempt %d/%d failed: %s", attempt + 1, 3, exc)

    return None


@traceable(
    run_type="chain",
    name="gather_instruction",
    process_inputs=lambda kw: {"current": kw.get("current"), "next": kw.get("next_mnemonic")},
)
def gather_instruction(
    current: str,
    next_mnemonic: str | None,
    settings: Any,
) -> InstructionDef:
    """Extract full InstructionDef for *current* from the vector store.

    *next_mnemonic* is passed as context so the LLM does not bleed fields
    from the following instruction into the current one.
    """
    try:
        result = _extract_instruction(current, next_mnemonic, settings)
        if result is not None:
            if not result.mnemonic or result.mnemonic.upper() == "UNKNOWN":
                result.mnemonic = current
            return result
    except Exception as exc:
        log.warning("gather failed for %s: %s", current, exc)

    log.warning("gather failed for %s: all extraction attempts returned None", current)
    return InstructionDef(
        mnemonic=current,
        encoding_bits=32,
        semantics=f"Stub for {current} — extraction failed.",
    )


@traceable(
    run_type="tool",
    name="enrich_pcode",
    process_inputs=lambda kw: {"mnemonic": kw["instr"].mnemonic if hasattr(kw.get("instr"), "mnemonic") else str(kw.get("instr"))},
)
def enrich_pcode(instr: InstructionDef, settings: Any) -> InstructionDef:
    """Fill instr.pcode_hint via a direct LLM call (no RAG needed)."""
    if instr.pcode_hint:
        return instr
    try:
        from rosetta_pcode.generator import generate_pcode
        instr.pcode_hint = generate_pcode(instr, settings)
    except Exception as exc:
        log.warning("pcode generation failed for %s: %s", instr.mnemonic, exc)
    return instr
