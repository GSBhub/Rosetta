"""LangGraph-based multi-strategy mnemonic discovery.

A single RAG query (top_k=5) sees ~5000 chars of a ~2 MB manual and
consistently misses most mnemonics.  This module fans out across multiple
targeted query strategies, including NEON/VFP type-suffix variants,
and loops until all strategies are exhausted.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

import docquery

log = logging.getLogger(__name__)

_STRATEGIES: list[str] = [
    "List ALL instruction mnemonics defined in this ISA manual. Include every base mnemonic and every data-type-suffixed variant (e.g. VABS.F32, VADD.I8).",
    "List all data-processing instruction mnemonics: arithmetic (ADD, SUB, ADC, SBC, RSB, RSC, ADDW, SUBW), logical (AND, ORR, EOR, BIC, ORN), move (MOV, MVN, MOVW, MOVT), and comparison (CMP, CMN, TST, TEQ).",
    "List all load and store instruction mnemonics (LDR, STR, LDM, STM, PUSH, POP and all their variants including LDRB, LDRH, LDRSB, LDRSH, LDREX, STREX, LDA, STL).",
    "List all branch and control-flow instruction mnemonics (B, BL, BX, BLX, CBZ, CBNZ, TBB, TBH, IT, BAL, BFC, BFI, BIC).",
    "List all multiply and divide instruction mnemonics (MUL, MLA, MLS, UMULL, UMLAL, SMULL, SMLAL, SDIV, UDIV, SMLA, SMULW, SMLAD, SMUAD).",
    "List all saturating arithmetic instruction mnemonics (QADD, QSUB, QDADD, QDSUB, QADD8, QADD16, QSUB8, QSUB16, QASX, QSAX, SADD8, SADD16, SSUB8, SSUB16, UADD8, UADD16).",
    "List all system, privileged, coprocessor, hint, and barrier instruction mnemonics (MRC, MCR, MCRR, MRRC, SVC, SMC, HVC, ISB, DSB, DMB, PLD, PLDW, PLI, WFE, WFI, SEV, YIELD, NOP, DBG, CLREX).",
    "List all bit-field and packing instruction mnemonics (BFC, BFI, UBFX, SBFX, PKHBT, PKHTB, REV, REV16, REVSH, RBIT, SXTB, SXTH, SXTB16, UXTB, UXTH, UXTB16, SEL, CLZ).",
    "List all VFP and NEON vector instruction base mnemonics that begin with V: VADD, VSUB, VMUL, VMOV, VMVN, VABS, VNEG, VCMP, VCMPE, VDIV, VSQRT, VFMA, VFMS, VFNMA, VFNMS, VMLA, VMLS, VMLA, VMLA, VNMLA, VNMLS, VNMUL.",
    "List all NEON load and store instruction mnemonics beginning with V: VLD1, VLD2, VLD3, VLD4, VST1, VST2, VST3, VST4, VLDMIA, VLDMDB, VSTMIA, VSTMDB, VLDR, VSTR, VPUSH, VPOP.",
    "List all NEON integer SIMD mnemonics with element-size data type suffixes. For each mnemonic list all type variants. Examples: VADD.I8, VADD.I16, VADD.I32, VADD.I64, VMUL.I8, VMUL.I16, VMUL.I32, VAND, VORR, VEOR, VBIC, VORN, VDUP.8, VDUP.16, VDUP.32.",
    "List all NEON floating-point instruction mnemonics with type suffixes: VADD.F32, VSUB.F32, VMUL.F32, VABS.F32, VNEG.F32, VCMP.F32, VDIV.F32, VSQRT.F32, VFMA.F32, VFMS.F32, VCVT.F32.S32, VCVT.S32.F32, VCVT.F16.F32, VMLA.F32, VMLS.F32.",
    "List all NEON shift instruction mnemonics with element-size suffixes: VSHL.I8, VSHR.S8, VSHR.U8, VRSHL.S16, VRSHR.U16, VSRA.S32, VRSRA.U32, VSLI.8, VSRI.16, VQSHL.S8, VQSHL.U16, VQRSHL.S32, VSHRN.I16, VRSHRN.I32.",
    "List all NEON comparison instruction mnemonics with type suffixes: VCEQ.I8, VCEQ.F32, VCGE.S8, VCGE.U16, VCGE.F32, VCGT.S32, VCGT.U8, VCGT.F32, VCLE.S16, VCLE.F32, VCLT.S8, VCLT.F32, VACGE.F32, VACGT.F32.",
    "List all NEON permute, table-lookup, and miscellaneous mnemonics: VTBL.8, VTBX.8, VTRN.8, VTRN.16, VTRN.32, VZIP.8, VZIP.16, VZIP.32, VUZP.8, VUZP.16, VSWP, VREV16.8, VREV32.8, VREV64.8, VEXT.8, VPADDL.S8, VPADAL.S8, VPADD.I8, VPADD.F32.",
    "List all NEON widening, narrowing, and long instruction mnemonics: VADDL.S8, VADDW.S16, VSUBL.U8, VSUBW.U16, VMOVL.S8, VMOVN.I16, VQMOVN.S16, VQMOVUN.S16, VMULL.S8, VMLAL.S8, VMLSL.S16, VQDMULL.S16, VQDMLAL.S16, VADDHN.I16, VSUBHN.I32.",
    "List all AES and SHA cryptographic extension mnemonics with type suffixes: AESD.8, AESE.8, AESIMC.8, AESMC.8, SHA1C.32, SHA1H.32, SHA1M.32, SHA1P.32, SHA1SU0.32, SHA1SU1.32, SHA256H.32, SHA256H2.32, SHA256SU0.32, SHA256SU1.32.",
    "List all VCVT conversion instruction mnemonics with type suffixes: VCVT.F32.S32, VCVT.F32.U32, VCVT.S32.F32, VCVT.U32.F32, VCVT.F16.F32, VCVT.F32.F16, VCVT.S16.F16, VCVT.U16.F16, VCVT.F16.S16, VCVT.F16.U16, VCVTB, VCVTT.",
    "List all NEON max, min, and absolute difference mnemonics: VMAX.S8, VMAX.U16, VMAX.F32, VMIN.S32, VMIN.U8, VMIN.F32, VABD.S8, VABD.U16, VABD.F32, VABA.S8, VABA.U16, VPMAX.S8, VPMIN.U16, VPMAX.F32, VPMIN.F32.",
]

_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Extract the complete list of instruction "
    "mnemonics defined in this manual. Return only JSON matching the schema. "
    "Include only base mnemonics (e.g. ADD, not ADDGE or ADDS). "
    "Upper-case all mnemonics."
)

_COUNT_SYSTEM_PROMPT = (
    "You are an expert ISA analyst. Answer with a single integer — "
    "the total number of unique instruction mnemonics defined in this ISA manual."
)

_VALID_MNEMONIC = re.compile(r"^[A-Z][A-Z0-9]{0,15}(?:\.[A-Z0-9\.]{0,10})?$")


class _MnemonicList(BaseModel):
    mnemonics: list[str]


class _InstructionCount(BaseModel):
    count: int


class _MnemonicState(TypedDict):
    settings: Any
    total_expected: int
    mnemonics: list[str]
    strategies: list[str]
    last_new_count: int
    done: bool


def _count_node(state: _MnemonicState) -> _MnemonicState:
    try:
        result = docquery.query(
            "How many unique instruction mnemonics are defined in this ISA manual? "
            "Return only a JSON object with a 'count' field containing the integer.",
            schema=_InstructionCount,
            system_prompt=_COUNT_SYSTEM_PROMPT,
            settings=state["settings"],
        )
        total = result.count if isinstance(result, _InstructionCount) else 50
    except Exception as exc:
        log.warning("Count query failed (%s); defaulting to 50", exc)
        total = 50

    log.info("Estimated instruction count: %d", total)
    return {**state, "total_expected": total}


def _fetch_node(state: _MnemonicState) -> _MnemonicState:
    strategies = list(state["strategies"])
    if not strategies:
        return {**state, "done": True, "last_new_count": 0}

    strategy = strategies.pop(0)
    log.info("Mnemonic strategy: %s", strategy[:80])

    existing = set(state["mnemonics"])
    new_mnemonics: list[str] = []

    try:
        result = docquery.query(
            strategy,
            schema=_MnemonicList,
            system_prompt=_SYSTEM_PROMPT,
            settings=state["settings"],
        )
        if isinstance(result, _MnemonicList):
            for m in result.mnemonics:
                cleaned = _clean_mnemonic(m)
                if cleaned and cleaned not in existing:
                    new_mnemonics.append(cleaned)
                    existing.add(cleaned)
    except Exception as exc:
        log.warning("Strategy fetch failed (%s)", exc)

    log.info("Found %d new mnemonic(s) (total: %d)", len(new_mnemonics), len(existing))
    return {
        **state,
        "strategies": strategies,
        "mnemonics": state["mnemonics"] + new_mnemonics,
        "last_new_count": len(new_mnemonics),
    }


def _check_node(state: _MnemonicState) -> _MnemonicState:
    unique = len(set(state["mnemonics"]))
    expected = max(state["total_expected"], 1)
    coverage = unique / expected
    done = len(state["strategies"]) == 0

    if done and coverage < 0.7:
        log.warning(
            "Low mnemonic coverage: found %d, expected ~%d (%.0f%%). "
            "Supplement with: rosetta ingest <supplement.pdf> --db <same-db-path>",
            unique, expected, coverage * 100,
        )
    else:
        log.info(
            "Check: unique=%d expected=%d coverage=%.0f%% strategies_left=%d → done=%s",
            unique, expected, coverage * 100, len(state["strategies"]), done,
        )
    return {**state, "done": done}


def _route(state: _MnemonicState) -> str:
    return END if state["done"] else "fetch"


def _clean_mnemonic(raw: str) -> str:
    m = raw.strip().upper()
    if " " in m or "," in m or "(" in m:
        return ""
    if _VALID_MNEMONIC.match(m):
        return m
    return ""


def discover_mnemonics(
    db_path: str,
    settings: Any,
    strategies: list[str] | None = None,
) -> list[str]:
    """Fan out across multiple RAG query strategies; return deduplicated uppercase mnemonics."""
    settings.db_path = db_path
    # settings.vs must already be set by the caller (mnemonics_node sets it via
    # get_chroma_wrapper before calling here). Do NOT call _build_chroma — that
    # overwrites settings.vs with the old SQLite-backed chroma and hangs.

    graph = StateGraph(_MnemonicState)
    graph.add_node("count", _count_node)
    graph.add_node("fetch", _fetch_node)
    graph.add_node("check", _check_node)

    graph.add_edge(START, "count")
    graph.add_edge("count", "fetch")
    graph.add_edge("fetch", "check")
    graph.add_conditional_edges("check", _route, {"fetch": "fetch", END: END})

    app = graph.compile()

    initial: _MnemonicState = {
        "settings": settings,
        "total_expected": 0,
        "mnemonics": [],
        "strategies": list(strategies if strategies is not None else _STRATEGIES),
        "last_new_count": 0,
        "done": False,
    }

    final = app.invoke(initial)
    result = sorted(set(m.upper() for m in final["mnemonics"] if m))
    log.info("Mnemonic discovery complete: %d unique mnemonics", len(result))
    return result
