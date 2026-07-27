"""Infer a starter ISA config (examples/*.toml) from a manual.

Authoring the per-ISA config by hand means guessing an entity regex and reading
bit-diagram conventions out of a 700-page PDF. Most of it is recoverable: the
entity pattern is *scored* against the document's own structure (see
:mod:`docquery._infer`), and the ``[decode]`` fields fall out of the encodings
that recovery already produced.

The output is a **starter** config to review, not a config to trust blindly. A
wrong ``parallel_field`` or ``predicate_fields`` corrupts every constructor's
decode pattern, and that failure is invisible until you score against ground
truth — so each inferred value is written with the evidence behind it.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

# A field must appear in at least this share of encodings to look structural
# (present in every instruction) rather than incidental to a few.
_UBIQUITY = 0.75


def _recovered_encodings(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Every recovered bit diagram in the manual, as parsed segment records."""
    import fitz  # pymupdf

    from docquery._bitgrid import encodings_from_words

    out: list[dict[str, Any]] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            try:
                out += [
                    e for e in encodings_from_words(page.get_text("words"))
                    if any(s.get("value") for s in e["segments"])
                ]
            except Exception:  # noqa: BLE001 - a bad page must not sink inference
                continue
    return out


def infer_decode_fields(pdf_path: str | Path, page_text: dict[int, str]) -> dict[str, Any]:
    """Candidate ``[decode]`` values, each with the evidence that suggested it.

    Returns ``{key: {"value": ..., "why": ...}}``. Everything here is a heuristic
    over recovered geometry and page text; the caller is expected to surface the
    reasoning so a human can accept or reject each one.
    """
    encodings = _recovered_encodings(pdf_path)
    found: dict[str, Any] = {}
    if not encodings:
        return found

    # Analyse only the dominant instruction width. Mixing widths dilutes every
    # ratio (a 16-bit compact form shares no fields with the 32-bit one) and a
    # single mis-parsed diagram would otherwise set the top bit for everything.
    width_counts = Counter(e.get("width") or 0 for e in encodings)
    main_width = width_counts.most_common(1)[0][0]
    encodings = [e for e in encodings if (e.get("width") or 0) == main_width]
    n = len(encodings)
    if not n:
        return found
    named: Counter[str] = Counter()
    widths: dict[str, Counter[int]] = {}
    los: dict[str, Counter[int]] = {}
    his: dict[str, Counter[int]] = {}
    for enc in encodings:
        for seg in enc["segments"]:
            name = seg.get("name")
            if not name or seg.get("value") is not None:
                continue
            named[name] += 1
            widths.setdefault(name, Counter())[seg["hi"] - seg["lo"] + 1] += 1
            los.setdefault(name, Counter())[seg["lo"]] += 1
            his.setdefault(name, Counter())[seg["hi"]] += 1

    ubiquitous = [f for f, c in named.items() if c >= _UBIQUITY * n]
    top_bit = main_width - 1

    # A 1-bit field pinned at bit 0 across nearly every encoding is the classic
    # VLIW parallel/"next instruction issues with me" marker.
    for f in ubiquitous:
        if widths[f].most_common(1)[0][0] == 1 and los[f].most_common(1)[0][0] == 0:
            found["parallel_field"] = {
                "value": f,
                "why": f"1-bit field at bit 0 in {named[f]}/{n} encodings",
            }
            break

    # Predication sits in the leading bits, ahead of the opcode, in every
    # instruction (e.g. creg/z on TI C6x).
    leading = sorted(
        (f for f in ubiquitous if his[f].most_common(1)[0][0] >= top_bit - 4),
        key=lambda f: -his[f].most_common(1)[0][0],
    )
    if leading:
        found["predicate_fields"] = {
            "value": leading[:2],
            "why": f"leading fields present in >={int(_UBIQUITY * 100)}% of {n} encodings "
                   f"(top bit {top_bit})",
        }

    # 1-bit ubiquitous fields that are neither the parallel bit nor part of the
    # predication group are selector-shaped (register side/bank/cross-path) —
    # good `attach names` candidates. Most common first.
    spoken_for = {(found.get("parallel_field") or {}).get("value")}
    spoken_for |= set((found.get("predicate_fields") or {}).get("value") or [])
    selectors = sorted(
        (f for f in ubiquitous
         if f not in spoken_for and widths[f].most_common(1)[0][0] == 1),
        key=lambda f: -named[f],
    )
    if selectors:
        found["field_attachments"] = {
            "value": {f: ["0", "1"] for f in selectors[:2]},
            "why": "1-bit selector fields; replace the placeholder names with the "
                   "manual's meaning (e.g. register side 1/2)",
        }

    text = "\n".join(page_text.values())
    little, big = len(re.findall(r"little[- ]endian", text, re.I)), len(re.findall(r"big[- ]endian", text, re.I))
    if little or big:
        found["endian"] = {
            "value": "little" if little >= big else "big",
            "why": f"{little} 'little-endian' vs {big} 'big-endian' mentions",
        }

    # "Compatibility  C62x, C64x, C67x, and C67x+ CPU" — device-family lists.
    fams: Counter[str] = Counter()
    for m in re.finditer(r"Compatibilit(?:y|ies)[^\n]*", text, re.I):
        # No trailing \b: a '+' suffix (C64x+) is not a word character, so a
        # boundary after it can never match and the variant would be truncated.
        for fam in re.findall(r"\b([A-Z]\d{2,4}[A-Za-z]*\+?)", m.group(0)):
            fams[fam] += 1
    if len(fams) >= 2:
        found["isa_variants"] = {
            "value": [f for f, _ in fams.most_common(8)],
            "why": f"families named on 'Compatibility' lines ({sum(fams.values())} matches)",
        }
    return found


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return str(value)


def render_config(
    name: str,
    manual: str,
    db: str,
    scored: list[dict[str, Any]],
    decode: dict[str, Any],
    entity_name: str = "instruction",
) -> str:
    """Render a reviewable starter config, with the evidence kept as comments."""
    lines = [
        f"# {name} — starter config inferred by `rosetta infer-config`.",
        "#",
        "# REVIEW BEFORE USE. Every value below is a heuristic over the manual's",
        "# recovered structure; a wrong decode field silently corrupts every",
        "# instruction's decode pattern. The evidence for each is noted inline.",
        "",
        f'name = "{name}"',
        f'manual = "{manual}"',
        f'db = "{db}"',
        "",
    ]

    if scored:
        best = scored[0]
        lines += [
            "# Entity pattern ranked by precision x recall against the document's own",
            "# structure blocks (an entity rule should tag things that own a diagram or",
            "# table, not every capitalised line). Runners-up are kept for comparison.",
        ]
        for r in scored[:4]:
            mark = "  <-- chosen" if r is best else ""
            lines.append(
                f"#   {r['name']:<20} score={r['score']:<7} prec={r['precision']:<5} "
                f"rec={r['recall']:<5} entities={r['entities']:<5} owners={r['owners']}{mark}"
            )
        key = "instruction_pattern" if entity_name == "instruction" else None
        if key:
            lines.append(f"instruction_pattern = '{best['pattern']}'")
        else:
            lines += ["[entity_patterns]", f"{entity_name} = '{best['pattern']}'"]
        if best["score"] < 0.2:
            lines += [
                "# WARNING: low score — this manual's headings do not match any known",
                "# shape. Expect poor grounding until the pattern is corrected by hand.",
            ]
        lines.append("")

    if decode:
        lines.append("[decode]")
        simple = {k: v for k, v in decode.items() if k != "field_attachments"}
        for key, info in simple.items():
            lines.append(f"# {info['why']}")
            lines.append(f"{key} = {_toml_value(info['value'])}")
        if attach := decode.get("field_attachments"):
            lines += ["", f"# {attach['why']}", "[decode.field_attachments]"]
            for field, names in attach["value"].items():
                lines.append(f"{field} = {_toml_value(names)}")
        lines.append("")

    return "\n".join(lines) + "\n"
