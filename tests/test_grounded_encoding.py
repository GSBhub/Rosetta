"""Grounded mnemonic + encoding extraction (no LLM).

Covers the deterministic path that replaces the hallucination-prone mnemonic
discovery and encoding transcription: mnemonics come from the ``instruction``
entity tags, bit fields/constraints from docquery's recovered ``encoding_grid``
bit diagrams.
"""

from types import SimpleNamespace

from rosetta_instructions.grounded import _record_to_fields, build_encoding_index
from rosetta_mnemonics.discovery import _grounded_mnemonics


def _store(docs):
    """Fake vector store whose _collection.get(where=...) filters by kind.

    docs: list of (kind, page, page_content, entity_instruction_or_None).
    """
    def get(where=None, include=None):
        kind = (where or {}).get("kind")
        rows = [d for d in docs if kind is None or d[0] == kind]
        return {
            "documents": [d[2] for d in rows],
            "metadatas": [
                {"kind": d[0], "page": d[1], **({"entity_instruction": d[3]} if d[3] else {})}
                for d in rows
            ],
        }
    return SimpleNamespace(_collection=SimpleNamespace(get=get))


# --- pure segment -> fields mapping ------------------------------------------

def test_record_to_fields_splits_constraints_and_operands():
    rec = {"width": 32, "segments": [
        {"name": None, "hi": 31, "lo": 27, "value": "11110"},   # fixed opcode bits
        {"name": "Rn", "hi": 18, "lo": 16, "value": None},      # operand field
        {"name": "Rd", "hi": 11, "lo": 7, "value": None},
    ]}
    out = _record_to_fields(rec)
    assert out["encoding_bits"] == 32
    assert out["bit_constraints"] == {"op_31_27": "11110"}
    assert out["bit_fields"]["op_31_27"] == "31:27"
    assert out["bit_fields"]["Rn"] == "18:16"
    assert out["operands"] == ["Rn", "Rd"]


def test_record_with_no_fields_is_none():
    assert _record_to_fields({"width": 16, "segments": []}) is None


# --- grounded mnemonics from entity tags -------------------------------------

def test_grounded_mnemonics_from_instruction_entities():
    vs = _store([
        ("register_fields", 5, "ROW ...", None),
        ("chunk", 100, "prose", "ADC"),
        ("chunk", 101, "prose", "ADD;ADR"),        # ;-joined
        ("chunk", 102, "prose", "adc"),             # dup, different case
    ])
    settings = SimpleNamespace(vs=vs)
    assert _grounded_mnemonics(settings) == ["ADC", "ADD", "ADR"]


def test_grounded_mnemonics_empty_without_tags():
    vs = _store([("register_fields", 5, "ROW ...", None)])
    assert _grounded_mnemonics(SimpleNamespace(vs=vs)) == []


# --- encoding index: instruction page -> its recovered bit diagram -----------

def test_build_encoding_index_matches_by_page():
    enc = "ENCODING 32-bit: bits[31:27]=11110 Rn[18:16] Rd[11:7]"
    vs = _store([
        ("chunk", 187, "ADC prose", "ADC"),
        ("encoding_grid", 187, enc, None),
        ("chunk", 999, "orphan prose", "ORPHAN"),   # no encoding on its page
    ])
    idx = build_encoding_index(SimpleNamespace(vs=vs))
    assert set(idx) == {"ADC"}                       # ORPHAN has no diagram
    assert len(idx["ADC"]) == 1
    assert idx["ADC"][0]["encoding_bits"] == 32
    assert idx["ADC"][0]["bit_constraints"] == {"op_31_27": "11110"}
    assert idx["ADC"][0]["operands"] == ["Rn", "Rd"]


def test_build_encoding_index_returns_all_page_encodings():
    # A C6x mnemonic maps to many opcode-map encodings — keep them all, not just
    # the widest.
    vs = _store([
        ("chunk", 50, "ADD prose", "ADD"),
        ("encoding_grid", 50, "ENCODING 32-bit: bits[31:27]=11110 bits[11:5]=0000011 imm[4:0]", None),
        ("encoding_grid", 50, "ENCODING 32-bit: bits[31:27]=11110 bits[11:5]=0100011 imm[4:0]", None),
    ])
    idx = build_encoding_index(SimpleNamespace(vs=vs))
    assert len(idx["ADD"]) == 2
    ops = {e["bit_constraints"]["op_11_5"] for e in idx["ADD"]}
    assert ops == {"0000011", "0100011"}
