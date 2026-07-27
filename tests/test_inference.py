"""Starter-config inference: rendering and the reviewability contract.

The inferred config is a starting point a human reviews, not a config to trust
blindly — a wrong decode field silently corrupts every instruction's decode
pattern — so these tests pin the evidence-carrying output as much as the values.
"""

from __future__ import annotations

import tomllib

from rosetta.inference import render_config

SCORED = [
    {"name": "syntax-anchor", "pattern": r"Syntax\s+([A-Z][A-Z0-9]*)", "entities": 238,
     "attributed": 389, "blocks": 521, "owners": 213, "precision": 0.895,
     "recall": 0.747, "score": 0.6682},
    {"name": "section-inline", "pattern": r"\n(\w+)\n", "entities": 22, "attributed": 23,
     "blocks": 521, "owners": 5, "precision": 0.227, "recall": 0.044, "score": 0.01},
]
DECODE = {
    "parallel_field": {"value": "p", "why": "1-bit field at bit 0 in 340/387 encodings"},
    "predicate_fields": {"value": ["creg", "z"], "why": "leading fields in >=75%"},
    "endian": {"value": "little", "why": "12 'little-endian' vs 4 'big-endian' mentions"},
    "field_attachments": {"value": {"s": ["0", "1"]}, "why": "1-bit selector fields"},
}


def _rendered(**kw):
    text = render_config("MyISA", "manuals/m.pdf", "dbs/myisa", SCORED, DECODE, **kw)
    return text, tomllib.loads(text)


def test_rendered_config_is_valid_toml_with_the_winning_pattern():
    text, cfg = _rendered()
    assert cfg["name"] == "MyISA"
    assert cfg["instruction_pattern"] == SCORED[0]["pattern"]
    assert cfg["decode"]["parallel_field"] == "p"
    assert cfg["decode"]["predicate_fields"] == ["creg", "z"]
    assert cfg["decode"]["field_attachments"] == {"s": ["0", "1"]}
    assert "REVIEW BEFORE USE" in text


def test_runners_up_and_evidence_are_kept_as_comments():
    """A reviewer needs to see what else was considered and why this won."""
    text, _ = _rendered()
    assert "section-inline" in text and "<-- chosen" in text
    for info in DECODE.values():
        assert info["why"] in text


def test_low_confidence_pattern_is_flagged():
    weak = [{**SCORED[0], "score": 0.05}]
    text = render_config("W", "m.pdf", "dbs/w", weak, {})
    assert "WARNING" in text


def test_non_instruction_entity_uses_the_generic_table():
    _text, cfg = _rendered(entity_name="register")
    assert "instruction_pattern" not in cfg
    assert cfg["entity_patterns"]["register"] == SCORED[0]["pattern"]


def test_renders_without_decode_candidates():
    text = render_config("Bare", "m.pdf", "dbs/bare", SCORED, {})
    cfg = tomllib.loads(text)
    assert "decode" not in cfg and cfg["name"] == "Bare"
