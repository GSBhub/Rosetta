"""`--entity-pattern NAME=REGEX` parsing.

`=` is common inside regexes, so a naive split on the first one silently
produces a garbage rule instead of erroring — the failure only surfaces later
as "no entities tagged".
"""

from __future__ import annotations

import pytest

from rosetta.cli import parse_entity_patterns


def test_named_spec_splits_on_the_kind():
    assert parse_entity_patterns(["register=abc(\\w+)"]) == [("register", "abc(\\w+)")]


def test_bare_regex_defaults_to_instruction():
    assert parse_entity_patterns([r"Syntax\s+([A-Z]+)"]) == [
        ("instruction", r"Syntax\s+([A-Z]+)")
    ]


def test_regex_containing_equals_is_not_split():
    """A lookahead must survive intact rather than becoming the rule name."""
    pat = r"Syntax\s+([A-Z]+)(?=\s)"
    assert parse_entity_patterns([pat]) == [("instruction", pat)]


def test_lookbehind_and_inline_equality_survive():
    for pat in [r"(?<=\n)([A-Z]+)", r"bits\[31:0\]=([01]+)"]:
        assert parse_entity_patterns([pat]) == [("instruction", pat)]


def test_repeatable_specs_keep_their_kinds():
    assert parse_entity_patterns(["instruction=A(\\w+)", "register=B(\\w+)"]) == [
        ("instruction", "A(\\w+)"),
        ("register", "B(\\w+)"),
    ]


def test_empty_pattern_is_rejected():
    with pytest.raises(ValueError, match="NAME=REGEX"):
        parse_entity_patterns(["register="])
