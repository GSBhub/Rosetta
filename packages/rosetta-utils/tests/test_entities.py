"""Tests for section-aware entity enumeration (rosetta_utils.entities)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from docquery.config import ENTITY_PREFIX

from rosetta_utils.entities import (
    read_tagged_entities,
    read_tagged_entities_by_section,
)

_KEY = f"{ENTITY_PREFIX}instruction"


def _settings(metadatas):
    collection = MagicMock()
    collection.get.return_value = {"metadatas": metadatas}
    return SimpleNamespace(vs=SimpleNamespace(_collection=collection))


def test_flat_page_order_when_no_sections():
    # no section_order anywhere → pure page order (back-compat)
    metas = [{_KEY: "SUB", "page": 5}, {_KEY: "ADD", "page": 2}]
    assert read_tagged_entities(_settings(metas), "instruction") == ["ADD", "SUB"]


def test_section_order_overrides_page_order():
    # MEM (section_order 1, page 9) must come AFTER DP (section_order 0, page 5)
    # even though page-only order would interleave differently.
    metas = [
        {_KEY: "LDR", "page": 9, "section_order": 1, "section": "Memory"},
        {_KEY: "ADD", "page": 5, "section_order": 0, "section": "Data Processing"},
        {_KEY: "ADC", "page": 6, "section_order": 0, "section": "Data Processing"},
    ]
    out = read_tagged_entities(_settings(metas), "instruction")
    assert out == ["ADD", "ADC", "LDR"]


def test_none_when_untagged():
    assert read_tagged_entities(_settings([{"page": 1}]), "instruction") is None
    assert read_tagged_entities_by_section(_settings([{"page": 1}]), "instruction") is None


def test_by_section_groups_and_orders():
    metas = [
        {_KEY: "LDR;STR", "page": 9, "section_order": 1, "section": "Memory"},
        {_KEY: "ADD", "page": 5, "section_order": 0, "section": "Data Processing"},
    ]
    groups = read_tagged_entities_by_section(_settings(metas), "instruction")
    assert groups == [("Data Processing", ["ADD"]), ("Memory", ["LDR", "STR"])]


def test_by_section_unsectioned_uses_none_key():
    metas = [{_KEY: "ADD", "page": 1}]
    groups = read_tagged_entities_by_section(_settings(metas), "instruction")
    assert groups == [(None, ["ADD"])]
