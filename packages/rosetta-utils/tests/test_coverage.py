"""Tests for the rosetta coverage gate (rosetta_utils.coverage)."""

from types import SimpleNamespace
from unittest.mock import patch

from docquery.config import EntityRule

from rosetta_utils.coverage import check_coverage, expected_from_outline


def _settings(rules=None):
    return SimpleNamespace(entity_rules=rules or [])


_RULE = EntityRule(name="instruction", pattern=r"^A7\.7\.\d+\s+([A-Z][A-Z0-9]+)")


def test_expected_from_outline_counts_rule_matches():
    outline = [
        {"title": "A7.7.1 ADD", "page": 1, "level": 2},
        {"title": "A7.7.2 SUB", "page": 2, "level": 2},
        {"title": "Chapter Overview", "page": 1, "level": 1},  # no match
    ]
    with patch("docquery.outline", return_value=outline):
        assert expected_from_outline(_settings([_RULE]), "instruction") == 2


def test_expected_from_outline_none_without_matching_rule():
    with patch("docquery.outline", return_value=[{"title": "x", "page": 1, "level": 1}]):
        assert expected_from_outline(_settings([]), "instruction") is None


def test_check_coverage_pass():
    cov = {"instruction": {"count": 95, "by_section": {}}}
    with patch("docquery.coverage", return_value=cov):
        ok, msg = check_coverage(_settings([_RULE]), expected=100, threshold=0.9)
    assert ok and "OK" in msg


def test_check_coverage_fails_below_threshold():
    cov = {"instruction": {"count": 50, "by_section": {}}}
    with patch("docquery.coverage", return_value=cov):
        ok, msg = check_coverage(_settings([_RULE]), expected=100, threshold=0.9)
    assert not ok and "LOW" in msg


def test_check_coverage_skips_without_reference():
    cov = {"instruction": {"count": 7, "by_section": {}}}
    with patch("docquery.coverage", return_value=cov), \
         patch("docquery.outline", return_value=[]):
        ok, msg = check_coverage(_settings([_RULE]))
    assert ok and "no reference" in msg


def test_check_coverage_uses_outline_estimate():
    cov = {"instruction": {"count": 2, "by_section": {}}}
    outline = [{"title": "A7.7.1 ADD", "page": 1, "level": 2},
               {"title": "A7.7.2 SUB", "page": 2, "level": 2}]
    with patch("docquery.coverage", return_value=cov), \
         patch("docquery.outline", return_value=outline):
        ok, msg = check_coverage(_settings([_RULE]), threshold=1.0)
    assert ok and "2/2" in msg
