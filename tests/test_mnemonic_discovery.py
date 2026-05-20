"""Tests for the LangGraph-based mnemonic discovery module."""

import pytest

from rosetta.extraction.mnemonic_discovery import _clean_mnemonic, _STRATEGIES


def test_clean_mnemonic_valid():
    assert _clean_mnemonic("add") == "ADD"
    assert _clean_mnemonic("  LDR  ") == "LDR"
    assert _clean_mnemonic("VLDM") == "VLDM"


def test_clean_mnemonic_rejects_invalid():
    assert _clean_mnemonic("ADD Rd, Rn") == ""        # contains spaces
    assert _clean_mnemonic("123") == ""               # starts with digit
    assert _clean_mnemonic("") == ""                  # empty
    assert _clean_mnemonic("AVERYLONGMNEMONIC") == "" # > 16 chars in base name
    assert _clean_mnemonic("ADD, Rn") == ""           # comma


def test_clean_mnemonic_rejects_punctuation():
    assert _clean_mnemonic("B{cond}") == ""   # braces not allowed


def test_clean_mnemonic_accepts_dot_suffixes():
    # Dot-suffixed mnemonics are valid ARM/NEON constructs
    assert _clean_mnemonic("VABS.F32") == "VABS.F32"
    assert _clean_mnemonic("AESD.8") == "AESD.8"
    assert _clean_mnemonic("ADD.W") == "ADD.W"
    assert _clean_mnemonic("DLSTP.") == "DLSTP."
    assert _clean_mnemonic("SHA1C.32") == "SHA1C.32"


def test_strategies_nonempty():
    assert len(_STRATEGIES) >= 5


def test_strategies_are_strings():
    for s in _STRATEGIES:
        assert isinstance(s, str) and len(s) > 10
