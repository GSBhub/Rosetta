"""Tests for rosetta_validate_sla.node — mocks subprocess, no Ghidra required."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from rosetta_validate_sla.node import validate_sla_node
from rosetta_validate_sla.sla.sleigh_compiler import SleighResult


def _state(**kwargs):
    base = {
        "lang_dir": None,
        "ghidra_home": "/fake/ghidra",
        "errors": [],
    }
    base.update(kwargs)
    return base


def _make_lang_dir(tmp_path: Path, slaspec_name: str = "TestISA.slaspec") -> Path:
    lang_dir = tmp_path / "data" / "languages"
    lang_dir.mkdir(parents=True)
    (lang_dir / slaspec_name).write_text("# fake slaspec")
    return lang_dir


# ---------------------------------------------------------------------------
# validate_sla_node
# ---------------------------------------------------------------------------

def test_validate_sla_node_missing_lang_dir():
    result = validate_sla_node(_state())
    assert result["compile_ok"] is False
    assert any("lang_dir" in e for e in result["errors"])


def test_validate_sla_node_missing_ghidra_home(tmp_path):
    lang_dir = _make_lang_dir(tmp_path)
    result = validate_sla_node(_state(lang_dir=str(lang_dir), ghidra_home=None))
    assert result["compile_ok"] is False
    assert any("ghidra_home" in e for e in result["errors"])


def test_validate_sla_node_no_slaspec(tmp_path):
    lang_dir = tmp_path / "empty"
    lang_dir.mkdir()
    result = validate_sla_node(_state(lang_dir=str(lang_dir)))
    assert result["compile_ok"] is False
    assert any("no .slaspec" in e for e in result["errors"])


def test_validate_sla_node_success(tmp_path):
    lang_dir = _make_lang_dir(tmp_path)
    ok_result = SleighResult(returncode=0, stdout="", stderr="")

    with patch("rosetta_validate_sla.sla.sleigh_compiler.compile_slaspec", return_value=ok_result):
        result = validate_sla_node(_state(lang_dir=str(lang_dir)))

    assert result["compile_ok"] is True
    assert result["compile_errors"] == []
    assert result["errors"] == []


def test_validate_sla_node_compile_failure(tmp_path):
    lang_dir = _make_lang_dir(tmp_path)
    fail_result = SleighResult(
        returncode=1, stdout="", stderr="",
        errors=["undefined symbol 'foo'", "parse error line 42"],
    )

    with patch("rosetta_validate_sla.sla.sleigh_compiler.compile_slaspec", return_value=fail_result):
        result = validate_sla_node(_state(lang_dir=str(lang_dir)))

    assert result["compile_ok"] is False
    assert "undefined symbol 'foo'" in result["compile_errors"]
    assert "parse error line 42" in result["compile_errors"]


def test_validate_sla_node_returns_no_new_errors_on_success(tmp_path):
    # Prior errors accumulate via LangGraph's operator.add reducer, not the node.
    lang_dir = _make_lang_dir(tmp_path)
    ok_result = SleighResult(returncode=0, stdout="", stderr="")

    with patch("rosetta_validate_sla.sla.sleigh_compiler.compile_slaspec", return_value=ok_result):
        result = validate_sla_node(_state(lang_dir=str(lang_dir), errors=["prior"]))

    assert result["errors"] == []
    assert result["compile_ok"] is True


# ---------------------------------------------------------------------------
# compile_slaspec via subprocess mock
# ---------------------------------------------------------------------------

def test_compile_slaspec_success(tmp_path):
    from rosetta_validate_sla.sla.sleigh_compiler import compile_slaspec

    slaspec = tmp_path / "test.slaspec"
    slaspec.write_text("# fake")
    ghidra = Path("/fake/ghidra")
    fake_sleigh = tmp_path / "sleigh"
    fake_sleigh.write_text("#!/bin/sh")

    with patch("rosetta_validate_sla.sla.sleigh_compiler._find_sleigh", return_value=fake_sleigh), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = compile_slaspec(slaspec, ghidra)

    assert result.ok is True
    assert result.errors == []


def test_compile_slaspec_failure(tmp_path):
    from rosetta_validate_sla.sla.sleigh_compiler import compile_slaspec

    slaspec = tmp_path / "test.slaspec"
    slaspec.write_text("# fake")
    ghidra = Path("/fake/ghidra")
    fake_sleigh = tmp_path / "sleigh"
    fake_sleigh.write_text("#!/bin/sh")
    error_output = "ERROR: undefined symbol at line 5\nERROR: parse failed"

    with patch("rosetta_validate_sla.sla.sleigh_compiler._find_sleigh", return_value=fake_sleigh), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr=error_output)
        result = compile_slaspec(slaspec, ghidra)

    assert result.ok is False
    assert len(result.errors) > 0
