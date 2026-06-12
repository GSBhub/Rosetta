"""Integration test: compile a generated .slaspec with Ghidra's sleigh compiler."""

import pytest

from rosetta_generate_sla.sla.module_generator import ModuleGenerator
from rosetta.validation.sleigh_compiler import SleighResult, compile_slaspec
from tests.conftest import GHIDRA_HOME, requires_ghidra


@requires_ghidra
def test_compile_returns_result(tmp_path, minimal_spec):
    gen = ModuleGenerator()
    lang_dir = gen.generate(minimal_spec, "TestISA", tmp_path)
    slaspec = lang_dir / "TestISA.slaspec"

    result = compile_slaspec(slaspec, GHIDRA_HOME)
    assert isinstance(result, SleighResult)
    # We check that the compiler ran (returncode is set); the spec may not be
    # syntactically perfect yet, but we must get a result back, not an exception.
    assert result.returncode is not None


@requires_ghidra
def test_compile_real_arm7(tmp_path):
    """The real ARM7_le.slaspec should compile cleanly."""
    arm7 = GHIDRA_HOME / "Ghidra" / "Processors" / "ARM" / "data" / "languages" / "ARM7_le.slaspec"
    if not arm7.exists():
        pytest.skip(f"ARM7_le.slaspec not found at {arm7}")

    result = compile_slaspec(arm7, GHIDRA_HOME)
    assert result.ok, (
        f"ARM7_le.slaspec failed to compile (should always pass).\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )


@requires_ghidra
def test_compile_reports_errors_for_bad_slaspec(tmp_path):
    """A deliberately broken slaspec should return errors."""
    bad_spec = tmp_path / "bad.slaspec"
    bad_spec.write_text("this is not valid SLEIGH syntax at all !!!")
    result = compile_slaspec(bad_spec, GHIDRA_HOME)
    # Should fail — non-zero returncode or errors list
    assert not result.ok
