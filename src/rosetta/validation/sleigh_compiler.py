"""Wraps the Ghidra SLEIGH compiler to validate .slaspec files."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class SleighResult:
    returncode: int
    stdout: str
    stderr: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.errors


def _find_sleigh(ghidra_home: Path) -> Path:
    candidates = [
        ghidra_home / "support" / "sleigh",
        ghidra_home / "support" / "sleigh.bat",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"sleigh compiler not found under {ghidra_home}/support/. "
        "Run scripts/setup_ghidra.sh first."
    )


def compile_slaspec(slaspec_path: Path, ghidra_home: Path) -> SleighResult:
    """
    Compile slaspec_path with the Ghidra SLEIGH compiler.
    The compiler is run in the directory containing the .slaspec so that
    relative @include paths resolve correctly.
    """
    sleigh = _find_sleigh(ghidra_home)
    cwd = slaspec_path.parent

    cmd = [str(sleigh), str(slaspec_path.name)]
    log.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )

    errors = [
        line for line in (proc.stdout + proc.stderr).splitlines()
        if "error" in line.lower() or "ERROR" in line
    ]

    result = SleighResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        errors=errors,
    )

    if result.ok:
        log.info("SLEIGH compile OK: %s", slaspec_path)
    else:
        log.warning("SLEIGH compile FAILED (%d errors)", len(errors))
        for err in errors:
            log.warning("  %s", err)

    return result
