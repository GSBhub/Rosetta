"""Wraps Ghidra's analyzeHeadless for disassembly of test binaries."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class HeadlessResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _find_headless(ghidra_home: Path) -> Path:
    for c in [ghidra_home / "support" / "analyzeHeadless", ghidra_home / "support" / "analyzeHeadless.bat"]:
        if c.exists():
            return c
    raise FileNotFoundError(f"analyzeHeadless not found under {ghidra_home}/support/.")


def run_headless(
    binary_path: Path,
    language_id: str,
    ghidra_home: Path,
    post_script: Path | None = None,
    project_dir: Path | None = None,
    project_name: str = "rosetta_tmp",
    analyze: bool = False,
) -> HeadlessResult:
    """Import (and optionally analyze) binary_path using the given Ghidra language_id."""
    headless = _find_headless(ghidra_home)

    with tempfile.TemporaryDirectory() as tmp:
        proj_dir = str(project_dir or tmp)
        cmd = [
            str(headless), proj_dir, project_name,
            "-import", str(binary_path),
            "-processor", language_id,
        ]
        if not analyze:
            cmd.append("-noanalysis")
        if post_script:
            cmd += ["-postScript", str(post_script)]

        env = os.environ.copy()
        java_home = env.get("JAVA_HOME", "")
        if java_home:
            java_bin = str(Path(java_home) / "bin")
            env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")

        log.info("Running headless: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    result = HeadlessResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if result.ok:
        log.info("Headless OK for %s with %s", binary_path, language_id)
    else:
        log.warning("Headless FAILED (rc=%d)", proc.returncode)
    return result
