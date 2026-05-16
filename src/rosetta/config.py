"""Rosetta settings: extends docquery Settings with Ghidra-specific config."""

from __future__ import annotations

import os
from pathlib import Path

from docquery.config import Settings as DocSettings


class Settings(DocSettings):
    """Docquery settings plus GHIDRA_HOME."""

    @property
    def ghidra_home(self) -> Path:
        raw = os.environ.get("GHIDRA_HOME", "")
        if not raw:
            raise RuntimeError(
                "GHIDRA_HOME is not set. Run scripts/setup_ghidra.sh or set it manually."
            )
        p = Path(raw)
        if not p.exists():
            raise RuntimeError(f"GHIDRA_HOME path does not exist: {p}")
        return p
