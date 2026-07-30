#!/usr/bin/env python3
"""Run a full ingest -> generate cycle for one ISA from an examples/ config.

The per-ISA specifics live entirely in the TOML config (examples/*.toml); this
driver only plumbs those values into the generic `rosetta` CLI, so adding a new
ISA means adding a data file, not code.

Config keys:
    name                 processor name / SLEIGH file prefix   (required)
    manual               PDF path, or source dir when source=true (required)
    db                   ChromaDB directory                    (required)
    source               bool; ingest a source-code dir via --source (default false)
    instruction_pattern  regex whose group 1 is a mnemonic; enables grounded
                         discovery via --instruction-pattern    (optional)
    reference            Ghidra language ID for `rosetta evaluate` (optional)

Usage:
    uv run python scripts/run_isa.py examples/armv7m.toml
    uv run python scripts/run_isa.py examples/m7700.toml --skip-generate
    uv run python scripts/run_isa.py examples/tms320c67x.toml --out ./output
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


def _load(config_path: Path) -> dict:
    with config_path.open("rb") as fh:
        cfg = tomllib.load(fh)
    for key in ("name", "manual", "db"):
        if not cfg.get(key):
            sys.exit(f"{config_path}: missing required key '{key}'")
    return cfg


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="Path to an examples/*.toml ISA config")
    ap.add_argument("--out", default="./output", help="Output dir for generate (default ./output)")
    ap.add_argument("--skip-ingest", action="store_true", help="Reuse an existing --db")
    ap.add_argument("--skip-generate", action="store_true", help="Ingest only")
    args = ap.parse_args()

    cfg = _load(args.config)

    if not args.skip_ingest:
        ingest = ["rosetta", "ingest", cfg["manual"], "--db", cfg["db"]]
        if cfg.get("source"):
            ingest.append("--source")
        if cfg.get("instruction_pattern"):
            ingest += ["--instruction-pattern", cfg["instruction_pattern"]]
        _run(ingest)

    if not args.skip_generate:
        gen = ["rosetta", "generate", "--name", cfg["name"],
               "--db", cfg["db"], "--out", args.out]
        # The config's own [decode] table (if any) overlays structured-decode
        # data at render time; a no-op when absent.
        if "decode" in cfg:
            gen += ["--isa-config", str(args.config)]
        _run(gen)

    print(f"\nDone: {cfg['name']} -> {args.out}/{cfg['name']}/data/languages/")
    if cfg.get("reference"):
        print(f"Optional: rosetta evaluate {args.out}/{cfg['name']} "
              f"--reference {cfg['reference']}")


if __name__ == "__main__":
    main()
