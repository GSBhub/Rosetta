#!/usr/bin/env python3
"""
Fix known LLM extraction errors in the ARMv6 ISA spec:
  - Endianness: "big" → "little" (ARMv6 is little-endian by default)
  - Register file: replace bad aliases with correct ARMv6 register layout
    (LLM conflated R12/R13/R14/PC)

Usage:
    python3 scripts/fix_armv6_spec.py dbs/ARMv6_isa_spec.json
"""

import json
import sys
from pathlib import Path

CORRECT_REGISTERS = [
    {"name": "R0",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R1",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R2",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R3",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R4",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R5",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R6",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R7",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R8",  "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R9",  "aliases": [],       "size_bits": 32, "description": "General-purpose (SB in AAPCS)"},
    {"name": "R10", "aliases": [],       "size_bits": 32, "description": "General-purpose"},
    {"name": "R11", "aliases": ["FP"],   "size_bits": 32, "description": "Frame pointer (FP)"},
    {"name": "R12", "aliases": ["IP"],   "size_bits": 32, "description": "Intra-procedure-call scratch (IP)"},
    {"name": "SP",  "aliases": ["R13"],  "size_bits": 32, "description": "Stack pointer"},
    {"name": "LR",  "aliases": ["R14"],  "size_bits": 32, "description": "Link register"},
    {"name": "PC",  "aliases": ["R15"],  "size_bits": 32, "description": "Program counter"},
]


def fix_spec(path: Path) -> None:
    data = json.loads(path.read_text())

    # Fix endianness
    if data["meta"]["endian"] == "big":
        print("  Fixing endianness: big → little")
        data["meta"]["endian"] = "little"

    # Fix alignment (ARMv6 ARM instructions are 4-byte aligned)
    if data["meta"].get("alignment") != 4:
        print(f"  Fixing alignment: {data['meta'].get('alignment')} → 4")
        data["meta"]["alignment"] = 4

    # Replace register file
    print(f"  Replacing register file ({len(data['registers'])} → {len(CORRECT_REGISTERS)} registers)")
    data["registers"] = CORRECT_REGISTERS

    path.write_text(json.dumps(data, indent=2))
    print(f"Saved fixed spec to {path}")


if __name__ == "__main__":
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dbs/ARMv6_isa_spec.json")
    if not spec_path.exists():
        print(f"Spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Fixing {spec_path}...")
    fix_spec(spec_path)
