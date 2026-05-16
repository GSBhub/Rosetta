"""Shared fixtures for the rosetta test suite."""

import os
from pathlib import Path

import pytest

from rosetta.extraction.schemas import ISAMeta, ISASpec, InstructionDef, RegisterDef

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def ghidra_home() -> Path | None:
    raw = os.environ.get("GHIDRA_HOME", "")
    if not raw:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GHIDRA_HOME="):
                    raw = line.partition("=")[2].strip()
    return Path(raw) if raw else None


GHIDRA_HOME = ghidra_home()

requires_ghidra = pytest.mark.skipif(
    GHIDRA_HOME is None or not GHIDRA_HOME.exists(),
    reason="GHIDRA_HOME not set or directory missing",
)

# ---------------------------------------------------------------------------
# Minimal ISASpec fixture (ARM-like, 3 registers, 3 instructions)
# ---------------------------------------------------------------------------

MINIMAL_META = ISAMeta(
    name="TestISA",
    endian="little",
    word_size_bits=32,
    alignment=4,
    instruction_sizes_bits=[32],
)

MINIMAL_REGISTERS = [
    RegisterDef(name="R0", aliases=[], size_bits=32, description="General purpose"),
    RegisterDef(name="R1", aliases=[], size_bits=32, description="General purpose"),
    RegisterDef(name="PC", aliases=["R15"], size_bits=32, description="Program counter"),
    RegisterDef(name="SP", aliases=["R13"], size_bits=32, description="Stack pointer"),
]

MINIMAL_INSTRUCTIONS = [
    InstructionDef(
        mnemonic="ADD",
        variants=["ADD R0, R1, R0"],
        encoding_bits=32,
        bit_fields={"opcode": "31:28", "rd": "15:12", "rn": "19:16", "rm": "3:0"},
        bit_constraints={"opcode": "0000"},
        operands=["R0", "R1"],
        semantics="Adds R1 to R0 and stores the result in R0.",
        pcode_hint="R0 = R1 + R0;",
    ),
    InstructionDef(
        mnemonic="MOV",
        variants=["MOV R0, R1"],
        encoding_bits=32,
        bit_fields={"opcode": "31:28"},
        bit_constraints={"opcode": "0001"},
        operands=["R0", "R1"],
        semantics="Copies R1 into R0.",
        pcode_hint="R0 = R1;",
    ),
    InstructionDef(
        mnemonic="B",
        variants=["B label"],
        encoding_bits=32,
        bit_fields={"opcode": "31:28", "offset": "23:0"},
        bit_constraints={"opcode": "1010"},
        operands=["label"],
        semantics="Branches to the target address.",
        pcode_hint="goto [PC + offset * 4 + 8];",
    ),
]

MINIMAL_SPEC = ISASpec(
    meta=MINIMAL_META,
    registers=MINIMAL_REGISTERS,
    instructions=MINIMAL_INSTRUCTIONS,
)


@pytest.fixture
def minimal_spec() -> ISASpec:
    return MINIMAL_SPEC


@pytest.fixture
def ghidra_arm_languages_dir() -> Path:
    if GHIDRA_HOME is None:
        pytest.skip("GHIDRA_HOME not set")
    return GHIDRA_HOME / "Ghidra" / "Processors" / "ARM" / "data" / "languages"
