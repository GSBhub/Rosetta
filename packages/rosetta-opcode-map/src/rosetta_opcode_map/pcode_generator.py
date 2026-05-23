"""SLEIGH pcode body generation for CISC opcode-table entries.

Strategy:
  1. Look up (mnemonic, mode) in _PATTERNS — a comprehensive dict for 65xx/65816/M7700
     mnemonic families.  Pattern strings use {ws} (word-size bytes) substitution.
  2. If not found, fall back to a docquery RAG call asking the manual for semantics.
  3. If the fallback also fails, return a minimal stub.

SLEIGH size-safety rules applied throughout:
  - `zext(expr)` is NEVER used directly as a memory address — always assigned
    to an explicit `local addr:{ws}` first so SLEIGH can infer the output size.
  - `zext(ptr)` on a ptr that is already {ws} bytes is replaced by plain `ptr`.
  - Indexed addressing drops `zext` on same-size registers (X, Y already {ws} bytes
    for a 16-bit ISA) and routes through a `local ea:{ws}` local instead.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers used inside pattern strings
# ---------------------------------------------------------------------------
# All patterns use {ws} (word-size bytes) which is substituted at lookup time.
#
# Pcode operand names:
#   b1  — 8-bit byte operand token field
#   w1  — 16-bit word operand token field
#   l1  — 24-bit long operand token field
#   Rel8 / Rel16 — relative-branch subtable constructors
#
# Address size is always {ws} bytes.  Every indirect address goes through
# `local ea:{ws}` or `local ptr:{ws}` so SLEIGH can resolve sizes.
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, dict[str, str]] = {

    # ── Load Accumulator ───────────────────────────────────────────────────
    "LDA": {
        "imm":     "A = zext(b1);",
        "#imm":    "A = zext(b1);",
        "imm8":    "A = zext(b1);",
        "#imm8":   "A = zext(b1);",
        "imm16":   "A = w1;",
        "#imm16":  "A = w1;",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = *[ram]:{ws} ea;",
        "zp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = *[ram]:{ws} ea;",
        "dp,X":    "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; A = *[ram]:{ws} ea;",
        "zp,X":    "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; A = *[ram]:{ws} ea;",
        "dp,Y":    "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + Y; A = *[ram]:{ws} ea;",
        "zp,Y":    "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + Y; A = *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = *[ram]:{ws} ptr;",
        "(zp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = *[ram]:{ws} ptr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = *[ram]:{ws} ptr;",
        "(zp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = *[ram]:{ws} addr;",
        "(zp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = *[ram]:{ws} addr;",
        "abs":          "local w1_a:{ws} = w1; A = *[ram]:{ws} w1_a;",
        "absolute":     "local w1_a:{ws} = w1; A = *[ram]:{ws} w1_a;",
        "abs,X":        "local ea:{ws} = w1 + X; A = *[ram]:{ws} ea;",
        "absolute,X":   "local ea:{ws} = w1 + X; A = *[ram]:{ws} ea;",
        "absolute+X":   "local ea:{ws} = w1 + X; A = *[ram]:{ws} ea;",
        "abs,Y":        "local ea:{ws} = w1 + Y; A = *[ram]:{ws} ea;",
        "absolute,Y":   "local ea:{ws} = w1 + Y; A = *[ram]:{ws} ea;",
        "(abs)":        "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; A = *[ram]:{ws} ptr;",
        "(abs,X)":      "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = *[ram]:{ws} ptr;",
        "(abs),Y":      "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; local addr:{ws} = ptr + Y; A = *[ram]:{ws} addr;",
        "long":         "local l1_a:{ws} = l1; A = *[ram]:{ws} l1_a;",
        "abs24":        "local l1_a:{ws} = l1; A = *[ram]:{ws} l1_a;",
        "long,X":       "local ea:{ws} = l1 + X; A = *[ram]:{ws} ea;",
        "abs24,X":      "local ea:{ws} = l1 + X; A = *[ram]:{ws} ea;",
        "long,Y":       "local ea:{ws} = l1 + Y; A = *[ram]:{ws} ea;",
        "(long)":       "local l1_a:{ws} = l1; local ptr:{ws} = *[ram]:{ws} l1_a; A = *[ram]:{ws} ptr;",
        "(long,X)":     "local ea:{ws} = l1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = *[ram]:{ws} ptr;",
        "(long),Y":     "local l1_a:{ws} = l1; local ptr:{ws} = *[ram]:{ws} l1_a; local addr:{ws} = ptr + Y; A = *[ram]:{ws} addr;",
        "sr":           "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; A = *[ram]:{ws} ea;",
        "(sr),Y":       "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = *[ram]:{ws} addr;",
        "rel":          "goto Rel8;",
        "rel16":        "goto Rel16;",
    },

    # ── Store Accumulator ──────────────────────────────────────────────────
    "STA": {
        "dp":       "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; *[ram]:{ws} ea = A;",
        "zp":       "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; *[ram]:{ws} ea = A;",
        "dp,X":     "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; *[ram]:{ws} ea = A;",
        "zp,X":     "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; *[ram]:{ws} ea = A;",
        "dp,Y":     "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + Y; *[ram]:{ws} ea = A;",
        "zp,Y":     "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + Y; *[ram]:{ws} ea = A;",
        "(dp)":     "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ptr = A;",
        "(dp,X)":   "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ptr = A;",
        "(dp),Y":   "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; *[ram]:{ws} addr = A;",
        "(zp),Y":   "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; *[ram]:{ws} addr = A;",
        "abs":          "local w1_a:{ws} = w1; *[ram]:{ws} w1_a = A;",
        "absolute":     "local w1_a:{ws} = w1; *[ram]:{ws} w1_a = A;",
        "abs,X":        "local ea:{ws} = w1 + X; *[ram]:{ws} ea = A;",
        "absolute,X":   "local ea:{ws} = w1 + X; *[ram]:{ws} ea = A;",
        "abs,Y":        "local ea:{ws} = w1 + Y; *[ram]:{ws} ea = A;",
        "absolute,Y":   "local ea:{ws} = w1 + Y; *[ram]:{ws} ea = A;",
        "(abs)":        "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} ptr = A;",
        "(abs,X)":      "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ptr = A;",
        "(abs),Y":      "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; local addr:{ws} = ptr + Y; *[ram]:{ws} addr = A;",
        "long":         "local l1_a:{ws} = l1; *[ram]:{ws} l1_a = A;",
        "long,X":       "local ea:{ws} = l1 + X; *[ram]:{ws} ea = A;",
        "long,Y":       "local ea:{ws} = l1 + Y; *[ram]:{ws} ea = A;",
        "(long),Y":     "local l1_a:{ws} = l1; local ptr:{ws} = *[ram]:{ws} l1_a; local addr:{ws} = ptr + Y; *[ram]:{ws} addr = A;",
        "sr":           "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; *[ram]:{ws} ea = A;",
        "(sr),Y":       "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; *[ram]:{ws} addr = A;",
        "rel16":        "local ea:{ws} = Rel16; *[ram]:{ws} ea = A;",
    },

    # ── Arithmetic ─────────────────────────────────────────────────────────
    "ADC": {
        "imm":     "local b1_z:{ws} = b1; A = A + b1_z;",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = A + *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = A + *[ram]:{ws} ptr;",
        "abs":     "local w1_a:{ws} = w1; A = A + *[ram]:{ws} w1_a;",
        "abs,X":   "local ea:{ws} = w1 + X; A = A + *[ram]:{ws} ea;",
        "abs,Y":   "local ea:{ws} = w1 + Y; A = A + *[ram]:{ws} ea;",
        "(abs,X)": "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = A + *[ram]:{ws} ptr;",
        "(abs),Y": "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; local addr:{ws} = ptr + Y; A = A + *[ram]:{ws} addr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = A + *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = A + *[ram]:{ws} addr;",
        "sr":      "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; A = A + *[ram]:{ws} ea;",
    },
    "SBC": {
        "imm":     "local b1_z:{ws} = b1; A = A - b1_z;",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = A - *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = A - *[ram]:{ws} ptr;",
        "abs":     "local w1_a:{ws} = w1; A = A - *[ram]:{ws} w1_a;",
        "abs,X":   "local ea:{ws} = w1 + X; A = A - *[ram]:{ws} ea;",
        "abs,Y":   "local ea:{ws} = w1 + Y; A = A - *[ram]:{ws} ea;",
        "(abs,X)": "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = A - *[ram]:{ws} ptr;",
        "(abs),Y": "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; local addr:{ws} = ptr + Y; A = A - *[ram]:{ws} addr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = A - *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = A - *[ram]:{ws} addr;",
        "sr":      "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; A = A - *[ram]:{ws} ea;",
    },

    # ── Logical ────────────────────────────────────────────────────────────
    "AND": {
        "imm":     "A = A & zext(b1);",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = A & *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = A & *[ram]:{ws} ptr;",
        "abs":     "local w1_a:{ws} = w1; A = A & *[ram]:{ws} w1_a;",
        "abs,X":   "local ea:{ws} = w1 + X; A = A & *[ram]:{ws} ea;",
        "abs,Y":   "local ea:{ws} = w1 + Y; A = A & *[ram]:{ws} ea;",
        "(abs,X)": "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = A & *[ram]:{ws} ptr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = A & *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = A & *[ram]:{ws} addr;",
    },
    "ORA": {
        "imm":     "A = A | zext(b1);",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = A | *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = A | *[ram]:{ws} ptr;",
        "abs":     "local w1_a:{ws} = w1; A = A | *[ram]:{ws} w1_a;",
        "abs,X":   "local ea:{ws} = w1 + X; A = A | *[ram]:{ws} ea;",
        "abs,Y":   "local ea:{ws} = w1 + Y; A = A | *[ram]:{ws} ea;",
        "(abs,X)": "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = A | *[ram]:{ws} ptr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = A | *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = A | *[ram]:{ws} addr;",
    },
    "EOR": {
        "imm":     "A = A ^ zext(b1);",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; A = A ^ *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; A = A ^ *[ram]:{ws} ptr;",
        "abs":     "local w1_a:{ws} = w1; A = A ^ *[ram]:{ws} w1_a;",
        "abs,X":   "local ea:{ws} = w1 + X; A = A ^ *[ram]:{ws} ea;",
        "abs,Y":   "local ea:{ws} = w1 + Y; A = A ^ *[ram]:{ws} ea;",
        "(abs,X)": "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; A = A ^ *[ram]:{ws} ptr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; A = A ^ *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; A = A ^ *[ram]:{ws} addr;",
    },

    # ── Compare ────────────────────────────────────────────────────────────
    "CMP": {
        "imm":     "local b1_z:{ws} = b1; local cmp_r:{ws} = A - b1_z;",
        "dp":      "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local cmp_r:{ws} = A - *[ram]:{ws} ea;",
        "(dp)":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local cmp_r:{ws} = A - *[ram]:{ws} ptr;",
        "abs":     "local w1_a:{ws} = w1; local cmp_r:{ws} = A - *[ram]:{ws} w1_a;",
        "abs,X":   "local ea:{ws} = w1 + X; local cmp_r:{ws} = A - *[ram]:{ws} ea;",
        "abs,Y":   "local ea:{ws} = w1 + Y; local cmp_r:{ws} = A - *[ram]:{ws} ea;",
        "(abs,X)": "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; local cmp_r:{ws} = A - *[ram]:{ws} ptr;",
        "(dp,X)":  "local b1_z:{ws} = b1; local base:{ws} = DPR + b1_z; local ea:{ws} = base + X; local ptr:{ws} = *[ram]:{ws} ea; local cmp_r:{ws} = A - *[ram]:{ws} ptr;",
        "(dp),Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local ptr:{ws} = *[ram]:{ws} ea; local addr:{ws} = ptr + Y; local cmp_r:{ws} = A - *[ram]:{ws} addr;",
        "sr":      "local b1_z:{ws} = b1; local base:{ws} = zext(SP); local ea:{ws} = base + b1_z; local cmp_r:{ws} = A - *[ram]:{ws} ea;",
    },
    "CPX": {
        "imm":  "local b1_z:{ws} = b1; local cmp_r:{ws} = X - b1_z;",
        "dp":   "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local cmp_r:{ws} = X - *[ram]:{ws} ea;",
        "abs":  "local w1_a:{ws} = w1; local cmp_r:{ws} = X - *[ram]:{ws} w1_a;",
    },
    "CPY": {
        "imm":  "local b1_z:{ws} = b1; local cmp_r:{ws} = Y - b1_z;",
        "dp":   "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local cmp_r:{ws} = Y - *[ram]:{ws} ea;",
        "abs":  "local w1_a:{ws} = w1; local cmp_r:{ws} = Y - *[ram]:{ws} w1_a;",
    },

    # ── Stack — SP is often 1 byte in extracted specs, use explicit addr local ──
    "PHA": {
        "imp": "SP = SP - {ws}; local sp_a:{ws} = zext(SP); *[ram]:{ws} sp_a = A;",
        "acc": "SP = SP - {ws}; local sp_a:{ws} = zext(SP); *[ram]:{ws} sp_a = A;",
    },
    "PLA": {
        "imp": "local sp_a:{ws} = zext(SP); A = *[ram]:{ws} sp_a; SP = SP + {ws};",
        "acc": "local sp_a:{ws} = zext(SP); A = *[ram]:{ws} sp_a; SP = SP + {ws};",
    },
    "PHP": {
        "imp": "SP = SP - 1; local sp_a:{ws} = zext(SP); *[ram]:1 sp_a = 0;",
        "acc": "SP = SP - 1; local sp_a:{ws} = zext(SP); *[ram]:1 sp_a = 0;",
    },
    "PLP": {
        "imp": "local sp_a:{ws} = zext(SP); local flags:1 = *[ram]:1 sp_a; SP = SP + 1;",
        "acc": "local sp_a:{ws} = zext(SP); local flags:1 = *[ram]:1 sp_a; SP = SP + 1;",
    },
    "PHX": {"imp": "SP = SP - {ws}; local sp_a:{ws} = zext(SP); *[ram]:{ws} sp_a = X;"},
    "PLX": {"imp": "local sp_a:{ws} = zext(SP); X = *[ram]:{ws} sp_a; SP = SP + {ws};"},
    "PHY": {"imp": "SP = SP - {ws}; local sp_a:{ws} = zext(SP); *[ram]:{ws} sp_a = Y;"},
    "PLY": {"imp": "local sp_a:{ws} = zext(SP); Y = *[ram]:{ws} sp_a; SP = SP + {ws};"},
    "PHB": {"imp": "SP = SP - 1; local sp_a:{ws} = zext(SP); *[ram]:1 sp_a = DT;"},
    "PLB": {"imp": "local sp_a:{ws} = zext(SP); DT = *[ram]:1 sp_a; SP = SP + 1;"},
    "PHD": {"imp": "SP = SP - 2; local sp_a:{ws} = zext(SP); *[ram]:2 sp_a = DPR;"},
    "PLD": {"imp": "local sp_a:{ws} = zext(SP); DPR = *[ram]:2 sp_a; SP = SP + 2;"},
    "PHK": {"imp": "SP = SP - 1; local sp_a:{ws} = zext(SP); *[ram]:1 sp_a = PG;"},

    # ── Load index registers ───────────────────────────────────────────────
    "LDX": {
        "imm":    "X = zext(b1);",
        "imm16":  "X = w1;",
        "#imm":   "X = zext(b1);",
        "dp":     "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; X = *[ram]:{ws} ea;",
        "dp,Y":   "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + Y; X = *[ram]:{ws} ea;",
        "abs":    "local w1_a:{ws} = w1; X = *[ram]:{ws} w1_a;",
        "abs,Y":  "local ea:{ws} = w1 + Y; X = *[ram]:{ws} ea;",
    },
    "LDY": {
        "imm":    "Y = zext(b1);",
        "imm16":  "Y = w1;",
        "#imm":   "Y = zext(b1);",
        "dp":     "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; Y = *[ram]:{ws} ea;",
        "dp,X":   "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + X; Y = *[ram]:{ws} ea;",
        "abs":    "local w1_a:{ws} = w1; Y = *[ram]:{ws} w1_a;",
        "abs,X":  "local ea:{ws} = w1 + X; Y = *[ram]:{ws} ea;",
    },
    "STX": {
        "dp":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; *[ram]:{ws} ea = X;",
        "dp,Y":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + Y; *[ram]:{ws} ea = X;",
        "abs":   "local w1_a:{ws} = w1; *[ram]:{ws} w1_a = X;",
    },
    "STY": {
        "dp":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; *[ram]:{ws} ea = Y;",
        "dp,X":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + X; *[ram]:{ws} ea = Y;",
        "abs":   "local w1_a:{ws} = w1; *[ram]:{ws} w1_a = Y;",
    },
    "STZ": {
        "dp":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; *[ram]:{ws} ea = 0;",
        "dp,X":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + X; *[ram]:{ws} ea = 0;",
        "abs":   "local w1_a:{ws} = w1; *[ram]:{ws} w1_a = 0;",
        "abs,X": "local ea:{ws} = w1 + X; *[ram]:{ws} ea = 0;",
    },

    # ── Transfers ──────────────────────────────────────────────────────────
    "TAX": {"imp": "X = A;"},
    "TAY": {"imp": "Y = A;"},
    "TXA": {"imp": "A = X;"},
    "TYA": {"imp": "A = Y;"},
    "TXS": {"imp": "SP = X[0,8];"},
    "TSX": {"imp": "X = zext(SP);"},
    "TXY": {"imp": "Y = X;"},
    "TYX": {"imp": "X = Y;"},
    "TAD": {"imp": "DPR = A;"},
    "TDA": {"imp": "A = DPR;"},
    "TAS": {"imp": "SP = A[0,8];"},
    "TSA": {"imp": "A = zext(SP);"},
    "XBA": {"imp": "local hi:1 = A[8,8]; local lo:1 = A[0,8]; A[0,8] = hi; A[8,8] = lo;"},
    "XCE": {"imp": "local tmp:1 = 0;"},

    # ── INC / DEC ──────────────────────────────────────────────────────────
    "INC": {
        "imp":   "A = A + 1;",
        "acc":   "A = A + 1;",
        "dp":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m + 1;",
        "dp,X":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + X; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m + 1;",
        "abs":   "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = m + 1;",
        "abs,X": "local ea:{ws} = w1 + X; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m + 1;",
    },
    "DEC": {
        "imp":   "A = A - 1;",
        "acc":   "A = A - 1;",
        "dp":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m - 1;",
        "dp,X":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + X; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m - 1;",
        "abs":   "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = m - 1;",
        "abs,X": "local ea:{ws} = w1 + X; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m - 1;",
    },
    "INX": {"imp": "X = X + 1;"},
    "DEX": {"imp": "X = X - 1;"},
    "INY": {"imp": "Y = Y + 1;"},
    "DEY": {"imp": "Y = Y - 1;"},

    # ── Shifts / Rotates ───────────────────────────────────────────────────
    "ASL": {
        "imp": "A = A << 1;",
        "acc": "A = A << 1;",
        "dp":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m << 1;",
        "abs": "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = m << 1;",
    },
    "LSR": {
        "imp": "A = A >> 1;",
        "acc": "A = A >> 1;",
        "dp":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m >> 1;",
        "abs": "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = m >> 1;",
    },
    "ROL": {
        "imp": "A = (A << 1) | (A >> 15);",
        "acc": "A = (A << 1) | (A >> 15);",
        "dp":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = (m << 1) | (m >> 15);",
        "abs": "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = (m << 1) | (m >> 15);",
    },
    "ROR": {
        "imp": "A = (A >> 1) | (A << 15);",
        "acc": "A = (A >> 1) | (A << 15);",
        "dp":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = (m >> 1) | (m << 15);",
        "abs": "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = (m >> 1) | (m << 15);",
    },

    # ── Bit ops ────────────────────────────────────────────────────────────
    "BIT": {
        "imm":   "local cmp_r:{ws} = A & zext(b1);",
        "dp":    "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local cmp_r:{ws} = A & *[ram]:{ws} ea;",
        "dp,X":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z + X; local cmp_r:{ws} = A & *[ram]:{ws} ea;",
        "abs":   "local w1_a:{ws} = w1; local cmp_r:{ws} = A & *[ram]:{ws} w1_a;",
        "abs,X": "local ea:{ws} = w1 + X; local cmp_r:{ws} = A & *[ram]:{ws} ea;",
    },
    "TRB": {
        "dp":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m & ~A;",
        "abs": "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = m & ~A;",
    },
    "TSB": {
        "dp":  "local b1_z:{ws} = b1; local ea:{ws} = DPR + b1_z; local m:{ws} = *[ram]:{ws} ea; *[ram]:{ws} ea = m | A;",
        "abs": "local w1_a:{ws} = w1; local m:{ws} = *[ram]:{ws} w1_a; *[ram]:{ws} w1_a = m | A;",
    },

    # ── Branches ───────────────────────────────────────────────────────────
    "BCC": {"rel": "if (C == 0) goto Rel8;", "rel8": "if (C == 0) goto Rel8;"},
    "BCS": {"rel": "if (C == 1) goto Rel8;", "rel8": "if (C == 1) goto Rel8;"},
    "BEQ": {"rel": "if (Z == 1) goto Rel8;", "rel8": "if (Z == 1) goto Rel8;"},
    "BNE": {"rel": "if (Z == 0) goto Rel8;", "rel8": "if (Z == 0) goto Rel8;"},
    "BMI": {"rel": "if (N == 1) goto Rel8;", "rel8": "if (N == 1) goto Rel8;"},
    "BPL": {"rel": "if (N == 0) goto Rel8;", "rel8": "if (N == 0) goto Rel8;"},
    "BVS": {"rel": "if (V == 1) goto Rel8;", "rel8": "if (V == 1) goto Rel8;"},
    "BVC": {"rel": "if (V == 0) goto Rel8;", "rel8": "if (V == 0) goto Rel8;"},
    "BRA": {"rel": "goto Rel8;", "rel8": "goto Rel8;"},
    "BRL": {"rel16": "goto Rel16;"},

    # ── Jump / Call / Return ───────────────────────────────────────────────
    "JMP": {
        "abs":      "goto w1;",
        "absolute": "goto w1;",
        "(abs)":     "local w1_a:{ws} = w1; local ptr:{ws} = *[ram]:{ws} w1_a; goto [ptr];",
        "(abs,X)":  "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; goto [ptr];",
        "long":     "goto l1;",
        "abs24":    "goto l1;",
    },
    "JML": {"long": "goto l1;", "abs24": "goto l1;"},
    "JSR": {
        "abs":      "call w1;",
        "absolute": "call w1;",
        "(abs,X)":  "local ea:{ws} = w1 + X; local ptr:{ws} = *[ram]:{ws} ea; call [ptr];",
    },
    "JSL": {"long": "call l1;", "abs24": "call l1;"},
    "RTS": {"imp": "return [0];"},
    "RTL": {"imp": "return [0];"},
    "RTI": {"imp": "return [0];"},

    # ── Processor status ───────────────────────────────────────────────────
    "SEC": {"imp": "C = 1;"},
    "CLC": {"imp": "C = 0;"},
    "SEI": {"imp": "I = 1;"},
    "CLI": {"imp": "I = 0;"},
    "SED": {"imp": "D = 1;"},
    "CLD": {"imp": "D = 0;"},
    "CLV": {"imp": "V = 0;"},
    "SEP": {"imm": "local tmp:1 = b1;"},
    "REP": {"imm": "local tmp:1 = b1;"},

    # ── Misc ───────────────────────────────────────────────────────────────
    "NOP": {"imp": "local tmp:1 = 0;"},
    "STP": {"imp": "local tmp:1 = 0;"},
    "WAI": {"imp": "local tmp:1 = 0;"},
    "WDM": {"imm": "local tmp:1 = b1;"},
    "BRK": {"imm": "local tmp:1 = 0;"},
    "COP": {"imm": "local tmp:1 = 0;"},
    "MVN": {"mvn": "local tmp:1 = 0;"},
    "MVP": {"mvp": "local tmp:1 = 0;"},

    # ── M7700 extended (0x89 prefix: multiply / divide) ───────────────────
    "MPY": {"imp": "A = A * B;"},
    "DIV": {"imp": "A = A / B;"},
}

_STUB = "local tmp:1 = 0;"


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _normalize_mnemonic(mnemonic: str) -> str:
    """Strip to first whitespace-delimited token, uppercase — matches OpcodeDef._strip_mnemonic."""
    word = mnemonic.strip().split()[0] if mnemonic.strip() else "UNK"
    return word.upper()


# Tokens available in the is-clause for each addressing mode.
# LLM-generated pcode that references tokens outside this set is invalid and
# will cause SLEIGH "not declared in pattern list" errors.
_MODE_TOKENS: dict[str, frozenset[str]] = {
    # implied / accumulator — no operand tokens
    "":            frozenset(),
    "imp":         frozenset(), "implied":      frozenset(),
    "acc":         frozenset(), "accumulator":  frozenset(),
    # byte operand → b1
    **{m: frozenset({"b1"}) for m in (
        "imm", "#imm", "imm8", "#imm8",
        "dp", "zp", "direct", "direct page",
        "dp,X", "zp,X", "direct,X",
        "dp,Y", "zp,Y", "direct,Y",
        "(dp)", "(zp)", "indirect",
        "(dp,X)", "(zp,X)", "dp indexed indirect", "indexed indirect",
        "(dp),Y", "(zp),Y", "indirect indexed",
        "sr", "stk", "stack relative", "stack,S",
        "(sr),Y", "(stk),Y", "stack relative indirect indexed",
        "sig", "signature", "mvn", "mvp", "block move",
    )},
    # word (16-bit) operand → w1
    **{m: frozenset({"w1"}) for m in (
        "imm16", "#imm16",
        "abs", "absolute", "abs16",
        "abs,X", "absolute,X", "absolute+X",
        "abs,Y", "absolute,Y", "absolute+Y",
        "(abs)", "(absolute)", "absolute indirect",
        "(abs,X)", "(absolute,X)", "absolute indirect indexed",
        "(abs),Y", "(absolute),Y", "absolute indirect indexed Y",
    )},
    # long (24-bit) operand → l1
    **{m: frozenset({"l1"}) for m in (
        "long", "abs24", "absolute long", "long absolute",
        "long,X", "abs24,X", "absolute long,X",
        "long,Y", "abs24,Y", "absolute long,Y",
        "(long,X)", "(abs24,X)",
        "(long),Y", "(abs24),Y", "absolute long indirect indexed Y",
    )},
    # relative branches — subtable names, NOT b1/w1
    **{m: frozenset({"Rel8"}) for m in (
        "rel", "rel8", "relative", "pcr8",
    )},
    **{m: frozenset({"Rel16"}) for m in (
        "rel16", "pcr16", "long relative",
    )},
}

_ALL_TOKENS = frozenset({"b1", "w1", "l1", "sb1", "sw1", "Rel8", "Rel16"})


def _validate_pcode_tokens(body: str, mode: str) -> bool:
    """Return True if body only references token fields declared for this mode."""
    import re
    allowed = _MODE_TOKENS.get(mode, frozenset({"b1"}))
    for tok in _ALL_TOKENS - allowed:
        if re.search(r"\b" + re.escape(tok) + r"\b", body):
            return False
    return True


def lookup_pattern(mnemonic: str, mode: str, word_size_bits: int) -> str | None:
    """Return a pcode body string for (mnemonic, mode), or None if not in table."""
    ws = word_size_bits // 8
    base = _normalize_mnemonic(mnemonic)
    entry = _PATTERNS.get(base)
    if entry is None:
        return None
    body = entry.get(mode) or entry.get(mode.lower())
    if body is None:
        return None
    return body.format(ws=ws)


# ---------------------------------------------------------------------------
# RAG + LLM fallback
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM = """\
You are an expert in Ghidra SLEIGH P-code for CISC processors.
Given an instruction mnemonic and addressing mode, output a single-line SLEIGH pcode body.

Token fields available per addressing mode (ONLY use the token for the current mode):
  b1  (8-bit)  — dp, zp, dp,X, dp,Y, (dp), (dp,X), (dp),Y, imm, sr, (sr),Y
  w1  (16-bit) — abs, abs,X, abs,Y, (abs), (abs,X), (abs),Y, imm16
  l1  (24-bit) — long, long,X, long,Y, (long,X), (long),Y
  Rel8         — rel, rel8, relative  (do NOT use b1 for branch modes)
  Rel16        — rel16, pcr16
  (none)       — imp, acc

SLEIGH size-safety rules (violating these causes compile errors):
1. NEVER call zext() on a token field (b1, w1, l1). Token fields extend implicitly.
   WRONG: local ea:2 = zext(b1);    RIGHT: local ea:2 = b1;
2. NEVER use a token field directly as a memory address.
   WRONG: A = *[ram]:2 w1;          RIGHT: local w1_a:2 = w1; A = *[ram]:2 w1_a;
3. NEVER do three-operand arithmetic in one step.
   WRONG: local ea:2 = DPR + b1 + X;
   RIGHT: local b1_z:2 = b1; local base:2 = DPR + b1_z; local ea:2 = base + X;

Correct pcode examples:
    LDA abs     → local w1_a:2 = w1; A = *[ram]:2 w1_a;
    STA dp,X    → local b1_z:2 = b1; local base:2 = DPR + b1_z; local ea:2 = base + X; *[ram]:2 ea = A;
    JMP (abs)   → local w1_a:2 = w1; local ptr:2 = *[ram]:2 w1_a; goto [ptr];
    BEQ rel     → if (Z == 1) goto Rel8;
    BRA rel     → goto Rel8;
    PHA         → SP = SP - 2; local sp_a:2 = zext(SP); *[ram]:2 sp_a = A;
    RTS         → return [0];
    Unknown     → local tmp:1 = 0;

Output ONLY the pcode body — no explanation, no surrounding braces.
"""


class _PcodeHint(BaseModel):
    pcode_body: str = Field(
        description="SLEIGH pcode body (one line, no surrounding braces). "
                    "Use b1/w1/l1 for operand bytes, Rel8/Rel16 for branches."
    )


def generate_via_llm(
    mnemonic: str,
    mode: str,
    description: str,
    isa_name: str,
    register_names: list[str],
    word_size_bits: int,
    settings: Any,
) -> str:
    """Ask the RAG-augmented LLM to produce a pcode body for one (mnemonic, mode) pair."""
    try:
        import docquery

        ws = word_size_bits // 8
        reg_hint = ", ".join(register_names[:10]) if register_names else "(unknown)"
        query = (
            f"In the {isa_name} processor ({word_size_bits}-bit, registers: {reg_hint}), "
            f"what SLEIGH pcode body implements the {mnemonic} instruction "
            f"in {mode or 'implied'} addressing mode? Word size = {ws} bytes. "
            f"Use b1 (byte operand), w1 (word/address operand). "
            f"Description: {description or '(none)'}"
        )
        result = docquery.query(
            query,
            schema=_PcodeHint,
            system_prompt=_FALLBACK_SYSTEM,
            settings=settings,
        )
        if isinstance(result, _PcodeHint) and result.pcode_body.strip():
            return result.pcode_body.strip().rstrip(";") + ";"
    except Exception as exc:
        log.warning("pcode RAG fallback failed for %s %s: %s", mnemonic, mode, exc)
    return _STUB


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_pcode_bodies(
    entries: list[dict[str, Any]],
    isa_name: str,
    register_names: list[str],
    word_size_bits: int,
    settings: Any,
    inter_entry_sleep: float = 0.5,
) -> dict[tuple[str, str], str]:
    """Return {(mnemonic, mode): pcode_body} for all non-UNK entries.

    Pattern-dict lookups are free; only genuinely unknown pairs trigger an LLM call.
    """
    import time

    pairs: dict[tuple[str, str], str] = {}
    for e in entries:
        raw_mn = e.get("mnemonic", "UNK")
        mn = _normalize_mnemonic(raw_mn)
        if mn == "UNK":
            continue
        key = (mn, e.get("mode", "imp"))
        if key not in pairs:
            pairs[key] = e.get("description", "")

    results: dict[tuple[str, str], str] = {}
    llm_calls = 0

    for (mn, mode), desc in sorted(pairs.items()):
        body = lookup_pattern(mn, mode, word_size_bits)
        if body is not None:
            log.debug("pcode: pattern hit %s %s", mn, mode)
            results[(mn, mode)] = body
        else:
            log.info("pcode: LLM fallback for %s %s", mn, mode)
            if llm_calls > 0 and inter_entry_sleep > 0:
                time.sleep(inter_entry_sleep)
            llm_body = generate_via_llm(
                mn, mode, desc, isa_name, register_names, word_size_bits, settings
            )
            llm_calls += 1
            if not _validate_pcode_tokens(llm_body, mode):
                log.warning(
                    "pcode: LLM body for (%s, %s) references undeclared tokens — discarding: %r",
                    mn, mode, llm_body,
                )
                llm_body = _STUB
            results[(mn, mode)] = llm_body

    log.info(
        "pcode: %d pairs — %d pattern hits, %d LLM calls",
        len(pairs), len(pairs) - llm_calls, llm_calls,
    )
    return results
