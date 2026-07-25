#!/usr/bin/env python3
"""Static SLEIGH decode simulation — score a generated .slaspec vs tic6x ground
truth without running Ghidra.

Parses the generated .slaspec's token fields + constructor patterns, replays
SLEIGH fixed-bit matching (most-constrained constructor wins) against the raw
instruction words in a binutils `.d` golden file, and reports mnemonic accuracy
plus *why* wrong decodes happen — the fast inner-loop check before spending a
headless-Ghidra `tier_a_eval.py` run.

Usage:
    uv run python scripts/tier_a_sim.py <module.slaspec> <ground_truth.d>
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

_TOKEN_FIELD = re.compile(r"^\s*(\w+)\s*=\s*\((\d+),(\d+)\)", re.MULTILINE)
_CONSTR = re.compile(r"^:(\S+)([^\n]*?)\s+is\s+(.+)$", re.MULTILINE)
_EQ = re.compile(r"(\w+)=0b([01]+)")
_GT_LINE = re.compile(r"> ([0-9a-f]{8})\[ \\t\]\+(.+)")


def parse_slaspec(path: Path):
    """Return (fields{name:(lo,hi)}, constructors[(mnemonic, {name:(hi,lo,val)}, nbits)])."""
    text = path.read_text()
    fields = {m[1]: (int(m[2]), int(m[3])) for m in _TOKEN_FIELD.finditer(text)}
    cons = []
    for m in _CONSTR.finditer(text):
        mnem = m.group(1).split(".")[0].lower()   # strip fused .unit
        pattern = m.group(3)
        fl = {}
        for f, v in _EQ.findall(pattern):
            if f in fields and "stub" not in f:
                lo, hi = fields[f]
                fl[f] = (hi, lo, int(v, 2))
        if fl:
            nbits = sum(hi - lo + 1 for hi, lo, _ in fl.values())
            cons.append((mnem, fl, nbits))
    return fields, cons


def parse_ground_truth(path: Path):
    out = []
    for line in path.read_text().splitlines():
        m = _GT_LINE.search(line)
        if m:
            word = int(m.group(1), 16)
            d = re.sub(r"\\(.)", r"\1", m.group(2)).strip()
            d = re.sub(r"^\[[^\]]*\]\s*", "", d)
            out.append((word, d.split()[0].lower()))
    return out


def _match(word: int, fl: dict) -> bool:
    return all((word >> lo) & ((1 << (hi - lo + 1)) - 1) == val for hi, lo, val in fl.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slaspec", type=Path)
    ap.add_argument("dfile", type=Path)
    args = ap.parse_args()

    _fields, cons = parse_slaspec(args.slaspec)
    truth = parse_ground_truth(args.dfile)
    if not cons or not truth:
        sys.exit("no constructors or no ground-truth lines parsed")

    by_mnem: dict[str, list] = {}
    for mn, fl, nb in cons:
        by_mnem.setdefault(mn, []).append((fl, nb))

    correct = wrong = nomatch = 0
    over = Counter()      # what we wrongly decode words AS
    mech = Counter()      # wrong-decode mechanism
    for word, exp in truth:
        ms = [(mn, nb) for mn, fl, nb in cons if _match(word, fl)]
        if not ms:
            nomatch += 1
            continue
        best = max(nb for _, nb in ms)
        winners = {mn for mn, nb in ms if nb == best}
        if exp in winners:
            correct += 1
            continue
        wrong += 1
        for mn in winners:
            over[mn] += 1
        exp_m = [nb for fl, nb in by_mnem.get(exp, []) if _match(word, fl)]
        if not by_mnem.get(exp):
            mech["exp not recovered"] += 1
        elif not exp_m:
            mech["exp recovered but doesn't match word (over-broad winner / missing variant)"] += 1
        elif max(exp_m) < best:
            mech["exp matches, less specific than winner"] += 1
        else:
            mech["exp ties winner (tie-break)"] += 1

    total = len(truth)
    print(f"slaspec      : {args.slaspec}  ({len(cons)} real constructors)")
    print(f"ground truth : {args.dfile}  ({total} instructions)\n")
    print(f"  correct     : {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"  wrong       : {wrong}")
    print(f"  not decoded : {nomatch}")
    print("\n  wrong-decode mechanism:")
    for k, v in mech.most_common():
        print(f"    {v:4d}  {k}")
    print("\n  top over-matchers (wrongly decoded AS):")
    for mn, c in over.most_common(12):
        print(f"    {mn:10s} x{c}")


if __name__ == "__main__":
    main()
