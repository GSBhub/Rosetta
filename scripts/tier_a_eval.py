#!/usr/bin/env python3
"""Tier A: score Rosetta's Ghidra decode against binutils tic6x ground truth.

The binutils gas testsuite ships `.d` golden files: each line carries the raw
32-bit instruction word plus objdump's canonical disassembly, e.g.

    [0-9a-f]+[048c] <[^>]*> 0c180b20[ \\t]+absdp \\.S1 a7:a6,a25:a24

We parse (word, expected mnemonic), pack the words little-endian into a flat
binary, disassemble it once in headless Ghidra with our generated processor,
and compare mnemonic-by-mnemonic. Reports overall mnemonic accuracy plus which
mnemonics we get right vs. miss — a concrete score to drive Tier 1.5/2.

Usage:
    uv run python scripts/tier_a_eval.py <ground_truth.d> [--language TMS320C67x:LE:32:C67x]

Needs GHIDRA_HOME/JAVA_HOME (from .env) and the processor installed
(`rosetta install ./output/TMS320C67x`).
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

_LINE = re.compile(r"> ([0-9a-f]{8})\[ \\t\]\+(.+)")


def _load_env(key: str) -> str | None:
    if os.environ.get(key):
        return os.environ[key]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def parse_d(path: Path) -> list[tuple[int, str, str]]:
    """Return [(word, expected_mnemonic, expected_disasm)] from a .d golden file."""
    out: list[tuple[int, str, str]] = []
    for line in path.read_text().splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        word = int(m.group(1), 16)
        disasm = re.sub(r"\\(.)", r"\1", m.group(2)).strip()  # unescape regex
        # strip a leading predication prefix like "[a1] " / "[!b2] "
        body = re.sub(r"^\[[^\]]*\]\s*", "", disasm)
        toks = body.split()
        mnem = toks[0].lower() if toks else ""
        out.append((word, mnem, disasm))
    return out


_DISASM_SCRIPT = """\
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.Address;
public class TierADisasm extends GhidraScript {
    @Override public void run() throws Exception {
        Address a = currentProgram.getImageBase();
        long n = getScriptArgs().length > 0 ? Long.parseLong(getScriptArgs()[0]) : 4096;
        for (long i=0;i<n;i++){ try { disassemble(a); } catch (Exception e){} a = a.add(4); }
        Listing listing = currentProgram.getListing();
        InstructionIterator ii = listing.getInstructions(true);
        while (ii.hasNext()){ Instruction in=ii.next();
            println("TIERA " + in.getAddressString(false,true) + " " + in.toString()); }
    }
}
"""


def run_ghidra(binary: Path, language: str, count: int, ghidra: str, java: str) -> dict[int, str]:
    """Disassemble *binary* in headless Ghidra; return {word_index: our_mnemonic}."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        (tmp_p / "TierADisasm.java").write_text(_DISASM_SCRIPT)
        proj = tmp_p / "proj"
        proj.mkdir()
        env = dict(os.environ, JAVA_HOME=java, PATH=f"{java}/bin:" + os.environ.get("PATH", ""))
        cmd = [
            f"{ghidra}/support/analyzeHeadless", str(proj), "tiera",
            "-import", str(binary), "-processor", language,
            "-scriptPath", str(tmp_p), "-postScript", "TierADisasm.java", str(count),
            "-noanalysis", "-deleteProject",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    decoded: dict[int, str] = {}
    for line in res.stdout.splitlines():
        m = re.search(r"TIERA ([0-9a-f]+) (\S+)", line)
        if m:
            idx = int(m.group(1), 16) // 4
            decoded[idx] = m.group(2).lower()
    return decoded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dfile", type=Path, help="binutils tic6x .d golden file")
    ap.add_argument("--language", default="TMS320C67x:LE:32:C67x")
    ap.add_argument("--limit", type=int, default=0, help="cap instructions (0 = all)")
    args = ap.parse_args()

    ghidra = _load_env("GHIDRA_HOME")
    java = _load_env("JAVA_HOME")
    if not ghidra or not java:
        sys.exit("GHIDRA_HOME/JAVA_HOME not set (check .env)")

    truth = parse_d(args.dfile)
    if args.limit:
        truth = truth[: args.limit]
    if not truth:
        sys.exit(f"No instruction lines parsed from {args.dfile}")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for word, _, _ in truth:
            f.write(struct.pack("<I", word))
        binary = Path(f.name)

    print(f"Ground truth : {args.dfile}  ({len(truth)} instructions)")
    print(f"Language     : {args.language}")
    print("Disassembling in headless Ghidra ...")
    try:
        decoded = run_ghidra(binary, args.language, len(truth), ghidra, java)
    finally:
        # delete=False above, so clean up even when Ghidra raises/times out.
        binary.unlink(missing_ok=True)

    correct = miss_decode = wrong = 0
    hit_mnems: Counter[str] = Counter()
    missed_mnems: Counter[str] = Counter()
    over_mnems: Counter[str] = Counter()   # what we wrongly decode words AS
    for i, (_word, exp_mnem, _full) in enumerate(truth):
        got = decoded.get(i)
        if got is None:
            miss_decode += 1
            missed_mnems[exp_mnem] += 1
        elif got == exp_mnem:
            correct += 1
            hit_mnems[exp_mnem] += 1
        else:
            wrong += 1
            missed_mnems[exp_mnem] += 1
            over_mnems[got] += 1

    total = len(truth)
    print("\n===== Tier A mnemonic accuracy =====")
    print(f"  correct        : {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"  not decoded    : {miss_decode}")
    print(f"  wrong mnemonic : {wrong}")
    print(f"\n  distinct mnemonics correct ({len(hit_mnems)}): "
          + " ".join(sorted(hit_mnems)))
    print(f"\n  top missed mnemonics:")
    for mnem, cnt in missed_mnems.most_common(15):
        print(f"    {mnem:10s} x{cnt}")
    print(f"\n  top over-matchers (wrongly decoded AS):")
    for mnem, cnt in over_mnems.most_common(15):
        print(f"    {mnem:10s} x{cnt}")


if __name__ == "__main__":
    main()
