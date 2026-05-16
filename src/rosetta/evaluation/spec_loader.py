"""Parse Ghidra .slaspec files to extract mnemonics and register names."""

from __future__ import annotations

import re
from pathlib import Path


def load_slaspec_text(slaspec_path: Path, _visited: set[Path] | None = None) -> str:
    """
    Read a .slaspec (and all recursively @include'd .sinc files) as a single text blob.
    Handles multi-level include chains like ARM7_le.slaspec → ARM.sinc → ARMinstructions.sinc.
    """
    if _visited is None:
        _visited = set()
    canonical = slaspec_path.resolve()
    if canonical in _visited:
        return ""
    _visited.add(canonical)

    base_dir = slaspec_path.parent
    text = slaspec_path.read_text(errors="replace")

    include_re = re.compile(r'@include\s+"([^"]+)"')
    extra = []
    for match in include_re.finditer(text):
        inc_path = base_dir / match.group(1)
        if inc_path.exists():
            extra.append(load_slaspec_text(inc_path, _visited))

    return text + "\n" + "\n".join(extra) if extra else text


def extract_mnemonics(slaspec_text: str) -> set[str]:
    """
    Extract instruction mnemonics from SLEIGH constructor lines.
    Constructor syntax: :<MNEMONIC> ... is ...
    """
    pattern = re.compile(r"^\s*:\s*([A-Za-z][A-Za-z0-9_\.]*)", re.MULTILINE)
    return {m.group(1).upper() for m in pattern.finditer(slaspec_text)}


def extract_register_names(slaspec_text: str) -> set[str]:
    """
    Extract register names from 'define register ... [ R0 R1 ... ]' blocks.
    """
    registers: set[str] = set()
    block_re = re.compile(r"define\s+register\b[^[]*\[([^\]]+)\]", re.DOTALL)
    for block_match in block_re.finditer(slaspec_text):
        names = re.findall(r"[A-Za-z][A-Za-z0-9_]*", block_match.group(1))
        registers.update(n.upper() for n in names)
    return registers


# Maps Ghidra language ID → (processor_dir, slaspec_filename) for Ghidra 12.x
# Naming convention changed in Ghidra 12: ARM variants are per-file (ARM7_le.slaspec, etc.)
_LANG_ID_TO_SLASPEC: dict[str, tuple[str, str]] = {
    # ARM 32-bit
    "ARM:LE:32:v4":               ("ARM", "ARM4_le.slaspec"),
    "ARM:BE:32:v4":               ("ARM", "ARM4_be.slaspec"),
    "ARM:LE:32:v4t":              ("ARM", "ARM4t_le.slaspec"),
    "ARM:BE:32:v4t":              ("ARM", "ARM4t_be.slaspec"),
    "ARM:LE:32:v5":               ("ARM", "ARM5_le.slaspec"),
    "ARM:BE:32:v5":               ("ARM", "ARM5_be.slaspec"),
    "ARM:LE:32:v5t":              ("ARM", "ARM5t_le.slaspec"),
    "ARM:BE:32:v5t":              ("ARM", "ARM5t_be.slaspec"),
    "ARM:LE:32:v6":               ("ARM", "ARM6_le.slaspec"),
    "ARM:BE:32:v6":               ("ARM", "ARM6_be.slaspec"),
    "ARM:LE:32:v7":               ("ARM", "ARM7_le.slaspec"),
    "ARM:BE:32:v7":               ("ARM", "ARM7_be.slaspec"),
    "ARM:LEBE:32:v7LEInstruction": ("ARM", "ARM7_le.slaspec"),
    "ARM:LE:32:v8":               ("ARM", "ARM8_le.slaspec"),
    "ARM:BE:32:v8":               ("ARM", "ARM8_be.slaspec"),
    "ARM:LE:32:v8T":              ("ARM", "ARM8_le.slaspec"),
    "ARM:BE:32:v8T":              ("ARM", "ARM8_be.slaspec"),
    "ARM:LEBE:32:v8LEInstruction": ("ARM", "ARM8_le.slaspec"),
    "ARM:LE:32:Cortex":           ("ARM", "ARM8_le.slaspec"),
    "ARM:BE:32:Cortex":           ("ARM", "ARM8_be.slaspec"),
    "ARM:LE:32:v8-m":             ("ARM", "ARM8m_le.slaspec"),
    "ARM:BE:32:v8-m":             ("ARM", "ARM8m_be.slaspec"),
    # AARCH64
    "AARCH64:LE:64:v8A":          ("AARCH64", "AARCH64.slaspec"),
    "AARCH64:BE:64:v8A":          ("AARCH64", "AARCH64BE.slaspec"),
    "AARCH64:LE:32:ilp32":        ("AARCH64", "AARCH64.slaspec"),
    "AARCH64:BE:32:ilp32":        ("AARCH64", "AARCH64BE.slaspec"),
}


def load_ghidra_reference(ghidra_home: Path, processor_or_lang_id: str) -> Path:
    """
    Return the path to the reference .slaspec for a Ghidra built-in processor.

    Accepts either:
      - A Ghidra language ID: "ARM:LE:32:v7"  → ARM7_le.slaspec
      - A processor name:     "ARM"            → first ARM*.slaspec found
    """
    # Try exact language ID match first
    if processor_or_lang_id in _LANG_ID_TO_SLASPEC:
        proc_dir, slaspec_name = _LANG_ID_TO_SLASPEC[processor_or_lang_id]
        path = ghidra_home / "Ghidra" / "Processors" / proc_dir / "data" / "languages" / slaspec_name
        if path.exists():
            return path

    # Fallback: treat as processor name and glob
    processor = processor_or_lang_id.split(":")[0]
    lang_dir = ghidra_home / "Ghidra" / "Processors" / processor / "data" / "languages"
    candidates = sorted(lang_dir.glob(f"{processor}*.slaspec"), key=lambda p: len(p.name))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No .slaspec found for '{processor_or_lang_id}' in {ghidra_home}. "
        f"Available ARM specs: ARM4_le, ARM7_le, ARM8_le, etc."
    )
