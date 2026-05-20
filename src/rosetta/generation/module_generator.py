"""Generates a complete Ghidra processor module directory from an ISASpec."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rosetta.extraction.schemas import ISASpec, InstructionDef, RegisterDef

log = logging.getLogger(__name__)

# Path to the templates/ directory relative to this package's repo root.
_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"

# Patterns that indicate the pcode_hint is natural language, not SLEIGH.
_BAD_PCODE = re.compile(
    r"Error:|Unknown\b|undefined\b|TODO\b|not extracted|"
    r"[A-Za-z_]\w*\s*\(|"              # function-call style: name(
    r"\bor\b|\band\b|\bnot\b|"         # logical English connectors
    r"[A-Z][a-z]+(?:_[A-Z][a-z]+)+|"  # CamelCase_Under pseudo-ops
    r"\bMem(?:ory)?\[|"                # Mem[...] / Memory[...] — ARM asm, not SLEIGH
    r"#[A-Za-z_]\w*|"                  # #imm ARM immediate syntax
    r"\b\w+\[\d+:\d+\]\s*=|"          # bit-slice LHS assignment (invalid in SLEIGH)
    r"\b[A-Z][a-z]\w*\b",             # mixed-case identifiers (Rd, Rn, RdLo…) — ARM
                                       # operand names, not defined SLEIGH varnodes
    re.IGNORECASE,
)

# Valid SLEIGH identifier: must start with letter or underscore.
_VALID_IDENT = re.compile(r'^[A-Za-z_]\w*$')
# Pure binary string (only 0s and 1s).
_PURE_BINARY = re.compile(r'^[01]+$')
# Single decimal number (a degenerate bit range like "19" instead of "19:19").
_SINGLE_INT = re.compile(r'^\d+$')


def _sanitize_pcode(hint: str) -> str:
    """Return hint if it looks like valid SLEIGH P-code, else a safe stub."""
    s = hint.strip() if hint else ""
    if not s:
        return "local tmp:4 = 0;"
    if not s.endswith(";"):
        return f"# {s[:80]}\n    local tmp:4 = 0;"
    if _BAD_PCODE.search(s):
        return f"# {s[:80]}\n    local tmp:4 = 0;"
    return s


def _normalize_bit_fields(bit_fields: dict[str, str]) -> dict[str, str]:
    """Return a cleaned bit_fields dict safe for SLEIGH token definitions."""
    result: dict[str, str] = {}
    for name, bit_range in bit_fields.items():
        if not _VALID_IDENT.match(name):
            continue
        s = str(bit_range).strip()
        # Skip anything with spaces (multi-range garbage like "0:2, 1:3").
        if ' ' in s or s.count(':') > 1:
            continue
        # Single integer like "19" → normalize to "19:19".
        if _SINGLE_INT.match(s):
            s = f"{s}:{s}"
        # Must now be exactly "high:low".
        parts = s.split(':')
        if len(parts) != 2:
            continue
        try:
            int(parts[0]), int(parts[1])
        except ValueError:
            continue
        result[name] = s
    return result


def _normalize_bit_constraints(
    bit_constraints: dict[str, str],
    bit_fields: dict[str, str],
) -> dict[str, str]:
    """Keep only constraints with valid identifiers and binary values fitting their field."""
    result: dict[str, str] = {}
    for field, val in bit_constraints.items():
        if not _VALID_IDENT.match(field):
            continue
        v = str(val).strip()
        if not _PURE_BINARY.match(v):
            continue
        # Determine field width from bit_fields (format "high:low").
        if field in bit_fields:
            parts = str(bit_fields[field]).split(":")
            try:
                field_width = abs(int(parts[0]) - int(parts[1])) + 1
            except (ValueError, IndexError):
                field_width = None
            if field_width is not None and len(v) > field_width:
                continue  # constraint value too wide for the field — skip it
        result[field] = v
    return result


def _normalize_instruction(instr: InstructionDef) -> InstructionDef:
    """Return a copy of instr with SLEIGH-safe field data."""
    ni = copy.copy(instr)
    # encoding_bits must be a positive multiple of 8.
    if ni.encoding_bits <= 0:
        ni.encoding_bits = 32
    elif ni.encoding_bits % 8 != 0:
        ni.encoding_bits = ((ni.encoding_bits + 7) // 8) * 8
    # Mnemonics with spaces are parsed by SLEIGH as display + operand tokens.
    ni.mnemonic = ni.mnemonic.replace(" ", "_")
    ni.bit_fields = _normalize_bit_fields(ni.bit_fields)
    # Bit constraints from LLM are unreliable (overlapping fields, wrong widths) and
    # produce SLEIGH "impossible to match" / "identical patterns" errors.  Clear them
    # all — every instruction falls back to a unique opXXstub=N pattern, which is safe.
    ni.bit_constraints = {}
    # Collapse multi-line semantics so newlines don't escape into the SLEIGH comment area.
    ni.semantics = " ".join(ni.semantics.splitlines())
    # Clear operands: SLEIGH requires operand names to be defined token fields or
    # sub-tables. Without attach registers, display operands are undefined symbols.
    # A stub spec with just mnemonics compiles and allows Ghidra to load binaries.
    ni.operands = []
    return ni


def _find_register(registers: list[RegisterDef], *candidate_aliases: str) -> str:
    """Return the canonical name of the first register that matches any alias."""
    upper_aliases = {a.upper() for a in candidate_aliases}
    for reg in registers:
        names = {reg.name.upper()} | {a.upper() for a in reg.aliases}
        if names & upper_aliases:
            return reg.name
    # Fallback: return the first register if no match
    return registers[0].name if registers else "PC"


class ModuleGenerator:
    def __init__(self, templates_dir: Path | None = None):
        tdir = templates_dir or _TEMPLATES_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(tdir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["sanitize_pcode"] = _sanitize_pcode

    def generate(self, spec: ISASpec, processor_name: str, out_dir: Path) -> Path:
        """
        Render all four processor module files into:
            <out_dir>/<processor_name>/data/languages/

        Returns the languages/ directory path.
        """
        lang_dir = out_dir / processor_name / "data" / "languages"
        lang_dir.mkdir(parents=True, exist_ok=True)

        pc = _find_register(spec.registers, "PC", "R15", "X30", "IP")
        sp = _find_register(spec.registers, "SP", "R13", "X2", "X29")

        # Normalize instructions: fix encoding_bits=0, bad field names, non-binary constraints.
        normalized_instructions = [_normalize_instruction(i) for i in spec.instructions]

        # Build canonical field widths: first declaration per field name wins (matches
        # template behaviour where only the first occurrence of a field name is emitted).
        canonical_widths: dict[str, int] = {}
        for instr in normalized_instructions:
            for fname, frange in instr.bit_fields.items():
                if fname not in canonical_widths:
                    parts = str(frange).split(":")
                    try:
                        canonical_widths[fname] = abs(int(parts[0]) - int(parts[1])) + 1
                    except (ValueError, IndexError):
                        pass

        # Drop constraints whose value is wider than the canonical token field width.
        for instr in normalized_instructions:
            instr.bit_constraints = {
                f: v for f, v in instr.bit_constraints.items()
                if f not in canonical_widths or len(v) <= canonical_widths[f]
            }

        # Resolve duplicate constraint patterns: if two+ instructions share the same
        # pattern (same field=value set), SLEIGH rejects them as ambiguous.  Clear
        # all constraints for the duplicate group so they fall back to unique stubs.
        pattern_seen: dict[frozenset, int] = {}  # signature → first index
        duplicate_indices: set[int] = set()
        for idx, instr in enumerate(normalized_instructions):
            sig = frozenset(instr.bit_constraints.items())
            if not sig:  # already a stub — no conflict possible
                continue
            if sig in pattern_seen:
                duplicate_indices.add(pattern_seen[sig])
                duplicate_indices.add(idx)
            else:
                pattern_seen[sig] = idx
        for idx in duplicate_indices:
            normalized_instructions[idx].bit_constraints = {}

        # Ensure every encoding_bits value used by an instruction has a token defined.
        declared_widths = set(spec.meta.instruction_sizes_bits)
        extra_widths = {i.encoding_bits for i in normalized_instructions} - declared_widths
        if extra_widths:
            log.warning("Adding undeclared encoding widths to token list: %s", sorted(extra_widths))
        all_widths = sorted(declared_widths | extra_widths)
        # Remove 0 from widths (invalid SLEIGH token size).
        all_widths = [w for w in all_widths if w > 0]

        meta = copy.copy(spec.meta)
        meta.instruction_sizes_bits = all_widths

        ctx = {
            "meta": meta,
            "registers": spec.registers,
            "instructions": normalized_instructions,
            "processor_name": processor_name,
            "pc_register": pc,
            "sp_register": sp,
        }

        files = {
            "processor.slaspec.j2": f"{processor_name}.slaspec",
            "processor.pspec.j2": f"{processor_name}.pspec",
            "processor.cspec.j2": f"{processor_name}.cspec",
            "processor.ldefs.j2": f"{processor_name}.ldefs",
        }

        for template_name, out_filename in files.items():
            tmpl = self._env.get_template(template_name)
            rendered = tmpl.render(**ctx)
            dest = lang_dir / out_filename
            dest.write_text(rendered)
            log.info("Wrote %s", dest)

        log.info("Module generated in %s", lang_dir)
        return lang_dir

    def append_to_slaspec(self, spec: ISASpec, slaspec_path: Path) -> int:
        """Append new instruction constructor blocks to an existing .slaspec file.

        Skips any instruction whose mnemonic is already present in the file.
        Returns the count of constructors appended.
        """
        existing_text = slaspec_path.read_text()

        defined: set[str] = set(re.findall(r"^:([A-Z][A-Z0-9]*)\b", existing_text, re.MULTILINE))
        new_instrs = [i for i in spec.instructions if i.mnemonic.upper() not in defined]

        if not new_instrs:
            log.info("append_to_slaspec: no new instructions to append to %s", slaspec_path.name)
            return 0

        constrained = [i.mnemonic for i in new_instrs if i.bit_constraints]
        if constrained:
            log.warning(
                "append_to_slaspec: %d instruction(s) have bit_constraints that may require "
                "token declarations not present in %s: %s",
                len(constrained), slaspec_path.name, ", ".join(constrained[:10]),
            )

        tmpl = self._env.get_template("constructor_block.j2")
        rendered = tmpl.render(instructions=[_normalize_instruction(i) for i in new_instrs])

        separator = f"\n# --- appended by rosetta ({len(new_instrs)} instruction(s)) ---\n"
        slaspec_path.write_text(existing_text.rstrip() + "\n" + separator + rendered)
        log.info("Appended %d constructor(s) to %s", len(new_instrs), slaspec_path)
        return len(new_instrs)
