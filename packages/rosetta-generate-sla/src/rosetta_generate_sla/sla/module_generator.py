"""Generates a complete Ghidra processor module directory from an ISASpec."""

from __future__ import annotations

import copy
import logging
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rosetta_schemas.models import ISASpec

from rosetta_generate_sla.sla.sanitize import (
    find_register,
    normalize_instruction,
    sanitize_pcode,
)

log = logging.getLogger(__name__)


def _get_templates_dir() -> Path:
    return Path(str(files("rosetta_generate_sla.sla") / "templates"))


class ModuleGenerator:
    def __init__(self, templates_dir: Path | None = None):
        tdir = templates_dir or _get_templates_dir()
        self._env = Environment(
            loader=FileSystemLoader(str(tdir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["sanitize_pcode"] = sanitize_pcode

    def generate(self, spec: ISASpec, processor_name: str, out_dir: Path) -> Path:
        """Render all four processor module files into <out_dir>/<processor_name>/data/languages/.

        Returns the languages/ directory path.
        """
        lang_dir = out_dir / processor_name / "data" / "languages"
        lang_dir.mkdir(parents=True, exist_ok=True)

        pc = find_register(spec.registers, "PC", "IP", "EIP", "RIP", description_keyword="program counter")
        sp = find_register(spec.registers, "SP", "ESP", "RSP", description_keyword="stack pointer")

        normalized_instructions = [normalize_instruction(i) for i in spec.instructions]

        canonical_widths: dict[str, int] = {}
        for instr in normalized_instructions:
            for fname, frange in instr.bit_fields.items():
                if fname not in canonical_widths:
                    parts = str(frange).split(":")
                    try:
                        canonical_widths[fname] = abs(int(parts[0]) - int(parts[1])) + 1
                    except (ValueError, IndexError):
                        pass

        for instr in normalized_instructions:
            instr.bit_constraints = {
                f: v for f, v in instr.bit_constraints.items()
                if f not in canonical_widths or len(v) <= canonical_widths[f]
            }

        pattern_seen: dict[frozenset, int] = {}
        duplicate_indices: set[int] = set()
        for idx, instr in enumerate(normalized_instructions):
            sig = frozenset(instr.bit_constraints.items())
            if not sig:
                continue
            if sig in pattern_seen:
                duplicate_indices.add(pattern_seen[sig])
                duplicate_indices.add(idx)
            else:
                pattern_seen[sig] = idx
        for idx in duplicate_indices:
            normalized_instructions[idx].bit_constraints = {}

        declared_widths = set(spec.meta.instruction_sizes_bits)
        extra_widths = {i.encoding_bits for i in normalized_instructions} - declared_widths
        if extra_widths:
            log.warning("Adding undeclared encoding widths: %s", sorted(extra_widths))
        all_widths = sorted(w for w in (declared_widths | extra_widths) if w > 0)

        meta = copy.copy(spec.meta)
        meta.instruction_sizes_bits = all_widths

        is_cisc = meta.encoding_style == "opcode_table"
        slaspec_template = "processor.slaspec.cisc.j2" if is_cisc else "processor.slaspec.j2"

        ctx = {
            "meta": meta,
            "registers": spec.registers,
            "instructions": normalized_instructions,
            "opcode_map": spec.opcode_map,
            "processor_name": processor_name,
            "pc_register": pc,
            "sp_register": sp,
        }

        if meta.endian == "bi":
            # Generate both LE and BE slaspec files plus shared pspec/cspec/ldefs.
            for endian_val, suffix in [("little", "_le"), ("big", "_be")]:
                endian_meta = copy.copy(meta)
                endian_meta.endian = endian_val  # type: ignore[assignment]
                endian_ctx = {**ctx, "meta": endian_meta}
                tmpl = self._env.get_template(slaspec_template)
                rendered = tmpl.render(**endian_ctx)
                dest = lang_dir / f"{processor_name}{suffix}.slaspec"
                dest.write_text(rendered)
                log.info("Wrote %s", dest)
            shared_files = {
                "processor.pspec.j2": f"{processor_name}.pspec",
                "processor.cspec.j2": f"{processor_name}.cspec",
                "processor.ldefs.j2": f"{processor_name}.ldefs",
            }
            for template_name, out_filename in shared_files.items():
                tmpl = self._env.get_template(template_name)
                rendered = tmpl.render(**ctx)
                dest = lang_dir / out_filename
                dest.write_text(rendered)
                log.info("Wrote %s", dest)
        else:
            files_map = {
                slaspec_template:       f"{processor_name}.slaspec",
                "processor.pspec.j2":  f"{processor_name}.pspec",
                "processor.cspec.j2":  f"{processor_name}.cspec",
                "processor.ldefs.j2":  f"{processor_name}.ldefs",
            }
            for template_name, out_filename in files_map.items():
                tmpl = self._env.get_template(template_name)
                rendered = tmpl.render(**ctx)
                dest = lang_dir / out_filename
                dest.write_text(rendered)
                log.info("Wrote %s", dest)

        log.info("Module generated in %s", lang_dir)
        return lang_dir

    def append_to_slaspec(self, spec: ISASpec, slaspec_path: Path) -> int:
        """Append new instruction constructor blocks to an existing .slaspec. Returns count appended."""
        import re
        existing_text = slaspec_path.read_text()
        defined: set[str] = set(re.findall(r"^:([A-Z][A-Z0-9]*)\b", existing_text, re.MULTILINE))
        new_instrs = [i for i in spec.instructions if i.mnemonic.upper() not in defined]

        if not new_instrs:
            log.info("append_to_slaspec: no new instructions for %s", slaspec_path.name)
            return 0

        tmpl = self._env.get_template("constructor_block.j2")
        rendered = tmpl.render(instructions=[normalize_instruction(i) for i in new_instrs])
        separator = f"\n# --- appended by rosetta ({len(new_instrs)} instruction(s)) ---\n"
        slaspec_path.write_text(existing_text.rstrip() + "\n" + separator + rendered)
        log.info("Appended %d constructor(s) to %s", len(new_instrs), slaspec_path)
        return len(new_instrs)
