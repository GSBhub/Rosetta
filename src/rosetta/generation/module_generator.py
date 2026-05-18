"""Generates a complete Ghidra processor module directory from an ISASpec."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rosetta.extraction.schemas import ISASpec, InstructionDef, RegisterDef

log = logging.getLogger(__name__)

# Path to the templates/ directory relative to this package's repo root.
_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"


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

        ctx = {
            "meta": spec.meta,
            "registers": spec.registers,
            "instructions": spec.instructions,
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
        rendered = tmpl.render(instructions=new_instrs)

        separator = f"\n# --- appended by rosetta ({len(new_instrs)} instruction(s)) ---\n"
        slaspec_path.write_text(existing_text.rstrip() + "\n" + separator + rendered)
        log.info("Appended %d constructor(s) to %s", len(new_instrs), slaspec_path)
        return len(new_instrs)
