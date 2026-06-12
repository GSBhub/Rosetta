"""SlaInstructionWriter — streams SLEIGH constructors via ModuleGenerator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosetta_schemas.models import InstructionDef, ISAMeta, OpcodeDef, RegisterDef

log = logging.getLogger(__name__)


class SlaInstructionWriter:
    """Writes a Ghidra processor module, one instruction constructor at a time.

    open()  → writes .pspec / .cspec / .ldefs + the .slaspec header (no constructors).
    write_instruction() → appends one constructor block using append_to_slaspec().
    write_opcode_table() → stashes rows; rendered at close() via the CISC template.
    close() → renders the CISC table if applicable, finalises lang_dir.
    """

    def __init__(self) -> None:
        self._lang_dir: Path | None = None
        self._processor_name: str = ""
        self._slaspec_paths: list[Path] = []
        self._is_cisc: bool = False
        self._opcode_map: list[OpcodeDef] = []
        self._meta: ISAMeta | None = None
        self._registers: list[RegisterDef] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def open(
        self,
        *,
        meta: ISAMeta,
        registers: list[RegisterDef],
        processor_name: str,
        out_dir: Path,
    ) -> None:
        from rosetta_generate_sla.sla.module_generator import ModuleGenerator
        from rosetta_schemas.models import ISASpec

        self._meta = meta
        self._registers = registers
        self._processor_name = processor_name
        self._is_cisc = meta.encoding_style == "opcode_table"

        # Generate the skeleton: aux files + slaspec header with zero constructors.
        spec = ISASpec(meta=meta, registers=registers, instructions=[], opcode_map=[])
        generator = ModuleGenerator()
        lang_dir = generator.generate(spec, processor_name, out_dir)
        self._lang_dir = lang_dir

        # Track slaspec paths (bi-endian produces two files).
        if meta.endian == "bi":
            self._slaspec_paths = [
                lang_dir / f"{processor_name}_le.slaspec",
                lang_dir / f"{processor_name}_be.slaspec",
            ]
        else:
            self._slaspec_paths = [lang_dir / f"{processor_name}.slaspec"]

        log.info("SlaInstructionWriter: opened %s", lang_dir)

    def write_instruction(self, instr: InstructionDef) -> None:
        if not self._slaspec_paths:
            raise RuntimeError("write_instruction() called before open()")

        from rosetta_generate_sla.sla.module_generator import ModuleGenerator
        from rosetta_schemas.models import ISASpec, ISAMeta

        # Stub meta — only encoding_style and instruction_sizes_bits matter here.
        stub_meta = ISAMeta(
            name=self._meta.name if self._meta else "Unknown",
            endian=self._meta.endian if self._meta else "little",  # type: ignore[arg-type]
            word_size_bits=self._meta.word_size_bits if self._meta else 32,
            alignment=self._meta.alignment if self._meta else 1,
            instruction_sizes_bits=self._meta.instruction_sizes_bits if self._meta else [32],
        )
        spec = ISASpec(meta=stub_meta, registers=[], instructions=[instr], opcode_map=[])
        generator = ModuleGenerator()
        for path in self._slaspec_paths:
            count = generator.append_to_slaspec(spec, path)
            if count:
                log.debug("Appended %s → %s", instr.mnemonic, path.name)

    def write_opcode_table(self, opcode_map: list[OpcodeDef]) -> None:
        self._opcode_map = list(opcode_map)

    def close(self) -> None:
        if self._is_cisc and self._opcode_map and self._slaspec_paths:
            self._render_cisc()
        log.info("SlaInstructionWriter: closed, lang_dir=%s", self._lang_dir)

    @property
    def lang_dir(self) -> Path | None:
        return self._lang_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_cisc(self) -> None:
        """Regenerate the slaspec from the accumulated opcode_map using the CISC template."""
        from rosetta_generate_sla.sla.module_generator import ModuleGenerator
        from rosetta_schemas.models import ISASpec

        assert self._meta is not None
        assert self._lang_dir is not None

        spec = ISASpec(
            meta=self._meta,
            registers=self._registers,
            instructions=[],
            opcode_map=self._opcode_map,
        )
        generator = ModuleGenerator()
        generator.generate(spec, self._processor_name, self._lang_dir.parent.parent.parent)
        log.info("SlaInstructionWriter: rendered CISC template (%d rows)", len(self._opcode_map))
