"""InstructionWriter Protocol and registry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosetta_schemas.models import InstructionDef, ISAMeta, OpcodeDef, RegisterDef


class InstructionWriter:
    """Protocol for streaming instruction output during the decode loop.

    open() is called once before the loop with ISA-level context and writes
    the non-instruction files (.pspec, .cspec, .ldefs) plus the .slaspec header.
    write_instruction() is called once per decoded instruction to append its
    constructor.  write_opcode_table() is used by the CISC path.  close()
    finalises the output and sets lang_dir.
    """

    def open(
        self,
        *,
        meta: ISAMeta,
        registers: list[RegisterDef],
        processor_name: str,
        out_dir: Path,
    ) -> None: ...

    def write_instruction(self, instr: InstructionDef) -> None: ...

    def write_opcode_table(self, opcode_map: list[OpcodeDef]) -> None: ...

    def close(self) -> None: ...

    @property
    def lang_dir(self) -> Path | None: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WRITER_REGISTRY: dict[str, type[Any]] = {}


def get_writer(name: str) -> Any:
    """Return a new InstructionWriter instance for *name*.

    Raises KeyError with a helpful message if the name is unknown.
    """
    from rosetta_generate_sla.writers.sla_writer import SlaInstructionWriter
    if not WRITER_REGISTRY:
        WRITER_REGISTRY["sla"] = SlaInstructionWriter
        WRITER_REGISTRY["sleigh"] = SlaInstructionWriter

    try:
        cls = WRITER_REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown output format {name!r}. Available: {sorted(WRITER_REGISTRY)}"
        )
    return cls()
