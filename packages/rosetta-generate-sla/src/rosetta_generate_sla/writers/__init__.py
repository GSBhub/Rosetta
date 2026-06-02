"""Pluggable output writers for the decode pipeline."""

from rosetta_generate_sla.writers.base import WRITER_REGISTRY, InstructionWriter, get_writer
from rosetta_generate_sla.writers.sla_writer import SlaInstructionWriter

__all__ = ["InstructionWriter", "SlaInstructionWriter", "WRITER_REGISTRY", "get_writer"]
