"""Thin wrapper kept for API compatibility; logic lives in ModuleGenerator."""
from pathlib import Path
from rosetta.extraction.schemas import ISASpec
from rosetta.generation.module_generator import ModuleGenerator


def build_slaspec(spec: ISASpec, processor_name: str, out_dir: Path) -> Path:
    gen = ModuleGenerator()
    lang_dir = gen.generate(spec, processor_name, out_dir)
    return lang_dir / f"{processor_name}.slaspec"
