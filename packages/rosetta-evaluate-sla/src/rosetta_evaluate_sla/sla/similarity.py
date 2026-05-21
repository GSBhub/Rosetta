"""Structural similarity between a generated and reference .slaspec."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rosetta_evaluate_sla.sla.spec_loader import (
    extract_mnemonics,
    extract_register_names,
    load_slaspec_text,
)

log = logging.getLogger(__name__)


@dataclass
class SimilarityReport:
    instruction_coverage: float
    register_overlap: float
    generated_mnemonic_count: int
    reference_mnemonic_count: int
    common_mnemonics: int
    generated_register_count: int
    reference_register_count: int
    common_registers: int

    def summary(self) -> str:
        return (
            f"Instruction coverage: {self.instruction_coverage:.3f} "
            f"({self.common_mnemonics}/{self.reference_mnemonic_count})\n"
            f"Register overlap    : {self.register_overlap:.3f} "
            f"({self.common_registers}/{self.reference_register_count})"
        )


def compare(
    generated_slaspec: Path,
    reference_slaspec: Path,
) -> SimilarityReport:
    """Compute structural similarity between two .slaspec files."""
    gen_text = load_slaspec_text(generated_slaspec)
    ref_text = load_slaspec_text(reference_slaspec)

    gen_mnemonics = extract_mnemonics(gen_text)
    ref_mnemonics = extract_mnemonics(ref_text)
    gen_registers = extract_register_names(gen_text)
    ref_registers = extract_register_names(ref_text)

    common_m = gen_mnemonics & ref_mnemonics
    coverage = len(common_m) / len(ref_mnemonics) if ref_mnemonics else 0.0

    union_r = gen_registers | ref_registers
    common_r = gen_registers & ref_registers
    reg_jaccard = len(common_r) / len(union_r) if union_r else 0.0

    return SimilarityReport(
        instruction_coverage=coverage,
        register_overlap=reg_jaccard,
        generated_mnemonic_count=len(gen_mnemonics),
        reference_mnemonic_count=len(ref_mnemonics),
        common_mnemonics=len(common_m),
        generated_register_count=len(gen_registers),
        reference_register_count=len(ref_registers),
        common_registers=len(common_r),
    )
