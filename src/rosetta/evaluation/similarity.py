"""Semantic similarity between a generated and reference .slaspec."""

from __future__ import annotations

import math
import logging
from pathlib import Path
from dataclasses import dataclass

from docquery.config import Settings as DocSettings
from docquery.embeddings.provider import get_embeddings
from langchain_core.documents import Document

from rosetta.evaluation.spec_loader import (
    extract_mnemonics,
    extract_register_names,
    load_slaspec_text,
)

log = logging.getLogger(__name__)

_CHUNK_SIZE = 1000


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """Split text into fixed-size chunks with 10% overlap."""
    overlap = chunk_size // 10
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean_pairwise_similarity(
    vecs_a: list[list[float]], vecs_b: list[list[float]]
) -> float:
    """
    For each chunk in A, find the max cosine similarity against all chunks in B.
    Return the mean of those per-chunk maxima.
    """
    if not vecs_a or not vecs_b:
        return 0.0
    scores = []
    for va in vecs_a:
        best = max(_cosine(va, vb) for vb in vecs_b)
        scores.append(best)
    return sum(scores) / len(scores)


@dataclass
class SimilarityReport:
    semantic_similarity: float        # 0-1, mean best-match cosine
    instruction_coverage: float       # |generated ∩ reference| / |reference|
    register_overlap: float           # Jaccard similarity of register name sets
    generated_mnemonic_count: int
    reference_mnemonic_count: int
    common_mnemonics: int
    generated_register_count: int
    reference_register_count: int
    common_registers: int

    def summary(self) -> str:
        return (
            f"Semantic similarity : {self.semantic_similarity:.3f}\n"
            f"Instruction coverage: {self.instruction_coverage:.3f} "
            f"({self.common_mnemonics}/{self.reference_mnemonic_count})\n"
            f"Register overlap    : {self.register_overlap:.3f} "
            f"({self.common_registers}/{self.reference_register_count})"
        )


def compare(
    generated_slaspec: Path,
    reference_slaspec: Path,
    settings: DocSettings | None = None,
) -> SimilarityReport:
    """Compute semantic + structural similarity between two .slaspec files."""
    settings = settings or DocSettings()

    gen_text = load_slaspec_text(generated_slaspec)
    ref_text = load_slaspec_text(reference_slaspec)

    gen_mnemonics = extract_mnemonics(gen_text)
    ref_mnemonics = extract_mnemonics(ref_text)
    gen_registers = extract_register_names(gen_text)
    ref_registers = extract_register_names(ref_text)

    # Structural metrics (no LLM needed)
    common_m = gen_mnemonics & ref_mnemonics
    coverage = len(common_m) / len(ref_mnemonics) if ref_mnemonics else 0.0

    union_r = gen_registers | ref_registers
    common_r = gen_registers & ref_registers
    reg_jaccard = len(common_r) / len(union_r) if union_r else 0.0

    # Semantic embedding similarity
    log.info("Embedding generated spec (%d chars) ...", len(gen_text))
    embedder = get_embeddings(settings)
    gen_chunks = _chunk_text(gen_text)
    ref_chunks = _chunk_text(ref_text)

    gen_vecs = embedder.embed_documents(gen_chunks)
    ref_vecs = embedder.embed_documents(ref_chunks)

    sem_sim = _mean_pairwise_similarity(gen_vecs, ref_vecs)

    return SimilarityReport(
        semantic_similarity=sem_sim,
        instruction_coverage=coverage,
        register_overlap=reg_jaccard,
        generated_mnemonic_count=len(gen_mnemonics),
        reference_mnemonic_count=len(ref_mnemonics),
        common_mnemonics=len(common_m),
        generated_register_count=len(gen_registers),
        reference_register_count=len(ref_registers),
        common_registers=len(common_r),
    )
