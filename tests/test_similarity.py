"""Tests for similarity metric math (no LLM/Ollama required)."""

import math
import pytest

from rosetta.evaluation.similarity import (
    SimilarityReport,
    _chunk_text,
    _cosine,
    _mean_pairwise_similarity,
)
from rosetta.evaluation.spec_loader import extract_mnemonics, extract_register_names


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = [1.0, 2.0, 3.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mean_pairwise_identical():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    score = _mean_pairwise_similarity(vecs, vecs)
    assert score == pytest.approx(1.0)


def test_mean_pairwise_empty():
    assert _mean_pairwise_similarity([], [[1.0, 0.0]]) == 0.0
    assert _mean_pairwise_similarity([[1.0, 0.0]], []) == 0.0


def test_mean_pairwise_partial_overlap():
    # A = two orthogonal unit vectors
    # B = one vector identical to A[0]
    a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    b = [[1.0, 0.0, 0.0]]
    score = _mean_pairwise_similarity(a, b)
    # A[0]→B[0]=1.0, A[1]→B[0]=0.0 → mean=0.5
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Chunk text
# ---------------------------------------------------------------------------

def test_chunk_text_single_chunk():
    text = "hello world"
    chunks = _chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_chunks():
    text = "x" * 1000
    chunks = _chunk_text(text, chunk_size=100)
    assert len(chunks) > 1
    # Each chunk should be at most 100 chars
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_overlap():
    text = "abcdefghij"  # 10 chars
    chunks = _chunk_text(text, chunk_size=5)
    # With 10% overlap (0 chars at chunk_size=5), step = 5
    # chunks start at 0, 5 → 2 chunks
    assert len(chunks) >= 2
    assert chunks[0][:5] == "abcde"


# ---------------------------------------------------------------------------
# Structural metrics (instruction coverage + register overlap)
# ---------------------------------------------------------------------------

def test_instruction_coverage_full():
    gen = ":ADD rd is op=0 {} \n:SUB rd is op=1 {}"
    ref = ":ADD rd is op=0 {} \n:SUB rd is op=1 {}"
    gen_m = extract_mnemonics(gen)
    ref_m = extract_mnemonics(ref)
    coverage = len(gen_m & ref_m) / len(ref_m)
    assert coverage == pytest.approx(1.0)


def test_instruction_coverage_zero():
    gen = ":MOV rd is op=0 {}"
    ref = ":ADD rd is op=0 {}"
    gen_m = extract_mnemonics(gen)
    ref_m = extract_mnemonics(ref)
    coverage = len(gen_m & ref_m) / len(ref_m)
    assert coverage == pytest.approx(0.0)


def test_instruction_coverage_partial():
    gen = ":ADD rd is op=0 {} \n:MOV rd is op=2 {}"
    ref = ":ADD rd is op=0 {} \n:SUB rd is op=1 {} \n:MOV rd is op=2 {}"
    gen_m = extract_mnemonics(gen)
    ref_m = extract_mnemonics(ref)
    coverage = len(gen_m & ref_m) / len(ref_m)
    assert coverage == pytest.approx(2 / 3)


def test_register_overlap_jaccard():
    gen = "define register offset=0 size=4 [ R0 R1 R2 ];"
    ref = "define register offset=0 size=4 [ R0 R1 R3 R4 ];"
    gen_r = extract_register_names(gen)
    ref_r = extract_register_names(ref)
    union = gen_r | ref_r
    common = gen_r & ref_r
    jaccard = len(common) / len(union)
    # {R0, R1} ∩ {R0, R1, R2, R3, R4} but gen={R0,R1,R2}, ref={R0,R1,R3,R4}
    # common={R0,R1}, union={R0,R1,R2,R3,R4}, jaccard=2/5
    assert jaccard == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# SimilarityReport
# ---------------------------------------------------------------------------

def test_similarity_report_summary_format():
    r = SimilarityReport(
        semantic_similarity=0.75,
        instruction_coverage=0.60,
        register_overlap=0.50,
        generated_mnemonic_count=30,
        reference_mnemonic_count=50,
        common_mnemonics=30,
        generated_register_count=10,
        reference_register_count=20,
        common_registers=10,
    )
    summary = r.summary()
    assert "0.750" in summary
    assert "0.600" in summary
    assert "0.500" in summary
