"""Generate comparison graphs of generated vs Ghidra reference specs."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

ARM_VARIANTS: list[tuple[str, str, str]] = [
    ("ARM4 LE",    "ARM",    "ARM4_le.slaspec"),
    ("ARM4 BE",    "ARM",    "ARM4_be.slaspec"),
    ("ARM4T LE",   "ARM",    "ARM4t_le.slaspec"),
    ("ARM4T BE",   "ARM",    "ARM4t_be.slaspec"),
    ("ARM5 LE",    "ARM",    "ARM5_le.slaspec"),
    ("ARM5 BE",    "ARM",    "ARM5_be.slaspec"),
    ("ARM5T LE",   "ARM",    "ARM5t_le.slaspec"),
    ("ARM5T BE",   "ARM",    "ARM5t_be.slaspec"),
    ("ARM6 LE",    "ARM",    "ARM6_le.slaspec"),
    ("ARM6 BE",    "ARM",    "ARM6_be.slaspec"),
    ("ARM7 LE",    "ARM",    "ARM7_le.slaspec"),
    ("ARM7 BE",    "ARM",    "ARM7_be.slaspec"),
    ("ARM8 LE",    "ARM",    "ARM8_le.slaspec"),
    ("ARM8 BE",    "ARM",    "ARM8_be.slaspec"),
    ("ARM8M LE",   "ARM",    "ARM8m_le.slaspec"),
    ("ARM8M BE",   "ARM",    "ARM8m_be.slaspec"),
    ("AARCH64 LE", "AARCH64","AARCH64.slaspec"),
    ("AARCH64 BE", "AARCH64","AARCH64BE.slaspec"),
]


def _structural_metrics(gen_text: str, ref_text: str) -> dict[str, float]:
    from rosetta_evaluate_sla.sla.spec_loader import extract_mnemonics, extract_register_names
    gen_m = extract_mnemonics(gen_text)
    ref_m = extract_mnemonics(ref_text)
    gen_r = extract_register_names(gen_text)
    ref_r = extract_register_names(ref_text)
    coverage = len(gen_m & ref_m) / len(ref_m) if ref_m else 0.0
    union_r = gen_r | ref_r
    reg_jaccard = len(gen_r & ref_r) / len(union_r) if union_r else 0.0
    return {"instruction_coverage": coverage, "register_overlap": reg_jaccard}


def compare_all_variants(
    generated_slaspec: Path,
    ghidra_home: Path,
    include_embeddings: bool = False,
    settings: object = None,
) -> list[dict]:
    from rosetta_evaluate_sla.sla.spec_loader import load_slaspec_text

    gen_text = load_slaspec_text(generated_slaspec)
    gen_vecs = None

    if include_embeddings:
        from docquery.config import Settings as DocSettings
        from docquery.embeddings.provider import get_embeddings
        from rosetta_evaluate_sla.sla.similarity import _chunk_text, _mean_pairwise_similarity
        cfg = settings or DocSettings()
        embedder = get_embeddings(cfg)
        gen_vecs = embedder.embed_documents(_chunk_text(gen_text))

    results = []
    for label, proc_dir, slaspec_name in ARM_VARIANTS:
        ref_path = ghidra_home / "Ghidra" / "Processors" / proc_dir / "data" / "languages" / slaspec_name
        if not ref_path.exists():
            log.warning("Reference not found: %s", ref_path)
            continue
        ref_text = load_slaspec_text(ref_path)
        row = {"variant": label, "slaspec": slaspec_name}
        row.update(_structural_metrics(gen_text, ref_text))
        if include_embeddings and gen_vecs is not None:
            from rosetta_evaluate_sla.sla.similarity import _chunk_text, _mean_pairwise_similarity
            ref_vecs = embedder.embed_documents(_chunk_text(ref_text))
            row["semantic_similarity"] = _mean_pairwise_similarity(gen_vecs, ref_vecs)
        results.append(row)
        log.info("%s → coverage=%.3f reg=%.3f", label, row["instruction_coverage"], row["register_overlap"])

    results.sort(key=lambda r: r["instruction_coverage"], reverse=True)
    return results


def plot_variant_comparison(
    results: list[dict],
    title: str = "Generated SLASpec vs Ghidra ARM Variants",
    out_path: Path | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["variant"] for r in results]
    coverage = [r["instruction_coverage"] for r in results]
    reg_overlap = [r["register_overlap"] for r in results]
    has_sem = "semantic_similarity" in results[0] if results else False
    sem_sim = [r.get("semantic_similarity", 0.0) for r in results]

    x = np.arange(len(labels))
    width = 0.25 if has_sem else 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6))
    bars1 = ax.bar(x - width, coverage,    width, label="Instruction coverage", color="#2196F3")
    bars2 = ax.bar(x,         reg_overlap, width, label="Register overlap (Jaccard)", color="#4CAF50")
    if has_sem:
        ax.bar(x + width, sem_sim, width, label="Semantic similarity", color="#FF9800")

    ax.set_xlabel("Ghidra ARM Variant")
    ax.set_ylabel("Score (0 – 1)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        log.info("Graph saved to %s", out_path)
    if show:
        plt.show()
    plt.close()


def plot_batch_results(
    results: list[dict],
    title: str = "Batch Evaluation: Generated vs Reference",
    out_path: Path | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in results if "instruction_coverage" in r]
    if not rows:
        log.warning("No metric data in results to plot")
        return

    labels = [r["id"] for r in rows]
    coverage = [r.get("instruction_coverage", 0) for r in rows]
    reg_overlap = [r.get("register_overlap", 0) for r in rows]
    has_sem = any("semantic_similarity" in r for r in rows)
    sem_sim = [r.get("semantic_similarity", 0) for r in rows]

    x = np.arange(len(labels))
    width = 0.25 if has_sem else 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    ax.bar(x - width, coverage,    width, label="Instruction coverage", color="#2196F3")
    ax.bar(x,         reg_overlap, width, label="Register overlap",     color="#4CAF50")
    if has_sem:
        ax.bar(x + width, sem_sim, width, label="Semantic similarity",  color="#FF9800")

    compile_status = [("✓" if r.get("compile_ok") else "✗") for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{s}" for l, s in zip(labels, compile_status)], fontsize=9)
    ax.set_xlabel("ARM Variant (✓=compiled OK  ✗=compile error)")
    ax.set_ylabel("Score (0 – 1)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        log.info("Graph saved to %s", out_path)
    if show:
        plt.show()
    plt.close()
