"""Aggregate evaluation across multiple manifest targets."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from docquery.config import Settings as DocSettings

from rosetta.evaluation.similarity import SimilarityReport, compare
from rosetta.evaluation.spec_loader import load_ghidra_reference
from rosetta.extraction.isa_extractor import ISAExtractor
from rosetta.generation.module_generator import ModuleGenerator
from rosetta.validation.sleigh_compiler import SleighResult, compile_slaspec

log = logging.getLogger(__name__)


def _resolve_env(value: str) -> str:
    """Expand $ENV_VAR references in manifest strings."""
    for key, val in os.environ.items():
        value = value.replace(f"${key}", val)
    return value


def load_manifest(manifest_path: Path) -> list[dict]:
    raw = yaml.safe_load(manifest_path.read_text())
    targets = raw.get("targets", [])
    for t in targets:
        for key in ("manual", "db", "reference_slaspec"):
            if key in t and isinstance(t[key], str):
                t[key] = _resolve_env(t[key])
    return targets


def run_batch(
    manifest_path: Path,
    out_dir: Path,
    ghidra_home: Path,
    settings: DocSettings | None = None,
    skip_extraction: bool = False,
    target_filter: str | None = None,
    max_concurrent: int = 4,
) -> list[dict]:
    """
    For each target in the manifest:
      1. (Optionally) extract ISASpec from the manual
      2. Generate the processor module
      3. Compile .slaspec → validate syntax
      4. Evaluate against the reference spec

    Returns a list of result dicts (one per target).
    """
    settings = settings or DocSettings()
    targets = load_manifest(manifest_path)
    if target_filter:
        targets = [t for t in targets if t["id"] == target_filter]
        if not targets:
            raise ValueError(f"No manifest target with id={target_filter!r}")
    generator = ModuleGenerator()
    results = []

    for target in targets:
        tid = target["id"]
        name = target["name"]
        db_path = Path(target["db"])
        ref_slaspec_path = Path(target["reference_slaspec"])
        log.info("=== Batch target: %s ===", tid)

        row: dict = {"id": tid, "name": name}

        # 1. Extract (unless the isa_spec.json already exists and skip_extraction=True)
        spec_json = db_path.parent / f"{tid}_isa_spec.json"
        if skip_extraction and spec_json.exists():
            log.info("Loading cached ISASpec from %s", spec_json)
            spec = ISAExtractor.load(spec_json)
        else:
            if not db_path.exists():
                log.warning("DB not found for target %s: %s — skipping", tid, db_path)
                row["error"] = "db not found"
                results.append(row)
                continue
            extractor = ISAExtractor(db_path=db_path, settings=settings)
            spec = extractor.extract(max_concurrent=max_concurrent)
            extractor.save(spec, spec_json)

        # 2. Generate
        lang_dir = generator.generate(spec, name, out_dir)
        slaspec_path = lang_dir / f"{name}.slaspec"
        row["generated_slaspec"] = str(slaspec_path)

        # 3. Validate (compile)
        compile_result: SleighResult = compile_slaspec(slaspec_path, ghidra_home)
        row["compile_ok"] = compile_result.ok
        row["compile_errors"] = len(compile_result.errors)

        # 4. Evaluate similarity
        if not ref_slaspec_path.exists():
            log.warning("Reference slaspec not found: %s", ref_slaspec_path)
            row["error"] = "reference not found"
        else:
            report = compare(slaspec_path, ref_slaspec_path, settings)
            row["semantic_similarity"] = round(report.semantic_similarity, 4)
            row["instruction_coverage"] = round(report.instruction_coverage, 4)
            row["register_overlap"] = round(report.register_overlap, 4)
            row["common_mnemonics"] = report.common_mnemonics
            row["reference_mnemonic_count"] = report.reference_mnemonic_count

        results.append(row)
        log.info("Target %s done: %s", tid, row)

    return results


def print_summary_table(results: list[dict]) -> None:
    cols = ["id", "compile_ok", "semantic_similarity", "instruction_coverage", "register_overlap"]
    header = "  ".join(f"{c:<25}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for row in results:
        line = "  ".join(
            f"{str(row.get(c, 'N/A')):<25}" for c in cols
        )
        print(line)
    print("=" * len(header) + "\n")
