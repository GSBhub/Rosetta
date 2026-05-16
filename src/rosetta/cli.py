"""Rosetta CLI: ingest / generate / validate / evaluate / batch."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _load_env() -> None:
    """Load .env from the project root if present (simple key=value parser)."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

    # If JAVA_HOME is set, prepend its bin/ to PATH so subprocesses (sleigh, analyzeHeadless) find java.
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        java_bin = str(Path(java_home) / "bin")
        if java_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")


_load_env()


@click.group()
def cli() -> None:
    """Rosetta: ISA manual → Ghidra processor module generator."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("manual", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", required=True, help="Output docquery SQLite database path")
def ingest(manual: str, db: str) -> None:
    """Ingest a PDF manual into a docquery RAG database."""
    from docquery.config import Settings
    from docquery.ingestion.chunker import chunk
    from docquery.ingestion.pdf_loader import load
    from docquery.embeddings.provider import get_embeddings
    from docquery.storage.vector_store import VectorStore

    settings = Settings()
    click.echo(f"Loading {manual} ...")
    docs = load(manual)
    chunks = chunk(docs, settings)
    click.echo(f"  {len(docs)} pages → {len(chunks)} chunks")

    embeddings = get_embeddings(settings)
    dim = len(embeddings.embed_query("probe"))
    vs = VectorStore(db, embedding_dim=dim)
    vs.add_chunks(chunks, embeddings, batch_size=settings.embed_batch_size)
    vs.close()
    click.echo(f"Ingested into {db}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--db", required=True, help="docquery database from 'ingest'")
@click.option("--name", required=True, help="Processor name (used as file prefix)")
@click.option("--out", default="./output", show_default=True, help="Output directory")
@click.option("--spec-json", default=None, help="Load existing isa_spec.json (skip extraction)")
@click.option("--concurrency", default=4, show_default=True, help="Parallel instruction extraction workers")
def generate(db: str, name: str, out: str, spec_json: str | None, concurrency: int) -> None:
    """Extract ISA from database and generate a Ghidra processor module."""
    from rosetta.config import Settings
    from rosetta.extraction.isa_extractor import ISAExtractor
    from rosetta.generation.module_generator import ModuleGenerator

    settings = Settings()
    out_dir = Path(out)

    if spec_json:
        click.echo(f"Loading ISASpec from {spec_json}")
        spec = ISAExtractor.load(spec_json)
    else:
        click.echo(f"Extracting ISA from {db} ...")
        extractor = ISAExtractor(db_path=db, settings=settings)
        spec = extractor.extract(max_concurrent=concurrency)
        cache = Path(db).with_suffix("") .parent / f"{name}_isa_spec.json"
        extractor.save(spec, cache)
        click.echo(f"ISASpec cached to {cache}")

    click.echo(f"Generating processor module '{name}' → {out_dir} ...")
    generator = ModuleGenerator()
    lang_dir = generator.generate(spec, name, out_dir)
    click.echo(f"Module written to {lang_dir}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("module_dir", type=click.Path(exists=True, file_okay=False))
def validate(module_dir: str) -> None:
    """Compile .slaspec files in a generated processor module directory."""
    from rosetta.config import Settings
    from rosetta.validation.sleigh_compiler import compile_slaspec

    settings = Settings()
    ghidra_home = settings.ghidra_home

    lang_dir = Path(module_dir)
    # Accept either the top-level module dir or the languages/ subdir
    if not any(lang_dir.glob("*.slaspec")):
        lang_dir = lang_dir / "data" / "languages"

    specs = list(lang_dir.glob("*.slaspec"))
    if not specs:
        click.echo(f"No .slaspec files found in {lang_dir}", err=True)
        sys.exit(1)

    all_ok = True
    for slaspec in specs:
        result = compile_slaspec(slaspec, ghidra_home)
        status = "OK" if result.ok else f"FAILED ({len(result.errors)} errors)"
        click.echo(f"{slaspec.name}: {status}")
        if not result.ok:
            all_ok = False
            for err in result.errors:
                click.echo(f"  {err}", err=True)

    sys.exit(0 if all_ok else 1)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("module_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--reference",
    required=True,
    help="Ghidra language ID (e.g. ARM:LE:32:v7) or path to a reference .slaspec",
)
def evaluate(module_dir: str, reference: str) -> None:
    """Semantic comparison of a generated spec against a Ghidra reference spec."""
    from rosetta.config import Settings
    from rosetta.evaluation.similarity import compare
    from rosetta.evaluation.spec_loader import load_ghidra_reference

    settings = Settings()

    lang_dir = Path(module_dir)
    if not any(lang_dir.glob("*.slaspec")):
        lang_dir = lang_dir / "data" / "languages"

    generated = next(lang_dir.glob("*.slaspec"), None)
    if not generated:
        click.echo("No .slaspec found in module directory", err=True)
        sys.exit(1)

    # Resolve reference: either a file path or a Ghidra processor name derived
    # from the language ID (e.g. "ARM:LE:32:v7" → processor "ARM")
    ref_path = Path(reference)
    if not ref_path.exists():
        processor = reference.split(":")[0]
        ref_path = load_ghidra_reference(settings.ghidra_home, processor)

    click.echo(f"Generated : {generated}")
    click.echo(f"Reference : {ref_path}")
    click.echo("Computing similarity (this may take a minute) ...")

    report = compare(generated, ref_path, settings)
    click.echo("\n" + report.summary())


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--manifest",
    default="manifests/arm.yaml",
    show_default=True,
    type=click.Path(exists=True),
    help="YAML manifest file",
)
@click.option("--out", default="./output", show_default=True, help="Output directory")
@click.option(
    "--skip-extraction",
    is_flag=True,
    default=False,
    help="Re-use cached isa_spec.json files when available",
)
def batch(manifest: str, out: str, skip_extraction: bool) -> None:
    """Run the full pipeline for every target in the manifest."""
    from rosetta.config import Settings
    from rosetta.evaluation.batch_eval import print_summary_table, run_batch

    settings = Settings()
    results = run_batch(
        manifest_path=Path(manifest),
        out_dir=Path(out),
        ghidra_home=settings.ghidra_home,
        settings=settings,
        skip_extraction=skip_extraction,
    )
    print_summary_table(results)
    # Write JSON results
    results_file = Path(out) / "batch_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2))
    click.echo(f"Full results written to {results_file}")


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--slaspec",
    "slaspec_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Generated .slaspec to compare against all ARM variants.",
)
@click.option(
    "--results",
    "results_json",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="batch_results.json from a previous 'rosetta batch' run.",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Save graph image to this path (e.g. report.png). Omit to display interactively.",
)
@click.option(
    "--embeddings",
    is_flag=True,
    default=False,
    help="Include semantic similarity (requires Ollama running).",
)
@click.option(
    "--no-display",
    is_flag=True,
    default=False,
    help="Do not open an interactive window (useful in headless/CI environments).",
)
def graph(
    slaspec_path: str | None,
    results_json: str | None,
    out_path: str | None,
    embeddings: bool,
    no_display: bool,
) -> None:
    """Graph generated SLASpec effectiveness vs every Ghidra ARM processor.

    Two modes:\n
      --slaspec <file>   Compare one generated .slaspec against all 18 ARM/AARCH64 variants.\n
      --results <file>   Graph a batch_results.json produced by 'rosetta batch'.
    """
    import matplotlib
    if no_display or not out_path is None:
        matplotlib.use("Agg")  # headless backend when saving to file

    from rosetta.config import Settings
    from rosetta.evaluation.graph import (
        compare_all_variants,
        plot_batch_results,
        plot_variant_comparison,
    )

    if slaspec_path is None and results_json is None:
        click.echo("Provide --slaspec or --results. See --help.", err=True)
        sys.exit(1)

    show = not no_display
    img_path = Path(out_path) if out_path else None

    if slaspec_path:
        settings = Settings()
        click.echo(f"Comparing {slaspec_path} against all Ghidra ARM variants ...")
        results = compare_all_variants(
            generated_slaspec=Path(slaspec_path),
            ghidra_home=settings.ghidra_home,
            include_embeddings=embeddings,
            settings=settings if embeddings else None,
        )
        # Print table
        click.echo(f"\n{'Variant':<14} {'Coverage':>10} {'Reg Overlap':>12}")
        click.echo("-" * 40)
        for r in results:
            click.echo(f"{r['variant']:<14} {r['instruction_coverage']:>10.3f} {r['register_overlap']:>12.3f}")

        title = f"Generated SLASpec vs All Ghidra ARM Variants\n({Path(slaspec_path).name})"
        plot_variant_comparison(results, title=title, out_path=img_path, show=show)

    else:
        data = json.loads(Path(results_json).read_text())
        click.echo(f"Graphing {len(data)} batch results from {results_json} ...")
        plot_batch_results(data, out_path=img_path, show=show)

    if img_path:
        click.echo(f"Graph saved to {img_path}")
