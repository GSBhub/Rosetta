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


def _install_llm_providers() -> None:
    """Patch docquery's get_llm in-place to support LLM_PROVIDER=anthropic."""
    if os.environ.get("LLM_PROVIDER") != "anthropic":
        return
    from rosetta.utils.llm import get_llm as _extended
    import docquery.embeddings.llm as _llm_mod
    _llm_mod.get_llm = _extended


_install_llm_providers()


def _apply_model_overrides(
    llm_model: str | None,
    llm_base_url: str | None,
    embed_model: str | None,
    embed_base_url: str | None,
) -> None:
    """Override model/endpoint env vars before Settings() is instantiated."""
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
    if llm_base_url:
        os.environ["LLM_BASE_URL"] = llm_base_url
    if embed_model:
        os.environ["EMBED_MODEL"] = embed_model
    if embed_base_url:
        os.environ["EMBED_BASE_URL"] = embed_base_url


@click.group()
def cli() -> None:
    """Rosetta: ISA manual → Ghidra processor module generator."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("manual", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", required=True, help="Output docquery SQLite database path")
@click.option("--embed-model", default=None, help="Override EMBED_MODEL env var")
@click.option("--embed-base-url", default=None, help="Override EMBED_BASE_URL env var")
def ingest(manual: str, db: str, embed_model: str | None, embed_base_url: str | None) -> None:
    """Ingest a PDF manual into a docquery RAG database."""
    from docquery.config import Settings
    from docquery.ingestion.chunker import chunk
    from docquery.ingestion.pdf_loader import load
    from docquery.embeddings.provider import get_embeddings
    from docquery.storage.vector_store import VectorStore

    _apply_model_overrides(None, None, embed_model, embed_base_url)
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
@click.option("--concurrency", default=2, show_default=True, help="Parallel instruction extraction workers")
@click.option("--max-instructions", default=None, type=int, help="Cap number of instructions extracted (useful for quick tests)")
@click.option("--max-pcode", default=None, type=int, help="Limit P-code generation (pass 5) to first N instructions")
@click.option("--stop-after", default=None,
    type=click.Choice(["meta", "registers", "mnemonics", "instructions"]),
    help="Stop extraction after this pass and skip Ghidra file generation")
@click.option("--filter-mnemonics", default=None,
    help="Comma-separated glob patterns to keep after pass 3, e.g. 'MOV*,ADD,SUB*'")
@click.option("--append-slaspec", default=None, type=click.Path(),
    help="Append new instruction constructors to this .slaspec instead of generating a full module")
@click.option("--memory-warn-gb", default=2.0, show_default=True, help="Warn when free system RAM falls below this threshold (GB)")
@click.option("--llm-model", default=None, help="Override LLM_MODEL env var (e.g. llama3:8b)")
@click.option("--llm-base-url", default=None, help="Override LLM_BASE_URL env var")
@click.option("--embed-model", default=None, help="Override EMBED_MODEL env var")
@click.option("--embed-base-url", default=None, help="Override EMBED_BASE_URL env var")
def generate(
    db: str,
    name: str,
    out: str,
    spec_json: str | None,
    concurrency: int,
    max_instructions: int | None,
    max_pcode: int | None,
    stop_after: str | None,
    filter_mnemonics: str | None,
    append_slaspec: str | None,
    memory_warn_gb: float,
    llm_model: str | None,
    llm_base_url: str | None,
    embed_model: str | None,
    embed_base_url: str | None,
) -> None:
    """Extract ISA from database and generate a Ghidra processor module."""
    from rosetta.config import Settings
    from rosetta.extraction.isa_extractor import ISAExtractor
    from rosetta.generation.module_generator import ModuleGenerator

    _apply_model_overrides(llm_model, llm_base_url, embed_model, embed_base_url)
    settings = Settings()
    out_dir = Path(out)
    debug_dir = Path(db).with_suffix("").parent

    if spec_json:
        click.echo(f"Loading ISASpec from {spec_json}")
        extractor = ISAExtractor(db_path=db, settings=settings)
        spec = ISAExtractor.load(spec_json)
    else:
        click.echo(f"Extracting ISA from {db} ...")
        extractor = ISAExtractor(db_path=db, settings=settings)
        spec = extractor.extract(
            max_concurrent=concurrency,
            max_instructions=max_instructions,
            max_pcode=max_pcode,
            stop_after=stop_after,
            filter_mnemonics=filter_mnemonics,
            memory_warn_gb=memory_warn_gb,
            debug_save_dir=debug_dir,
            debug_prefix=name,
        )

        if stop_after:
            partial_path = debug_dir / f"{name}_partial_{stop_after}.json"
            extractor.save(spec, partial_path)
            click.echo(f"Stopped after '{stop_after}' → {partial_path}")
            if stop_after == "mnemonics":
                mn_path = debug_dir / f"{name}_debug_pass3_mnemonics.json"
                click.echo(f"Mnemonic list → {mn_path}")
            return

        cache = debug_dir / f"{name}_isa_spec.json"
        extractor.save(spec, cache)
        click.echo(f"ISASpec cached to {cache}")

    generator = ModuleGenerator()

    if append_slaspec:
        target = Path(append_slaspec)
        if not target.exists():
            click.echo(f"Error: --append-slaspec target not found: {target}", err=True)
            sys.exit(1)
        n = generator.append_to_slaspec(spec, target)
        click.echo(f"Appended {n} constructor(s) to {target}")
        return

    click.echo(f"Generating processor module '{name}' → {out_dir} ...")
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
@click.option("--embed-model", default=None, help="Override EMBED_MODEL env var")
@click.option("--embed-base-url", default=None, help="Override EMBED_BASE_URL env var")
def evaluate(module_dir: str, reference: str, embed_model: str | None, embed_base_url: str | None) -> None:
    """Semantic comparison of a generated spec against a Ghidra reference spec."""
    from rosetta.config import Settings
    from rosetta.evaluation.similarity import compare
    from rosetta.evaluation.spec_loader import load_ghidra_reference

    _apply_model_overrides(None, None, embed_model, embed_base_url)
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
@click.option("--target", default=None, help="Run only this manifest target ID (e.g. armv7)")
@click.option("--concurrency", default=1, show_default=True, help="Max concurrent LLM calls in Pass 4")
@click.option("--llm-model", default=None, help="Override LLM_MODEL env var (e.g. llama3:8b)")
@click.option("--llm-base-url", default=None, help="Override LLM_BASE_URL env var")
@click.option("--embed-model", default=None, help="Override EMBED_MODEL env var")
@click.option("--embed-base-url", default=None, help="Override EMBED_BASE_URL env var")
def batch(
    manifest: str,
    out: str,
    skip_extraction: bool,
    target: str | None,
    concurrency: int,
    llm_model: str | None,
    llm_base_url: str | None,
    embed_model: str | None,
    embed_base_url: str | None,
) -> None:
    """Run the full pipeline for every target in the manifest."""
    from rosetta.config import Settings
    from rosetta.evaluation.batch_eval import print_summary_table, run_batch

    _apply_model_overrides(llm_model, llm_base_url, embed_model, embed_base_url)
    settings = Settings()
    results = run_batch(
        manifest_path=Path(manifest),
        out_dir=Path(out),
        ghidra_home=settings.ghidra_home,
        settings=settings,
        skip_extraction=skip_extraction,
        target_filter=target,
        max_concurrent=concurrency,
    )
    print_summary_table(results)
    # Write JSON results
    results_file = Path(out) / "batch_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2))
    click.echo(f"Full results written to {results_file}")


# ---------------------------------------------------------------------------
# install  (copy generated module into Ghidra's Processors directory)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("module_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--force", is_flag=True, default=False, help="Overwrite if already installed")
def install(module_dir: str, force: bool) -> None:
    """Install a generated processor module into Ghidra's Processors directory."""
    import shutil
    from rosetta.config import Settings

    settings = Settings()
    src = Path(module_dir)
    proc_name = src.name
    dest = settings.ghidra_home / "Ghidra" / "Processors" / proc_name

    if dest.exists():
        if not force:
            click.echo(f"Already installed at {dest}. Use --force to overwrite.")
            return
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    click.echo(f"Installed {proc_name} → {dest}")

    # Verify the .sla file exists (sleigh must have been run first)
    lang_dir = dest / "data" / "languages"
    sla_files = list(lang_dir.glob("*.sla"))
    if not sla_files:
        click.echo(
            "WARNING: no .sla file found — run 'rosetta validate' first to compile the .slaspec.",
            err=True,
        )
    else:
        click.echo(f"  .sla file present: {sla_files[0].name}")


# ---------------------------------------------------------------------------
# load-test  (run Ghidra headless to verify the processor loads)
# ---------------------------------------------------------------------------


def _make_test_binary(path: Path, endian: str = "little") -> None:
    """Write a minimal flat ARM binary (8 bytes) for headless import."""
    import struct

    fmt = "<I" if endian == "little" else ">I"
    instructions = [
        struct.pack(fmt, 0xE3A00001),   # MOV R0, #1
        struct.pack(fmt, 0xE12FFF1E),   # BX LR
    ]
    path.write_bytes(b"".join(instructions))


@cli.command("load-test")
@click.argument("module_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--language-id",
    default=None,
    help="Ghidra language ID to use (auto-detected from .ldefs if omitted)",
)
@click.option(
    "--endian",
    default="little",
    type=click.Choice(["little", "big"]),
    help="Endianness for the test binary",
)
def load_test(module_dir: str, language_id: str | None, endian: str) -> None:
    """Import a test binary into Ghidra headless using the installed processor.

    Verifies that the generated processor module loads without errors.
    Run 'rosetta install' and 'rosetta validate' first.
    """
    import re
    import shutil
    import tempfile
    from rosetta.config import Settings
    from rosetta.validation.headless_runner import run_headless

    settings = Settings()
    mod_dir = Path(module_dir)

    # Auto-detect language ID from .ldefs
    if language_id is None:
        lang_dir = mod_dir / "data" / "languages"
        ldefs_files = list(lang_dir.glob("*.ldefs"))
        if not ldefs_files:
            click.echo("No .ldefs found — cannot determine language ID", err=True)
            sys.exit(1)
        ldefs_text = ldefs_files[0].read_text()
        m = re.search(r'id="([^"]+)"', ldefs_text)
        if not m:
            click.echo("Could not parse language ID from .ldefs", err=True)
            sys.exit(1)
        language_id = m.group(1)

    click.echo(f"Language ID : {language_id}")

    # Check the processor is installed in Ghidra
    proc_name = mod_dir.name
    installed = settings.ghidra_home / "Ghidra" / "Processors" / proc_name
    if not installed.exists():
        click.echo(
            f"Processor not installed — run 'rosetta install {module_dir}' first.", err=True
        )
        sys.exit(1)

    # Check .sla exists
    lang_dir_installed = installed / "data" / "languages"
    sla_files = list(lang_dir_installed.glob("*.sla"))
    if not sla_files:
        click.echo(
            "No .sla file found in installed module — run 'rosetta validate' first.", err=True
        )
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        # Write tiny test binary
        binary = Path(tmp) / "test.bin"
        _make_test_binary(binary, endian=endian)
        click.echo(f"Test binary : {binary} ({binary.stat().st_size} bytes)")

        # Run headless
        click.echo(f"Running Ghidra headless ...")
        result = run_headless(
            binary_path=binary,
            language_id=language_id,
            ghidra_home=settings.ghidra_home,
            project_dir=Path(tmp),
            project_name="rosetta_load_test",
        )

    if result.ok:
        click.echo("PASS — processor loaded successfully")
    else:
        click.echo("FAIL — Ghidra returned a non-zero exit code", err=True)
        # Surface the first relevant error line
        for line in (result.stdout + result.stderr).splitlines():
            if any(kw in line for kw in ("ERROR", "error", "Exception", "not found", "WARN")):
                click.echo(f"  {line}", err=True)
        sys.exit(1)


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
