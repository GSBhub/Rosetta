# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rosetta is an ISA manual → Ghidra processor module generator. It ingests a PDF manual into a RAG database (via `docquery`), runs a five-pass LLM extraction pipeline to produce a structured `ISASpec`, renders that spec into Ghidra SLEIGH files using Jinja2 templates, and optionally validates/evaluates the output against known Ghidra reference specs.

## Commands

### Install / Setup

```bash
uv pip install -e .          # install rosetta in editable mode
```

Environment is configured via `.env` at the repo root (not committed). Key variables:

```
EMBED_PROVIDER=ollama
EMBED_MODEL=embeddinggemma:latest
EMBED_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e2b
LLM_BASE_URL=http://localhost:11434
GHIDRA_HOME=/path/to/ghidra_12.1_PUBLIC
JAVA_HOME=/path/to/jdk
```

Ghidra and JDK can be placed under `tools/` (gitignored). The CLI auto-prepends `$JAVA_HOME/bin` to `PATH`.

### Run Tests

```bash
uv run pytest                     # all tests
uv run pytest tests/test_schemas.py   # single file
uv run pytest -k "test_generate"      # single test by name
```

Tests that require a live Ghidra installation are guarded by `requires_ghidra` (skip if `GHIDRA_HOME` is unset or missing).

### CLI Pipeline

```bash
# 1. Ingest a PDF manual into a docquery SQLite database
rosetta ingest manual.pdf --db dbs/myarch.db

# 2. Extract ISA + generate Ghidra processor module
rosetta generate --db dbs/myarch.db --name MyArch --out ./output

# 3. Validate: compile the generated .slaspec with the Ghidra SLEIGH compiler
rosetta validate ./output/MyArch

# 4. Evaluate: compare generated spec against a Ghidra reference
rosetta evaluate ./output/MyArch --reference ARM:LE:32:v7

# 5. Install into Ghidra's Processors directory
rosetta install ./output/MyArch [--force]

# 6. Load-test: import a test binary in headless Ghidra
rosetta load-test ./output/MyArch [--language-id ARM:LE:32:v7]

# 7. Batch: run full pipeline for all targets in a manifest
rosetta batch --manifest manifests/arm.yaml --out ./output [--skip-extraction]

# 8. Graph batch results or compare one spec vs all ARM variants
rosetta graph --results ./output/batch_results.json
rosetta graph --slaspec ./output/ARM_v7_generated/data/languages/ARM_v7_generated.slaspec
```

`--skip-extraction` re-uses cached `*_isa_spec.json` files for faster re-runs.

## Architecture

### Five-Pass Extraction (`src/rosetta/extraction/`)

`ISAExtractor.extract()` runs five sequential passes against a `docquery` VectorStore:

1. **Pass 1** – ISA metadata (`ISAMeta`): endianness, word size, alignment, instruction widths
2. **Pass 2** – Register file (`list[RegisterDef]`): names, aliases, sizes, roles
3. **Pass 3** – Mnemonic list: full set of instruction mnemonics
4. **Pass 4** – Per-instruction details (`InstructionDef`): encoding, bit fields, operands, semantics — run concurrently via `asyncio` with a configurable semaphore
5. **Pass 5** – P-code hints: LLM translates each instruction's natural-language semantics into a SLEIGH P-code statement

Each pass instantiates `docquery.pipeline.extractor.ExtractionPipeline` with a Pydantic output model; the pipeline handles RAG retrieval + structured JSON extraction.

The `ISASpec` (meta + registers + instructions) is serialized to `*_isa_spec.json` and can be loaded to skip extraction on subsequent runs.

### Schemas (`src/rosetta/extraction/schemas.py`)

Core Pydantic models: `ISAMeta`, `RegisterDef`, `InstructionDef`, `ISASpec`. These are the single source of truth for the data contract between extraction, generation, and evaluation.

### Generation (`src/rosetta/generation/`)

`ModuleGenerator.generate()` renders four Jinja2 templates from `templates/` into `<out_dir>/<processor_name>/data/languages/`:

- `*.slaspec` — SLEIGH disassembler/emulator spec (the main artifact)
- `*.pspec` — processor spec (PC, SP, context registers)
- `*.cspec` — compiler/calling-convention spec
- `*.ldefs` — language definitions (language ID, variant, compiler)

The template context includes auto-detected `pc_register` and `sp_register` resolved by matching common aliases.

### Validation (`src/rosetta/validation/`)

- `sleigh_compiler.py` — wraps `$GHIDRA_HOME/support/sleigh` subprocess to compile `.slaspec → .sla`
- `headless_runner.py` — wraps `analyzeHeadless` to import a tiny test binary and verify the processor loads

### Evaluation (`src/rosetta/evaluation/`)

- `similarity.py` — `SimilarityReport` with three metrics: semantic similarity (mean best-match cosine over chunked embeddings), instruction coverage (|generated ∩ reference| / |reference|), register overlap (Jaccard)
- `spec_loader.py` — regex-based mnemonic/register extraction from raw `.slaspec` text; also resolves Ghidra reference specs by processor name
- `batch_eval.py` — orchestrates the full generate → validate → evaluate pipeline for each target in a YAML manifest
- `graph.py` — matplotlib charts for batch results and single-spec vs all ARM variant comparisons

### Settings (`src/rosetta/config.py`)

`rosetta.config.Settings` extends `docquery.config.Settings` by adding the `ghidra_home` property (reads `GHIDRA_HOME` env var). Both the embedding model and LLM are configured entirely through env vars consumed by `docquery`.

### Batch Manifests (`manifests/`)

YAML files listing pipeline targets. Each entry has `id`, `name`, `manual`, `db`, `ghidra_reference_lang`, and `reference_slaspec`. Paths support `$ENV_VAR` expansion (evaluated at load time, not by the shell).

## Key Conventions

- `docquery` is a local git dependency (`git+https://github.com/gsbhub/docquery`); it provides embeddings, LLM access, chunking, PDF loading, and the `ExtractionPipeline`.
- All Ghidra/Java tool invocations happen via subprocess; `JAVA_HOME` must be set or `java` must be on `PATH`.
- `dbs/`, `output/`, `manuals/`, `tools/`, and `*_isa_spec.json` are all gitignored — they are runtime artifacts.
- Templates use `StrictUndefined` — missing template variables raise immediately rather than silently rendering empty.
