# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rosetta is an ISA manual → Ghidra processor module generator. It ingests a PDF manual into a ChromaDB vector store (via `docquery`), runs a LangGraph extraction pipeline to produce a structured `ISASpec`, renders that spec into Ghidra SLEIGH files using Jinja2 templates, and optionally validates/evaluates the output against known Ghidra reference specs.

## Commands

### Install / Setup

```bash
uv sync          # install all workspace packages + docquery from git
```

Environment is configured via `.env` at the repo root (not committed). Key variables:

```
EMBED_PROVIDER=ollama
EMBED_MODEL=embeddinggemma:latest
EMBED_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e2b
LLM_BASE_URL=http://localhost:11434
CHROMA_DB_PATH=dbs/myarch    # default --db for ingest/generate/run-stage
GHIDRA_HOME=/path/to/ghidra_12.1_PUBLIC
JAVA_HOME=/path/to/jdk
```

Ghidra and JDK can be placed under `tools/` (gitignored). The CLI auto-prepends `$JAVA_HOME/bin` to `PATH`.

### Run Tests

```bash
uv run pytest                          # all tests (86 total)
uv run pytest tests/                   # root package only
uv run pytest packages/                # workspace node packages only
uv run pytest -k "test_generate"       # single test by name
```

Tests that require a live Ghidra installation are guarded by `requires_ghidra` (skip if `GHIDRA_HOME` is unset or missing). No other external services (Ollama, ChromaDB) are required — all are mocked.

### CLI Pipeline

```bash
# 1. Ingest a PDF manual into a ChromaDB vector store
rosetta ingest manual.pdf --db dbs/myarch

# 2. Extract ISA + generate Ghidra processor module (full LangGraph pipeline)
rosetta generate --db dbs/myarch --name MyArch --out ./output

# 2b. Run one stage at a time (checkpointed, for debugging or large ISAs)
rosetta run-stage ingest --db dbs/myarch --name MyArch --checkpoint state/myarch.json --source manual.pdf
rosetta run-stage meta   --name MyArch --checkpoint state/myarch.json
rosetta run-stage all    --name MyArch --checkpoint state/myarch.json --out ./output

# 3. Validate: compile the generated .slaspec with the Ghidra SLEIGH compiler
rosetta validate ./output/MyArch

# 4. Evaluate: compare generated spec against a Ghidra reference
rosetta evaluate ./output/MyArch --reference ARM:LE:32:v7

# 5. Install into Ghidra's Processors directory
rosetta install ./output/MyArch [--force]

# 6. Load-test: import a test binary in headless Ghidra
rosetta load-test ./output/MyArch [--language-id ARM:LE:32:v7]

# 7. Graph: compare generated .slaspec against all Ghidra ARM variants
rosetta graph ./output/MyArch/data/languages/MyArch.slaspec [--out report.png] [--no-display]
```

## Architecture

### LangGraph Pipeline (`src/rosetta/graph.py`)

`build_compiled_graph()` returns a `StateGraph[PipelineState]` split across nine uv packages under `packages/`:

```
START → ingest → ┌─ meta ──────────┐
                 ├─ registers ─────┤→ instructions → pcode → generate_sla → validate_sla → evaluate_sla → END
                 └─ mnemonics ─────┘
                   (parallel)         (fan-in barrier)
```

Each node is in its own package (`rosetta-meta`, `rosetta-registers`, etc.) with its own tests.

### Schemas (`packages/rosetta-schemas/`)

Core Pydantic models: `ISAMeta`, `RegisterDef`, `InstructionDef`, `ISASpec`, `PipelineState`. Single source of truth for the data contract between all nodes.

### Generation (`packages/rosetta-generate-sla/`)

`ModuleGenerator.generate()` renders four Jinja2 templates into `<out_dir>/<processor_name>/data/languages/`:

- `*.slaspec` — SLEIGH disassembler/emulator spec (the main artifact)
- `*.pspec` — processor spec (PC, SP, context registers)
- `*.cspec` — compiler/calling-convention spec
- `*.ldefs` — language definitions (language ID, variant, compiler)

Templates use `StrictUndefined` — missing variables raise immediately.

### Validation (`packages/rosetta-validate-sla/`)

- `sleigh_compiler.py` — wraps `$GHIDRA_HOME/support/sleigh` subprocess to compile `.slaspec → .sla`
- `headless_runner.py` — wraps `analyzeHeadless` to import a tiny test binary and verify the processor loads

### Evaluation (`packages/rosetta-evaluate-sla/` and `src/rosetta/evaluation/`)

- `similarity.py` — `SimilarityReport` with two metrics: instruction coverage (`|generated ∩ reference| / |reference|`) and register overlap (Jaccard). Both sides may be a file, a `languages/` directory, a Ghidra language ID, or a bare processor name.
- `spec_loader.py` — regex-based mnemonic/register extraction from `.slaspec` text; resolves Ghidra reference specs by processor name or language ID
- `graph.py` — matplotlib chart comparing a generated `.slaspec` against all Ghidra ARM/AARCH64 variants

### Stage Runner (`src/rosetta/stage_runner.py`)

`run_stage(state, stage_name)` executes one node synchronously against a `PipelineState` dict and returns the updated state. Used by `rosetta run-stage` to support checkpoint-based incremental execution and per-stage inspection.

### Settings (`src/rosetta/config.py`)

`rosetta.config.Settings` extends `docquery.config.Settings` by adding the `ghidra_home` property (reads `GHIDRA_HOME` env var). All embedding and LLM config comes from env vars consumed by `docquery`.

## Key Conventions

- `docquery` is a git dependency (`git+https://github.com/gsbhub/docquery`); it provides embeddings, LLM access, chunking, PDF loading, and `ExtractionPipeline`.
- All Ghidra/Java tool invocations happen via subprocess; `JAVA_HOME` must be set or `java` must be on `PATH`.
- `dbs/`, `output/`, `manuals/`, `tools/`, and `*_isa_spec.json` are all gitignored — they are runtime artifacts.
- `CHROMA_DB_PATH` env var sets the default `--db` for `ingest`, `generate`, and `run-stage` so it doesn't have to be repeated on every invocation.
