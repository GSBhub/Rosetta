# Rosetta

Rosetta ingests an ISA reference manual (PDF) into a ChromaDB vector store, runs a LangGraph extraction pipeline to produce a structured `ISASpec`, renders that spec into Ghidra SLEIGH processor module files, and evaluates the output against Ghidra's built-in reference specs.

```
PDF manual
    │  rosetta ingest
    ▼
ChromaDB vector store
    │  rosetta generate
    ▼  (LangGraph StateGraph — meta/registers/mnemonics in parallel)
ISASpec JSON  ──────────────────────────────────────────────────────────────┐
    │                                                                        │
    ▼                                                                        │
Ghidra processor module (.slaspec / .pspec / .cspec / .ldefs)               │
    │  rosetta validate                                                      │
    ▼                                                                        │
Compiled .sla                                                               │
    │  rosetta evaluate                                                      │
    ▼                                                                        │
Similarity report (coverage · register overlap) ◄───────────────────────────┘
```

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd rosetta
uv sync      # installs all workspace packages + docquery from git
```

This is a **uv workspace** — `uv sync` resolves all packages under `packages/` together with a single lockfile. `docquery` is pulled from GitHub automatically.

## Environment

Create a `.env` file at the repo root (not committed):

```
# Embedding model (Ollama)
EMBED_PROVIDER=ollama
EMBED_MODEL=embeddinggemma:latest
EMBED_BASE_URL=http://localhost:11434

# LLM (Ollama)
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e2b
LLM_BASE_URL=http://localhost:11434

# Ghidra (for validate / evaluate / install / load-test)
GHIDRA_HOME=/path/to/ghidra_12.1_PUBLIC
JAVA_HOME=/path/to/jdk
```

Ghidra and the JDK can be placed under `tools/` (gitignored). The CLI auto-prepends `$JAVA_HOME/bin` to `PATH` so subprocesses find `java`.

## CLI

### `rosetta ingest` — PDF → ChromaDB vector store

```bash
rosetta ingest manuals/armv7_ref.pdf --db dbs/armv7
```

`--db` is a **directory** (ChromaDB). Defaults to `CHROMA_DB_PATH` env var if unset. Loads the PDF with `docquery`'s table-aware loader, chunks it, embeds the chunks, and writes to a ChromaDB persistent store. Re-ingesting the same file adds 0 new chunks (content-hash deduplication).

Use `--source` to ingest source code instead of a PDF — pass a directory and all `.c`, `.h`, `.cpp`, `.py` files are loaded as text chunks. Run `ingest` multiple times against the same `--db` to supplement an existing store with additional manuals.

### `rosetta generate` — extract ISA + generate processor module

```bash
rosetta generate --db dbs/armv7 --name ARM_v7_generated --out ./output
```

Invokes the compiled LangGraph pipeline: after ingestion, **meta / registers / mnemonics** run in parallel, then fan in to instruction extraction, P-code generation, and SLEIGH rendering. Serializes the `ISASpec` to `*_isa_spec.json` alongside the output files.

Options:
- `--spec-json PATH` — skip extraction, load a cached `isa_spec.json` directly
- `--concurrency N` — max concurrent LLM calls during instruction extraction (default 2; use 1 on RAM-constrained systems)
- `--max-instructions N` — cap extraction at N instructions (useful for smoke tests)
- `--stop-after STAGE` — stop early after `meta`, `registers`, `mnemonics`, `stubs`, or `instructions`
- `--filter-mnemonics GLOBS` — comma-separated glob patterns to keep after mnemonic discovery, e.g. `MOV*,ADD,SUB*`
- `--resume` — resume pass 4 from a partial JSONL save (skips already-extracted mnemonics)
- `--append-slaspec PATH` — append new instruction constructors to an existing `.slaspec` instead of generating a full module
- `--inter-chunk-sleep N` — seconds to sleep between pass-4 chunks (use 2.0 for local Ollama to allow KV-cache GC)

### `rosetta validate` — compile `.slaspec` with Ghidra's SLEIGH compiler

```bash
rosetta validate ./output/ARM_v7_generated
```

Runs `$GHIDRA_HOME/support/sleigh` to compile the `.slaspec` to `.sla`. Exits non-zero on errors.

### `rosetta evaluate` — similarity report vs a Ghidra reference

```bash
rosetta evaluate ./output/ARM_v7_generated --reference ARM:LE:32:v7
```

Computes two structural metrics against the corresponding Ghidra built-in spec:

| Metric | Description |
|--------|-------------|
| **Instruction coverage** | `|generated ∩ reference| / |reference|` |
| **Register overlap** | Jaccard coefficient on register name sets |

Both sides may be a single `.slaspec`, a `languages/` directory, a Ghidra language ID, or a bare processor name (e.g. `ARM` — unions all variants).

### `rosetta run-stage` — checkpointed single-stage execution

```bash
rosetta run-stage ingest --db dbs/armv7 --name ARM_v7 --checkpoint state/arm.json --source manuals/armv7.pdf
rosetta run-stage meta    --name ARM_v7 --checkpoint state/arm.json
rosetta run-stage all     --name ARM_v7 --checkpoint state/arm.json
```

Runs one pipeline stage at a time (or `all` to run the full sequence), persisting `PipelineState` to a checkpoint JSON between stages. Each invocation reads the previous stage's output from the checkpoint and adds its own. Per-stage snapshot files (`<name>.<stage>.json`) are written alongside the checkpoint for inspection.

Valid stages: `ingest meta registers mnemonics instructions pcode generate validate evaluate`

Options:
- `--checkpoint PATH` — checkpoint file (required); created on first run
- `--source PATH` — PDF or source directory (ingest stage only)
- `--reference PATH` — reference `.slaspec` or language ID (evaluate stage)
- `--max-instructions N` — cap instruction count (instructions stage)
- `--inter-chunk-sleep N` — sleep between instruction chunks (default 2.0; helps Ollama KV GC)

### `rosetta install` — copy module into Ghidra

```bash
rosetta install ./output/ARM_v7_generated [--force]
```

### `rosetta load-test` — headless Ghidra import test

```bash
rosetta load-test ./output/ARM_v7_generated [--language-id ARM:LE:32:v7]
```

Writes a minimal test binary and imports it into Ghidra headless to verify the processor module loads without errors.

### `rosetta graph` — compare generated spec against all Ghidra ARM variants

```bash
rosetta graph ./output/ARM_v7_generated/data/languages/ARM_v7_generated.slaspec
rosetta graph ./output/ARM_v7_generated/data/languages/ARM_v7_generated.slaspec --out report.png --no-display
```

Takes a single `.slaspec` as a positional argument and compares it against every Ghidra ARM/AARCH64 variant. Use `--embeddings` to include semantic similarity (requires Ollama). Use `--no-display` in headless/SSH environments.

## Pipeline Architecture

`rosetta generate` compiles and invokes a `StateGraph[PipelineState]` (see `src/rosetta/graph.py`).
The pipeline is split across nine uv packages under `packages/`:

```
START → ingest → ┌─ meta ──────────┐
                 ├─ registers ─────┤→ instructions → pcode → generate_sla → validate_sla → evaluate_sla → END
                 └─ mnemonics ─────┘
                   (parallel)         (fan-in barrier)
```

| Node | Package | Output |
|------|---------|--------|
| `ingest` | `rosetta-ingest` | ChromaDB populated |
| `meta` | `rosetta-meta` | `ISAMeta` — endianness, word size, alignment, instruction widths |
| `registers` | `rosetta-registers` | `list[RegisterDef]` — names, aliases, sizes, roles |
| `mnemonics` | `rosetta-mnemonics` | `list[str]` — full mnemonic list via inner LangGraph multi-strategy loop |
| `instructions` | `rosetta-instructions` | `list[InstructionDef]` — encoding, bit fields, operands, semantics (async concurrent) |
| `pcode` | `rosetta-pcode` | `pcode_hint` on each instruction — direct LLM call |
| `generate_sla` | `rosetta-generate-sla` | `.slaspec`, `.pspec`, `.cspec`, `.ldefs` via Jinja2 |
| `validate_sla` | `rosetta-validate-sla` | `compile_ok`, `compile_errors` — Ghidra sleigh subprocess |
| `evaluate_sla` | `rosetta-evaluate-sla` | `instruction_coverage`, `register_overlap` |

The mnemonic node uses a LangGraph inner graph with 10 query strategies (data-processing, memory, branch, multiply, alpha A–F, alpha G–N, alpha O–Z, SIMD/VFP, system/coprocessor) to find the full instruction set — a single-query approach typically finds only ~10% of ARM instructions.

The `ISASpec` (meta + registers + instructions) is serialised to `*_isa_spec.json` and can be reloaded to skip re-extraction on subsequent runs.

## Data Models

```python
class ISAMeta:      name, endian, word_size_bits, alignment, instruction_sizes_bits
class RegisterDef:  name, aliases, size_bits, description
class InstructionDef: mnemonic, variants, encoding_bits, bit_fields,
                      bit_constraints, operands, semantics, pcode_hint
class ISASpec:      meta, registers, instructions
```

## Generated Files

For a processor named `ARM_v7_generated`, rosetta writes:

```
output/
└── ARM_v7_generated/
    └── data/
        └── languages/
            ├── ARM_v7_generated.slaspec   # SLEIGH disassembler/emulator spec
            ├── ARM_v7_generated.pspec     # Processor spec (PC, SP, context regs)
            ├── ARM_v7_generated.cspec     # Compiler / calling-convention spec
            └── ARM_v7_generated.ldefs     # Language definitions (ID, variant, compiler)
```

`dbs/`, `output/`, `manuals/`, `tools/`, and `*_isa_spec.json` are all gitignored — they are runtime artifacts.

## Development

```bash
uv run pytest tests/     # root package tests
uv run pytest packages/  # all workspace node package tests (all mocked)
uv run pytest tests/ packages/          # everything (86 tests)
uv run pytest -k "test_generate"        # by name
uv run pytest packages/rosetta-mnemonics/tests/ -v  # single package
```

Tests that require a live Ghidra installation are guarded by `requires_ghidra` and skipped if `GHIDRA_HOME` is unset. No other external services (Ollama, ChromaDB) are required — all are mocked.

See `docs/data-flow.md` for a full diagram of node connections and `PipelineState` key contracts.
