# Rosetta

Rosetta ingests an ISA reference manual (PDF) into a ChromaDB vector store, runs a LangGraph extraction pipeline to produce a structured `ISASpec`, renders that spec into Ghidra SLEIGH processor module files, and evaluates the output against Ghidra's built-in reference specs.

```
PDF manual
    │  rosetta ingest
    ▼
ChromaDB vector store
    │  rosetta generate (or rosetta batch)
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
Similarity report (semantic · coverage · register overlap) ◄────────────────┘
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

`--db` is a **directory** (ChromaDB). Loads the PDF with `docquery`'s table-aware loader, chunks it, embeds the chunks, and writes to a ChromaDB persistent store. Re-ingesting the same file adds 0 new chunks (content-hash deduplication).

### `rosetta generate` — extract ISA + generate processor module

```bash
rosetta generate --db dbs/armv7 --name ARM_v7_generated --out ./output
```

Invokes the compiled LangGraph pipeline: after ingestion, **meta / registers / mnemonics** run in parallel, then fan in to instruction extraction, P-code generation, and SLEIGH rendering. Serializes the `ISASpec` to `*_isa_spec.json` alongside the output files.

Options:
- `--spec-json PATH` — skip extraction, load a cached `isa_spec.json` directly
- `--concurrency N` — max concurrent LLM calls during instruction extraction (default 4; use 1 on RAM-constrained systems)
- `--max-instructions N` — cap extraction at N instructions (useful for smoke tests)
- `--stop-after STAGE` — stop early after `meta`, `registers`, `mnemonics`, `stubs`, or `instructions`

### `rosetta validate` — compile `.slaspec` with Ghidra's SLEIGH compiler

```bash
rosetta validate ./output/ARM_v7_generated
```

Runs `$GHIDRA_HOME/support/sleigh` to compile the `.slaspec` to `.sla`. Exits non-zero on errors.

### `rosetta evaluate` — similarity report vs a Ghidra reference

```bash
rosetta evaluate ./output/ARM_v7_generated --reference ARM:LE:32:v7
```

Computes three metrics against the corresponding Ghidra built-in spec:

| Metric | Description |
|--------|-------------|
| **Semantic similarity** | Mean best-match cosine over chunked embeddings |
| **Instruction coverage** | `|generated ∩ reference| / |reference|` |
| **Register overlap** | Jaccard coefficient on register name sets |

### `rosetta batch` — full pipeline for all targets in a manifest

```bash
rosetta batch --manifest manifests/arm.yaml --out ./output
```

Runs ingest → extract → generate → validate → evaluate for every target listed in the manifest, then prints a summary table and writes `output/batch_results.json`.

Options:
- `--target ID` — run only the manifest target with this ID (e.g. `armv7`)
- `--concurrency N` — Pass 4 LLM concurrency per target (default 1)
- `--skip-extraction` — reuse cached `*_isa_spec.json` files (skip re-extraction)

Example — run targets one at a time to limit RAM usage:

```bash
rosetta batch --manifest manifests/arm.yaml --target armv7 --out ./output --concurrency 1
rosetta batch --manifest manifests/arm.yaml --target armv6 --out ./output --concurrency 1
rosetta batch --manifest manifests/arm.yaml --target armv8_32 --out ./output --concurrency 1
```

### `rosetta install` — copy module into Ghidra

```bash
rosetta install ./output/ARM_v7_generated [--force]
```

### `rosetta load-test` — headless Ghidra import test

```bash
rosetta load-test ./output/ARM_v7_generated [--language-id ARM:LE:32:v7]
```

Writes a minimal test binary and imports it into Ghidra headless to verify the processor module loads without errors.

### `rosetta graph` — visualise batch results

```bash
# Graph a batch_results.json
rosetta graph --results ./output/batch_results.json

# Compare one generated .slaspec against all Ghidra ARM variants
rosetta graph --slaspec ./output/ARM_v7_generated/data/languages/ARM_v7_generated.slaspec
```

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
| `evaluate_sla` | `rosetta-evaluate-sla` | `semantic_similarity`, `instruction_coverage`, `register_overlap` |

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

## Batch Manifests

YAML files under `manifests/` list pipeline targets. Paths support `$ENV_VAR` expansion:

```yaml
targets:
  - id: armv7
    name: ARM_v7_generated
    manual: manuals/armv7_ref.pdf
    db: dbs/armv7        # ChromaDB directory (no .db extension)
    ghidra_reference_lang: ARM:LE:32:v7
    reference_slaspec: $GHIDRA_HOME/Ghidra/Processors/ARM/data/languages/ARM7_le.slaspec
```

`dbs/`, `output/`, `manuals/`, `tools/`, and `*_isa_spec.json` are all gitignored — they are runtime artifacts.

## Development

```bash
uv run pytest tests/     # root package tests (59 tests, no external services)
uv run pytest packages/  # all workspace node package tests (100 tests, all mocked)
uv run pytest tests/ packages/          # everything
uv run pytest -k "test_generate"        # by name
uv run pytest packages/rosetta-mnemonics/tests/ -v  # single package
```

Tests that require a live Ghidra installation are guarded by `requires_ghidra` and skipped if `GHIDRA_HOME` is unset. No other external services (Ollama, ChromaDB) are required — all are mocked.

See `docs/data-flow.md` for a full diagram of node connections and `PipelineState` key contracts.
