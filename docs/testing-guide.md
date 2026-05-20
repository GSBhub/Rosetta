# Rosetta User Testing Guide

Step-by-step guide to verifying every stage of the pipeline from PDF ingestion through Ghidra module generation.

---

## Prerequisites

### Required for all tests
- Python 3.12+, `uv` installed
- Ollama running locally at `http://localhost:11434` with:
  - An embedding model: `ollama pull embeddinggemma:latest` (or set `EMBED_MODEL` to another)
  - *(For local LLM only)* An LLM model: `ollama pull gemma4:e2b`
- Or: an Anthropic API key in `.env` (`LLM_PROVIDER=anthropic`, `LLM_API_KEY=sk-ant-…`)

### Required for validate / install / load-test
- Ghidra installation at `GHIDRA_HOME` (e.g. `tools/ghidra_12.1_PUBLIC`)
- JDK 21+ at `JAVA_HOME` (e.g. `tools/jdk-21.0.7+6`)

### Install
```bash
uv sync          # installs all workspace packages + docquery from git
```

### Confirm `.env` is populated
```bash
cat .env
# Should show EMBED_MODEL, LLM_PROVIDER, LLM_API_KEY (if using Claude), GHIDRA_HOME, JAVA_HOME
```

---

## Stage 0 — Unit Tests

Run the full test suite before doing anything else. No live Ollama, Ghidra, or ChromaDB instance is required — all external services are mocked.

```bash
# Root package tests (legacy ISAExtractor, batch eval, module generator, …)
uv run pytest tests/ -v

# All workspace packages (LangGraph nodes, schemas, utils, …)
uv run pytest packages/ -v

# Everything at once
uv run pytest tests/ packages/ -v
```

Expected: all tests pass (`PASSED`). Ghidra-dependent tests in `tests/` are auto-skipped when `GHIDRA_HOME` is unset.

To run a specific package:
```bash
uv run pytest packages/rosetta-mnemonics/tests/ -v
```

To run a single test by name:
```bash
uv run pytest -k "test_mnemonics_node_filter"
```

### What each package tests

| Package | Tests | Mocks |
|---------|-------|-------|
| `rosetta-schemas` | Pydantic round-trip, `PipelineState` accessors | none |
| `rosetta-utils` | `check_memory_headroom`, `log_memory` | `psutil.virtual_memory` |
| `rosetta-ingest` | success path, missing db_path, ingest errors | `docquery.ingest` |
| `rosetta-meta` | success, fallback on bad LLM response | `docquery.query` |
| `rosetta-registers` | success, empty result | `docquery.query` |
| `rosetta-mnemonics` | discovery deduplication, strategy exhaustion, filter globs, `mnemonics_node` | `ExtractionPipeline.run`, `_build_chroma` |
| `rosetta-instructions` | concurrency, `max_instructions` cap, `stop_after`, `resume` | `extract_instruction_async`, `_build_chroma` |
| `rosetta-pcode` | hint generation, skip existing, `max_pcode` limit | `generate_pcode` |
| `rosetta-generate-sla` | all 4 output files written, mnemonic/register content | none (real Jinja2 render) |
| `rosetta-validate-sla` | success/failure compile paths, error propagation | `compile_slaspec`, `subprocess.run` |
| `rosetta-evaluate-sla` | similarity metrics, cosine math, coverage formula | `similarity.compare` |

---

## Stage 1 — PDF Ingestion

Ingest a PDF manual into a ChromaDB vector store. Any ISA reference manual works.

```bash
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test
```

`--db` points to a **directory** (ChromaDB), not a single file.

**What to watch for:**
- Log line: `Ingested N items into dbs/arm_test`
- No exceptions

**Verify the database was created:**
```bash
ls dbs/arm_test/
# Expected: chroma.sqlite3  and one or more UUID subdirectories (segment data)

python3 -c "
import chromadb
client = chromadb.PersistentClient('dbs/arm_test')
col = client.get_collection('documents')
print('Chunks in DB:', col.count())
"
```

Expected: `Chunks in DB: N` where N is typically 2 000–15 000 for a full ISA manual.

**Re-ingest idempotency:** Running `ingest` again on the same PDF should add 0 new chunks (docquery deduplicates by content hash):

```bash
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test
# Expected: "Ingested 0 items into dbs/arm_test"  (or similar skip message)
```

---

## Stage 2 — ISA Extraction (Full Pipeline)

The `rosetta generate` command invokes the compiled LangGraph pipeline. After ingestion,
meta / registers / mnemonics run in **parallel**, then fan in to instruction extraction.

```bash
rosetta generate \
  --db dbs/arm_test \
  --name ARM_test \
  --out ./output \
  --concurrency 2
```

**Expected log sequence:**
```
INFO  [ingest node skipped — db already exists]
INFO  [meta] extracting ISA metadata
INFO  [registers] extracting register file
INFO  [mnemonics] starting multi-strategy discovery
INFO  [mnemonics] strategy 1/10: list ALL mnemonics
...
INFO  [mnemonics] found N unique mnemonics after 10 strategies
INFO  [instructions] starting pass (concurrency=2)
INFO  [instructions] chunk 1–20 / N remaining
...
INFO  [pcode] generating P-code hints
INFO  [generate_sla] rendered 4 SLEIGH files → output/ARM_test/data/languages
```

**Spot-check the generated spec:**
```bash
python3 -c "
import json
spec = json.load(open('dbs/ARM_test_isa_spec.json'))
print('name      :', spec['meta']['name'])
print('endian    :', spec['meta']['endian'])
print('word bits :', spec['meta']['word_size_bits'])
print('registers :', len(spec['registers']))
print('instructions:', len(spec['instructions']))
i = spec['instructions'][0]
print('first instr:', i['mnemonic'], '|', i['semantics'][:60])
print('pcode_hint :', i['pcode_hint'])
"
```

Expected: `registers` ≥ 10, `instructions` ≥ 30, `pcode_hint` is a non-empty SLEIGH expression (not `# TODO:`).

**Stop after a specific node (for debugging):**
```bash
# Only run meta/registers/mnemonics fan-out; skip instruction extraction
rosetta generate --db dbs/arm_test --name ARM_test --out ./output --stop-after mnemonics

# Only run up to instruction extraction; skip P-code + generation
rosetta generate --db dbs/arm_test --name ARM_test --out ./output --stop-after instructions
```

Valid `--stop-after` values: `meta`, `registers`, `mnemonics`, `stubs`, `instructions`.

**Limit instructions for a quick smoke test:**
```bash
rosetta generate \
  --db dbs/arm_test \
  --name ARM_smoke \
  --out ./output \
  --max-instructions 5
```

Runs in under a minute; useful for verifying the LLM connection before a full run.

**Skip extraction using a cached spec:**
```bash
rosetta generate \
  --db dbs/arm_test \
  --name ARM_test \
  --out ./output \
  --spec-json dbs/ARM_test_isa_spec.json
```

Runs in seconds (no LLM calls); output files are re-rendered from the cached spec.

---

## Stage 3 — Validate Generated SLEIGH

Compile the `.slaspec` with Ghidra's SLEIGH compiler. Requires `GHIDRA_HOME` and `JAVA_HOME`.

```bash
rosetta validate ./output/ARM_test
```

**Expected:** `ARM_test.slaspec: OK`

If you see `FAILED`, the compiler lists specific line numbers and error messages. Common causes:
- Missing P-code operand types — adjust templates or edit the `.slaspec` directly
- Duplicate `define` statements — the LLM extracted duplicate registers

**Verify the `.sla` binary was produced:**
```bash
ls output/ARM_test/data/languages/*.sla
```

---

## Stage 4 — Install into Ghidra

```bash
rosetta install ./output/ARM_test
```

Expected: `Installed ARM_test → <GHIDRA_HOME>/Ghidra/Processors/ARM_test`

Use `--force` to overwrite a previous installation:
```bash
rosetta install ./output/ARM_test --force
```

---

## Stage 5 — Headless Load Test

Imports a tiny test binary into Ghidra headless to verify the processor module loads.

```bash
rosetta load-test ./output/ARM_test --language-id ARM_test:LE:32:default
```

**Expected:** `PASS — processor loaded successfully`

If it fails, the command prints the first `ERROR` / `Exception` lines from Ghidra's stdout. Common causes:
- `.sla` not present (run `rosetta validate` first)
- Module not installed (run `rosetta install` first)
- Language ID mismatch — inspect the generated `.ldefs` file:
  ```bash
  grep 'id=' output/ARM_test/data/languages/*.ldefs
  ```

---

## Stage 6 — Evaluate Against a Ghidra Reference

Compare the generated spec to a Ghidra built-in reference processor.

```bash
rosetta evaluate ./output/ARM_test --reference ARM:LE:32:v7
```

**Expected output (example):**
```
Semantic similarity : 0.72
Instruction coverage: 0.58   (58% of ARM v7 mnemonics present)
Register overlap    : 0.81   (Jaccard)
```

Instruction coverage below 0.5 suggests mnemonic discovery (Node 3) missed a large portion of the ISA — consider re-running with a more comprehensive manual or increasing `TOP_K`.

---

## Stage 7 — Batch Pipeline

Run the full pipeline for multiple ISA targets defined in a YAML manifest.

```bash
rosetta batch --manifest manifests/arm.yaml --out ./output
```

Append `--skip-extraction` to re-use cached `*_isa_spec.json` files (skips LLM calls):
```bash
rosetta batch --manifest manifests/arm.yaml --out ./output --skip-extraction
```

Results are written to `./output/batch_results.json`.

---

## Stage 8 — Graph Results

Visualise batch results or compare a generated spec against all Ghidra ARM variants.

```bash
# Graph batch results
rosetta graph --results ./output/batch_results.json --out ./output/report.png --no-display

# Compare one spec against all Ghidra ARM/AARCH64 variants
rosetta graph --slaspec ./output/ARM_test/data/languages/ARM_test.slaspec \
              --out ./output/arm_variants.png --no-display
```

Use `--no-display` in headless/SSH environments. Omit `--out` to open an interactive matplotlib window.

---

## Troubleshooting

### Ollama connection errors
```
ConnectionRefusedError: [Errno 111] Connection refused
```
Ollama is not running. Start it:
```bash
ollama serve
```

### Anthropic API key not set
```
AuthenticationError: No API key provided
```
Check `.env` has `LLM_API_KEY=sk-ant-…` and `LLM_PROVIDER=anthropic`.

### LangGraph structured output parse error
```
ValueError: Extraction failed after 3 retries.
```
The LLM is not producing valid JSON matching the Pydantic schema. Try:
1. Switching to a more capable model (e.g. `claude-sonnet-4-6` → `claude-opus-4-7`)
2. Increasing `MAX_RETRIES=5` in `.env`
3. Lowering `TEMPERATURE=0` (should already be 0)

### ChromaDB directory missing or empty
```
chromadb.errors.NotFoundError: Collection 'documents' does not exist.
```
The database was not ingested or the `--db` path is wrong. Re-run:
```bash
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test
```

### ChromaDB embedding model mismatch
If you change `EMBED_MODEL` after ingesting, similarity search will silently return bad results because the stored and query embeddings have different semantic spaces. Delete and re-ingest:
```bash
rm -rf dbs/arm_test
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test
```

### GHIDRA_HOME not set
```
RuntimeError: GHIDRA_HOME is not set.
```
Add `GHIDRA_HOME=/path/to/ghidra_12.1_PUBLIC` to `.env` and ensure the path exists.

### Out of memory during instruction extraction (Node 4)
Node 4 runs concurrent LLM calls. Reduce concurrency:
```bash
rosetta generate --db dbs/arm_test --name ARM_test --out ./output --concurrency 1
```

---

## Quick Reference — All CLI Commands

```bash
rosetta ingest  <manual.pdf>  --db <dir>
rosetta generate              --db <dir> --name <Name> --out <dir>
                              [--spec-json <cache.json>]
                              [--concurrency N]
                              [--max-instructions N]
                              [--stop-after meta|registers|mnemonics|stubs|instructions]
rosetta validate  <module_dir>
rosetta install   <module_dir>  [--force]
rosetta load-test <module_dir>  [--language-id ID]
rosetta evaluate  <module_dir>  --reference <LANG_ID or .slaspec path>
rosetta batch   --manifest <file.yaml> --out <dir>  [--skip-extraction]
rosetta graph   --results <batch_results.json>  [--out img.png] [--no-display]
rosetta graph   --slaspec  <generated.slaspec>  [--out img.png] [--no-display]
```
