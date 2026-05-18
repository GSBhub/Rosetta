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
uv pip install -e .
```

### Confirm `.env` is populated
```bash
cat .env
# Should show EMBED_MODEL, LLM_PROVIDER, LLM_API_KEY (if using Claude), GHIDRA_HOME, JAVA_HOME
```

---

## Stage 0 — Unit Tests

Run the test suite before doing anything else. All tests should pass without a live Ollama or Ghidra instance (Ghidra-dependent tests are auto-skipped when `GHIDRA_HOME` is unset or the path does not exist).

```bash
uv run pytest
```

Expected output: all tests pass or skip (`PASSED` / `SKIPPED`), no `FAILED`.

To run a specific test file:
```bash
uv run pytest tests/test_schemas.py -v
```

---

## Stage 1 — PDF Ingestion

Ingest a PDF manual into a SQLite RAG database. Any ISA reference manual works; the ARM Architecture Reference Manual is a good large-scale test.

```bash
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test.db
```

**What to watch for:**
- Progress bars for page loading and embedding batches
- Log line: `Loaded N pages from ... (M via OCR, K with tables)` — M > 0 means OCR fired on scanned pages; K > 0 means tables were extracted
- Log line: `Inserted N chunks, skipped M duplicates`
- No exceptions

**Verify the database was created:**
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('dbs/arm_test.db')
n_chunks = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
n_emb    = conn.execute('SELECT COUNT(*) FROM chunk_embeddings').fetchone()[0]
print(f'chunks: {n_chunks}  embeddings: {n_emb}')
assert n_chunks == n_emb, 'chunk/embedding count mismatch'
conn.close()
"
```

Expected: chunk count matches embedding count; typically 2 000–15 000 chunks for a full ISA manual.

**Re-ingest idempotency:** Running `ingest` a second time on the same PDF should report `Inserted 0 chunks, skipped N duplicates` — confirming deduplication by SHA-256 hash.

```bash
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test.db
# Expected: "Inserted 0 chunks, skipped N duplicates"
```

---

## Stage 2 — ISA Extraction (Full Pipeline)

This is the most time-intensive step. It runs five LLM passes against the database.

```bash
rosetta generate \
  --db dbs/arm_test.db \
  --name ARM_test \
  --out ./output \
  --concurrency 2
```

**Expected log sequence:**
```
INFO  Pass 1: extracting ISA metadata
INFO  Pass 2: extracting register file
INFO  Pass 3: extracting instruction mnemonic list
INFO  Estimated instruction count: N
INFO  Mnemonic discovery strategy: List ALL instruction mnemonics ...
...   (10 strategy iterations)
INFO  Mnemonic discovery complete: N unique mnemonics found
INFO  Found N mnemonics
INFO  Pass 4: extracting per-instruction details (concurrency=2)
INFO  Pass 4: instructions 1–20 / N
...
INFO  Pass 5: generating P-code hints
INFO  ISASpec cached to dbs/ARM_test_isa_spec.json
INFO  Module written to output/ARM_test/data/languages
```

**Spot-check the ISASpec JSON cache:**
```bash
python3 -c "
import json
spec = json.load(open('dbs/ARM_test_isa_spec.json'))
print('name      :', spec['meta']['name'])
print('endian    :', spec['meta']['endian'])
print('word bits :', spec['meta']['word_size_bits'])
print('registers :', len(spec['registers']))
print('instructions:', len(spec['instructions']))
# Spot-check one instruction
i = spec['instructions'][0]
print('first instr:', i['mnemonic'], '|', i['semantics'][:60])
print('pcode_hint :', i['pcode_hint'])
"
```

Expected: `name` is a real ISA name, `registers` ≥ 10, `instructions` ≥ 30, `pcode_hint` is a non-empty SLEIGH P-code string (not a `# TODO:` fallback).

**Test cache re-use (skip extraction):**
```bash
rosetta generate \
  --db dbs/arm_test.db \
  --name ARM_test \
  --out ./output \
  --spec-json dbs/ARM_test_isa_spec.json
```

Expected: runs in seconds (no LLM calls); output files are regenerated from the cached spec.

**Limit instructions for a quick smoke test:**
```bash
rosetta generate \
  --db dbs/arm_test.db \
  --name ARM_smoke \
  --out ./output \
  --max-instructions 5
```

Runs in under a minute; useful for verifying the LLM connection is working before committing to a full run.

---

## Stage 3 — Validate Generated SLEIGH

Compile the `.slaspec` with Ghidra's SLEIGH compiler. Requires `GHIDRA_HOME` and `JAVA_HOME`.

```bash
rosetta validate ./output/ARM_test
```

**Expected:** `ARM_test.slaspec: OK`

If you see `FAILED`, the compiler output lists specific line numbers and error messages. Common causes:
- Missing P-code operand types — edit the `.slaspec` manually or adjust Jinja2 templates
- Duplicate `define` statements — indicates the LLM extracted duplicate registers

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

Imports a tiny test binary into Ghidra headless to verify the processor module actually loads without runtime errors.

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

Compare the generated spec to a built-in Ghidra reference processor. Requires Ghidra.

```bash
rosetta evaluate ./output/ARM_test --reference ARM:LE:32:v7
```

**Expected output (example):**
```
Semantic similarity : 0.72
Instruction coverage: 0.58   (58% of ARM v7 mnemonics present)
Register overlap    : 0.81   (Jaccard)
```

Instruction coverage below 0.5 suggests Pass 3 (mnemonic discovery) missed a large portion of the ISA — consider re-running with a more comprehensive manual or increasing `TOP_K`.

---

## Stage 7 — Batch Pipeline

Run the full pipeline for multiple ISA targets defined in a YAML manifest.

```bash
rosetta batch --manifest manifests/arm.yaml --out ./output
```

Append `--skip-extraction` to re-use cached `*_isa_spec.json` files (skips all LLM calls):
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
Check `.env` has `LLM_API_KEY=sk-ant-…` and that `LLM_PROVIDER=anthropic`.

### LangGraph structured output parse error
```
ValueError: Extraction failed after 3 retries.
```
The LLM is not producing valid JSON matching the Pydantic schema. Try:
1. Switching to a more capable model (e.g. `claude-sonnet-4-6` → `claude-opus-4-7`)
2. Increasing `MAX_RETRIES=5` in `.env`
3. Lowering `TEMPERATURE=0` (should already be 0)

### SQLite embedding dimension mismatch
```
struct.error: unpack requires a buffer of N bytes
```
The `.db` was created with a different embedding model than the current `EMBED_MODEL`. Delete the database and re-ingest:
```bash
rm dbs/arm_test.db
rosetta ingest manuals/arm_reference.pdf --db dbs/arm_test.db
```

### GHIDRA_HOME not set
```
RuntimeError: GHIDRA_HOME is not set.
```
Add `GHIDRA_HOME=/path/to/ghidra_12.1_PUBLIC` to `.env` and ensure the path exists.

### Out of memory during Pass 4
Pass 4 runs concurrent LLM calls. Reduce concurrency:
```bash
rosetta generate --db dbs/arm_test.db --name ARM_test --out ./output --concurrency 1
```

---

## Quick Reference — All CLI Commands

```bash
rosetta ingest  <manual.pdf>  --db <path.db>
rosetta generate              --db <path.db> --name <Name> --out <dir>
                              [--spec-json <cache.json>]
                              [--concurrency N]
                              [--max-instructions N]
rosetta validate  <module_dir>
rosetta install   <module_dir>  [--force]
rosetta load-test <module_dir>  [--language-id ID]
rosetta evaluate  <module_dir>  --reference <LANG_ID or .slaspec path>
rosetta batch   --manifest <file.yaml> --out <dir>  [--skip-extraction]
rosetta graph   --results <batch_results.json>  [--out img.png] [--no-display]
rosetta graph   --slaspec  <generated.slaspec>  [--out img.png] [--no-display]
```
