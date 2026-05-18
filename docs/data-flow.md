# Rosetta Data Flow

Two diagrams: the overall pipeline end-to-end, then a zoom-in on the extraction passes.

---

## 1. Full Pipeline

```mermaid
flowchart TD
    PDF([PDF Manual])

    subgraph INGEST["rosetta ingest  (docquery)"]
        direction TB
        LOAD["pdf_loader.load()\n─────────────────\nfitz  → raw text per page\npdfplumber → tables as markdown\npytesseract → OCR fallback\n(<50 chars triggers OCR @ 200 dpi)"]
        CHUNK["chunker.chunk()\n─────────────────\nRecursiveCharacterTextSplitter\nchunk_size=1000  overlap=200"]
        EMBED["get_embeddings()\n─────────────────\nOllama embed_documents()\nin batches of 32"]
        DB[("SQLite VectorStore\n─────────────────\nchunks         (text + metadata)\nchunk_embeddings  (float32 BLOB)\nchunks_fts        (FTS5 index)")]
        LOAD --> CHUNK --> EMBED --> DB
    end

    subgraph EXTRACT["rosetta generate  — ISAExtractor.extract()"]
        direction TB
        P1["Pass 1  ISA Metadata\n→ ISAMeta"]
        P2["Pass 2  Register File\n→ list[RegisterDef]"]
        P3["Pass 3  Mnemonic Discovery\n→ list[str]\n(LangGraph loop — see diagram 2)"]
        P4["Pass 4  Per-instruction Details\n→ list[InstructionDef]\nasync gather · semaphore=2 · chunks of 20"]
        P5["Pass 5  P-code Hints\n→ InstructionDef.pcode_hint\nDirect LLM call per instruction"]
        SPEC[("ISASpec\n*_isa_spec.json")]
        P1 --> P2 --> P3 --> P4 --> P5 --> SPEC
    end

    subgraph GEN["ModuleGenerator.generate()  (Jinja2)"]
        direction LR
        T1["processor.slaspec.j2"]
        T2["processor.pspec.j2"]
        T3["processor.cspec.j2"]
        T4["processor.ldefs.j2"]
        OUT["output/Name/data/languages/\n.slaspec  .pspec  .cspec  .ldefs"]
        T1 & T2 & T3 & T4 --> OUT
    end

    subgraph POSTGEN["rosetta validate / install / evaluate"]
        direction TB
        VAL["sleigh compiler\n(GHIDRA_HOME/support/sleigh)\n→ .sla binary"]
        INST["rosetta install\n→ GHIDRA_HOME/Ghidra/Processors/"]
        EVAL["similarity.compare()\n· semantic similarity (cosine)\n· instruction coverage\n· register Jaccard overlap"]
    end

    PDF --> INGEST
    INGEST --> EXTRACT
    SPEC --> GEN
    OUT --> VAL --> INST
    OUT --> EVAL
```

---

## 2. Extraction Pipeline Internals

Each of passes 1, 2, and 4 runs an **ExtractionPipeline** (a three-node LangGraph).
Pass 3 (mnemonic discovery) runs a separate **multi-strategy LangGraph loop** that calls ExtractionPipeline internally.
Pass 5 calls the LLM directly with no RAG retrieval.

### ExtractionPipeline (one call, used by passes 1, 2, 4)

```mermaid
flowchart LR
    START(["query string"])

    subgraph EP["ExtractionPipeline  (LangGraph)"]
        direction LR
        RET["retrieve\n─────────────\nEmbed query\n→ cosine search (top_k=5)\nconcatenate chunk text\n→ retrieved_context"]
        EXT["extract\n─────────────\nSystemMessage:\n  system_prompt\n  + JSON schema\nHumanMessage:\n  retrieved_context\n  + query\n  + prior errors\n→ LLM.invoke()\n→ raw JSON string"]
        VAL["validate\n─────────────\nPydantic\nmodel_validate_json()\n→ validated BaseModel\nor increment retry_count"]
        RET --> EXT --> VAL
        VAL -->|"validation error\n(≤ max_retries=3)"| EXT
    end

    DONE(["validated Pydantic object"])
    START --> RET
    VAL -->|success| DONE
```

### Pass 3 — Mnemonic Discovery (LangGraph)

```mermaid
flowchart TD
    START(["discover_mnemonics()"])

    COUNT["count node\n─────────────────────\nExtractionPipeline → _InstructionCount\n'How many mnemonics in this manual?'\n→ total_expected (rough estimate)"]

    FETCH["fetch node\n─────────────────────\nPop next strategy from queue (10 total):\n  1. 'List ALL mnemonics'\n  2. Data-processing (ADD, SUB, AND…)\n  3. Load/store (LDR, STR, LDM…)\n  4. Branch/control (B, BL, BX…)\n  5. Multiply/divide (MUL, MLA…)\n  6. Mnemonics A–F\n  7. Mnemonics G–N\n  8. Mnemonics O–Z\n  9. SIMD/VFP/NEON/FP\n  10. System/coprocessor/barrier\nEach → ExtractionPipeline → _MnemonicList\nNew names cleaned + deduplicated"]

    CHECK["check node\n─────────────────────\nAll 10 strategies exhausted? → done\n(coverage check disabled: LLM\ncount estimate is unreliable)"]

    DONE(["sorted deduplicated list[str]"])

    START --> COUNT --> FETCH --> CHECK
    CHECK -->|"strategies remain"| FETCH
    CHECK -->|done| DONE
```

### Pass 5 — P-code Generation (direct LLM)

```mermaid
flowchart LR
    IN(["InstructionDef\n(mnemonic + semantics)"])
    LLM["LLM.invoke()\n─────────────────────\nSystemMessage: SLEIGH P-code expert prompt\nHumanMessage: mnemonic + semantics text\n→ single-line P-code string"]
    OUT(["pcode_hint stored\non InstructionDef"])
    IN --> LLM --> OUT
```

---

## Data Shapes

| Stage | Input | Output |
|---|---|---|
| `pdf_loader.load()` | `.pdf` file | `list[Document]` — one per page |
| `chunker.chunk()` | `list[Document]` | `list[Document]` — overlapping text chunks |
| `get_embeddings().embed_documents()` | `list[str]` | `list[list[float]]` (dim = model-dependent) |
| `VectorStore.add_chunks()` | chunks + embeddings | SQLite rows: `chunks`, `chunk_embeddings`, `chunks_fts` |
| `VectorStore.similarity_search()` | query vector | top-k `dict` rows by cosine distance |
| `ExtractionPipeline.run()` | query string | validated Pydantic `BaseModel` |
| `ISAExtractor.extract()` | db path | `ISASpec` (meta + registers + instructions) |
| `ModuleGenerator.generate()` | `ISASpec` + name | `.slaspec`, `.pspec`, `.cspec`, `.ldefs` |

---

## Key Configuration (`.env`)

| Variable | Controls |
|---|---|
| `EMBED_PROVIDER` / `EMBED_MODEL` / `EMBED_BASE_URL` | Embedding model for ingest + retrieval |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` | LLM for all five extraction passes |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text splitter parameters (default 1000/200) |
| `TOP_K` | Number of chunks returned per RAG query (default 5) |
| `MAX_RETRIES` | LangGraph retry budget per ExtractionPipeline call (default 3) |
| `TEMPERATURE` | LLM sampling temperature (default 0) |
| `GHIDRA_HOME` / `JAVA_HOME` | Required for validate, install, load-test |
