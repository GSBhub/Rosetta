"""Thread-safe LangChain Chroma singleton.

ChromaDB 1.5.x's Rust backend cannot handle multiple concurrent
PersistentClient instances pointing at the same directory.  When
LangGraph runs meta/registers/mnemonics in parallel they would all race
to open the same DB.  This module gates creation behind a lock and
caches one LangChain Chroma wrapper per db_path so every thread reuses it.

The wrapper (langchain_chroma.vectorstores.Chroma) is what docquery's
ExtractionPipeline expects — it has similarity_search() and add_documents().
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_wrappers: dict[str, Any] = {}  # db_path → LangChain Chroma wrapper


def get_chroma_wrapper(db_path: str, settings: Any) -> Any:
    """Return (and cache) the LangChain Chroma wrapper for *db_path*.

    *settings* is only used for its embedding configuration on the first call
    for a given path; subsequent calls return the cached instance.
    """
    if db_path in _wrappers:
        return _wrappers[db_path]
    with _lock:
        if db_path not in _wrappers:
            import chromadb
            from langchain_chroma.vectorstores import Chroma
            from docquery.embeddings.provider import get_embeddings
            client = chromadb.PersistentClient(path=db_path)
            _wrappers[db_path] = Chroma(
                client=client,
                collection_name="db_knowledge",
                embedding_function=get_embeddings(settings),
            )
    return _wrappers[db_path]
