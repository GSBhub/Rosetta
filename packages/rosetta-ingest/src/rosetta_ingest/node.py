"""Node 0: Ingest a PDF or source directory into a docquery ChromaDB database.

Reads from state:  source_path, db_path, settings_dict
Returns to state:  errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosetta_schemas.state import PipelineState

log = logging.getLogger(__name__)


def ingest_node(state: PipelineState) -> dict[str, Any]:
    """Ingest source_path into the ChromaDB at db_path.

    If source_path is not set the node is a no-op: the generate workflow
    always ingests separately via 'rosetta ingest' before calling the graph.
    """
    import docquery
    from docquery.config import Settings

    source_path = state.get("source_path", "")
    db_path = state.get("db_path", "")

    if not source_path:
        log.debug("ingest_node: source_path not set — skipping (db already ingested)")
        return {"errors": []}

    if not db_path:
        return {"errors": ["ingest_node: db_path is required"]}

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        settings.db_path = db_path
        src = Path(source_path)

        if src.is_dir():
            exts = ("*.c", "*.h", "*.cpp", "*.hpp", "*.py")
            src_files = [f for ext in exts for f in sorted(src.rglob(ext))]
            if not src_files:
                return {"errors": [f"ingest_node: no source files found in {src}"]}
            items: list = [str(f) for f in src_files]
            log.info("Ingesting %d source files from %s", len(src_files), src)
        else:
            items = [str(src)]
            log.info("Ingesting PDF: %s", src)

        n = docquery.ingest(items, settings=settings)
        log.info("Ingested %d documents into %s", n, db_path)
        return {"errors": []}

    except Exception as exc:
        log.exception("ingest_node failed")
        return {"errors": [f"ingest_node: {exc}"]}
