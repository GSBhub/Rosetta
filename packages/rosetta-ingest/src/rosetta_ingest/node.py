"""Node 0: Ingest a PDF or source directory into a docquery ChromaDB database.

Reads from state:  source_path, db_path, settings_dict
Returns to state:  errors
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langsmith import traceable
from rosetta_schemas.state import PipelineState
from rosetta_utils.tracing import state_summary

log = logging.getLogger(__name__)


@traceable(run_type="chain", name="stage:ingest", process_inputs=state_summary)
def ingest_node(state: PipelineState) -> dict[str, Any]:
    """Ingest source_path into the ChromaDB at db_path.

    If source_path is not set the node is a no-op: the generate workflow
    always ingests separately via 'rosetta ingest' before calling the graph.
    """
    import docquery
    from docquery.config import EntityRule, Settings

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

        # entity_rules round-trips through settings_dict as plain dicts
        # (dataclasses.asdict); reconstruct EntityRule objects so tag_entities works.
        rules = []
        for r in getattr(settings, "entity_rules", None) or []:
            if isinstance(r, EntityRule):
                rules.append(r)
            elif isinstance(r, dict) and r.get("name") and r.get("pattern"):
                rules.append(EntityRule(name=r["name"], pattern=r["pattern"]))
        settings.entity_rules = rules
        if rules:
            log.info("ingest_node: tagging entities %s", [r.name for r in rules])
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
