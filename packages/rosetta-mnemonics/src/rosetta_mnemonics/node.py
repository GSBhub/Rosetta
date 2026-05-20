"""Node 3: Mnemonic discovery (Pass 3).

Reads from state:  db_path, settings_dict, filter_mnemonics
Returns to state:  mnemonics, errors
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any

from rosetta_schemas.state import PipelineState

log = logging.getLogger(__name__)


def mnemonics_node(state: PipelineState) -> dict[str, Any]:
    """Discover all instruction mnemonics via the inner LangGraph multi-strategy loop."""
    from docquery.config import Settings
    from rosetta_mnemonics.discovery import discover_mnemonics
    from rosetta_utils.chroma import get_chroma_collection

    try:
        settings = Settings(**(state.get("settings_dict") or {}))
        db_path = state["db_path"]
        settings.db_path = db_path
        settings.vs = get_chroma_collection(db_path)
        mnemonics = discover_mnemonics(
            db_path=db_path,
            settings=settings,
        )

        filter_pat = state.get("filter_mnemonics")
        if filter_pat:
            patterns = [p.strip().upper() for p in filter_pat.split(",")]
            before = len(mnemonics)
            mnemonics = [m for m in mnemonics if any(fnmatch.fnmatch(m, p) for p in patterns)]
            log.info("filter %r: %d → %d mnemonics", filter_pat, before, len(mnemonics))

        log.info("mnemonics_node: %d mnemonics discovered", len(mnemonics))
        return {"mnemonics": mnemonics, "errors": []}

    except Exception as exc:
        log.exception("mnemonics_node failed")
        return {"mnemonics": [], "errors": [f"mnemonics_node: {exc}"]}
