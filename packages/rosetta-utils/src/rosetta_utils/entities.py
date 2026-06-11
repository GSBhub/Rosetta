"""Read docquery structural entity tags from a Chroma collection.

docquery's ingest-time `tag_entities` writes ``entity_<type>`` metadata on each
matching chunk (a ";"-joined list of distinct names). This module turns those
tags into an ordered, deduplicated list of entity names — the deterministic,
complete equivalent of docquery's ``cursor_enumerate(entity_type)``, usable
programmatically to drive the instruction / register decode loops.
"""

from __future__ import annotations

import logging
from typing import Any

from docquery.config import ENTITY_PREFIX

log = logging.getLogger(__name__)


def _page_sort_key(item: "tuple[Any, int, str]") -> tuple[int, int, int]:
    """Page-ascending, then retrieval order — mirrors docquery's cursor_enumerate.

    Missing/non-numeric pages sort last but keep a stable order.
    """
    page, order, _name = item
    try:
        return (0, int(page), order)
    except (TypeError, ValueError):
        return (1, 0, order)


def read_tagged_entities(settings: Any, entity_type: str) -> list[str] | None:
    """Return distinct ``entity_<entity_type>`` names from the DB, page-ordered.

    Returns ``None`` (not ``[]``) when the store has no tags of this type, so a
    caller can distinguish "untagged DB → fall back" from "tagged but empty".
    """
    key = f"{ENTITY_PREFIX}{entity_type}"
    try:
        collection = settings.vs._collection
        raw = collection.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        if not isinstance(metas, list):
            return None
    except Exception as exc:
        log.warning("read_tagged_entities(%r): DB access failed: %s", entity_type, exc)
        return None

    items: list[tuple[Any, int, str]] = []
    seen: set[str] = set()
    for order, meta in enumerate(metas):
        val = (meta or {}).get(key)
        if not val:
            continue
        page = (meta or {}).get("page")
        for name in str(val).split(";"):
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            items.append((page, order, name))

    if not items:
        return None

    items.sort(key=_page_sort_key)
    return [name for _page, _order, name in items]
