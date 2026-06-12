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


def _sort_key(item: "tuple[Any, Any, int, str]") -> tuple[int, int, int, int]:
    """Section-order, then page-ascending, then retrieval order.

    Mirrors docquery's ``cursor_enumerate`` section-aware sort: entities are
    ordered by the document's own section sequence (from the PDF outline) and
    only fall back to page order within a section or when the store has no
    sections. Missing/non-numeric section_order sorts before sectioned content
    (front matter); missing/non-numeric pages sort last but keep a stable order.
    """
    section_order, page, order, _name = item
    try:
        section = int(section_order)
    except (TypeError, ValueError):
        section = -1
    try:
        return (section, 0, int(page), order)
    except (TypeError, ValueError):
        return (section, 1, 0, order)


def _read_entity_rows(settings: Any, entity_type: str) -> "list[tuple[Any, Any, int, str, Any]] | None":
    """Return ``(section_order, page, retrieval_order, name, section_title)`` rows.

    Distinct entity names only, in deterministic section/page order. ``None``
    when the store is unreachable or has no tags of this type.
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

    items: list[tuple[Any, Any, int, str, Any]] = []
    seen: set[str] = set()
    for order, meta in enumerate(metas):
        meta = meta or {}
        val = meta.get(key)
        if not val:
            continue
        page = meta.get("page")
        section_order = meta.get("section_order")
        section_title = meta.get("section")
        for name in str(val).split(";"):
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            items.append((section_order, page, order, name, section_title))

    if not items:
        return None

    items.sort(key=lambda it: _sort_key(it[:4]))
    return items


def read_tagged_entities(settings: Any, entity_type: str) -> list[str] | None:
    """Return distinct ``entity_<entity_type>`` names from the DB, in document order.

    Ordered by the document's section sequence (PDF outline) then page; on a
    store with no sections this is exactly page order. Returns ``None`` (not
    ``[]``) when the store has no tags of this type, so a caller can distinguish
    "untagged DB → fall back" from "tagged but empty".
    """
    rows = _read_entity_rows(settings, entity_type)
    if rows is None:
        return None
    return [name for _so, _page, _order, name, _sec in rows]


def read_tagged_entities_by_section(
    settings: Any, entity_type: str,
) -> "list[tuple[str | None, list[str]]] | None":
    """Like :func:`read_tagged_entities` but grouped by section, in document order.

    Returns an ordered list of ``(section_title, [names])`` — contiguous runs of
    the deterministic ordering grouped by ``section`` title (``None`` for
    unsectioned entities). ``None`` when the store has no tags of this type.
    """
    rows = _read_entity_rows(settings, entity_type)
    if rows is None:
        return None
    groups: list[tuple[str | None, list[str]]] = []
    for _so, _page, _order, name, section_title in rows:
        title = section_title if section_title else None
        if groups and groups[-1][0] == title:
            groups[-1][1].append(name)
        else:
            groups.append((title, [name]))
    return groups
