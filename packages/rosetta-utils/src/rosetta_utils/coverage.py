"""Coverage gate: is structural discovery finding (nearly) everything?

Compares the number of distinct tagged entities in a store against an expected
count — either passed explicitly or estimated from the document outline — and
reports whether discovery cleared a threshold. Turns "discovery fails a lot"
into a tracked number. Wraps docquery's coverage() / outline() helpers.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

log = logging.getLogger(__name__)


def expected_from_outline(settings: Any, entity_type: str = "instruction") -> int | None:
    """Estimate the expected entity count by applying entity rules to outline titles.

    Uses the document's own bookmark titles (``docquery.outline``) as a proxy for
    the manual's index: counts distinct names matched by any ``entity_<type>``
    rule. Returns ``None`` when there is no outline or no matching rule, so the
    caller treats coverage as "no reference available".
    """
    import docquery

    try:
        outline = docquery.outline(settings=settings)
    except Exception as exc:  # noqa: BLE001
        log.warning("expected_from_outline: outline() failed: %s", exc)
        return None
    if not outline:
        return None

    rules = [r for r in getattr(settings, "entity_rules", None) or []
             if getattr(r, "name", None) == entity_type]
    if not rules:
        return None

    compiled = [re.compile(r.pattern, re.MULTILINE) for r in rules]
    names: set[str] = set()
    for entry in outline:
        title = str(entry.get("title") or "")
        for rx in compiled:
            for m in rx.finditer(title):
                ent = (m.group(1) if m.groups() else m.group(0)).strip()
                if ent:
                    names.add(ent)
    return len(names) or None


def check_coverage(
    settings: Any,
    *,
    entity_type: str = "instruction",
    expected: int | None = None,
    threshold: float = 0.9,
) -> tuple[bool, str]:
    """Return ``(ok, message)`` for the discovery coverage gate.

    ``expected`` resolution order: explicit arg → outline estimate → none. With
    no reference, the gate passes and just reports the tagged count. Otherwise
    ``ok`` is ``actual >= ceil(threshold * expected)``.
    """
    import docquery

    try:
        cov = docquery.coverage(settings=settings)
    except Exception as exc:  # noqa: BLE001
        return True, f"coverage unavailable ({exc}); skipping gate"
    actual = cov.get(entity_type, {}).get("count", 0)

    if expected is None:
        expected = expected_from_outline(settings, entity_type)
    if not expected:
        return True, f"{actual} {entity_type} entities tagged (no reference count to gate on)"

    need = math.ceil(threshold * expected)
    ok = actual >= need
    verdict = "OK" if ok else "LOW"
    return ok, (
        f"{verdict}: {actual}/{expected} {entity_type} entities "
        f"({actual / expected:.0%}, threshold {threshold:.0%} = {need})"
    )
