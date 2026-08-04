"""Crash-resilient persistence helpers for the classify phase.

This module lives in the ``dash`` app (not the Django-free ``pipeline/``
library) because it uses the ORM. It provides:

* :func:`retry_save` — a per-row save that reconnects and retries on a transient
  :class:`~django.db.utils.OperationalError` (Neon serverless can drop an idle or
  stale connection at any time; a self-contained single-row ``UPDATE`` is safe to
  retry after dropping the dead connection).
* :func:`apply_judgments_to_db` — applies validated judgments to
  :class:`~dash.models.IncidentReport` rows one at a time, in autocommit mode, so
  a mid-batch connection drop preserves everything committed before the drop.

Neither wraps the loop in ``transaction.atomic()``: each ``save`` is an
independent single-row ``UPDATE`` and ``status`` already tracks per-row progress,
so committed rows are excluded from the next run's queue and uncommitted rows are
retried.
"""

import sys
import time

from django.db import close_old_connections
from django.db.utils import OperationalError

from dash.models import IncidentReport


def retry_save(report, update_fields, attempts=3, base_delay=1.0):
    """Save ``report`` (limited to ``update_fields``), retrying on drops.

    On a transient :class:`OperationalError` the stale connection is dropped via
    :func:`django.db.close_old_connections` and the single-row ``UPDATE`` is
    retried after exponential backoff. Only ``OperationalError`` is caught;
    ``IntegrityError`` and other database errors propagate immediately. The last
    attempt re-raises so a genuinely dead DB still fails loudly.
    """
    for attempt in range(attempts):
        try:
            report.save(update_fields=update_fields)
            return
        except OperationalError:
            # Drop the (possibly dead) connection so the retry opens a fresh one.
            close_old_connections()
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2 ** attempt)


def apply_judgments_to_db(judgments, reports_by_entry_id, model, now):
    """Apply validated ``judgments`` to reports; return a counts summary.

    Each judgment is persisted with its own :func:`retry_save` call (no
    surrounding transaction), so a failure part-way through leaves earlier rows
    committed. Rows not present in ``reports_by_entry_id`` are logged and counted
    as ``unknown``; rows no longer ``new`` are skipped defensively (queue
    selection should already exclude them).

    Returns ``(judged_ids, relevant, rejected, skipped, unknown)`` where
    ``judged_ids`` is the set of entry_ids successfully saved this call.
    """
    judged_ids: set[str] = set()
    relevant = 0
    rejected = 0
    skipped = 0
    unknown = 0

    for j in judgments:
        entry_id = j["entry_id"]
        report = reports_by_entry_id.get(entry_id)
        if report is None:
            print(
                "apply_judgments: judgment has unknown entry_id "
                f"{entry_id!r} (not among {len(reports_by_entry_id)} "
                "candidate rows); skipping",
                file=sys.stderr,
            )
            unknown += 1
            continue

        if report.status != IncidentReport.StatusType.new:
            # Defensive: already judged (or human-edited); never clobber.
            skipped += 1
            continue

        if j["keep"]:
            report.status = IncidentReport.StatusType.llm_rel
        else:
            report.status = IncidentReport.StatusType.llm_rej

        report.llm_reasoning = j["reasoning"]
        report.confidence = IncidentReport.CONFIDENCE_MAP.get(j["confidence"], 0)
        report.llm_model = model
        report.time_judged = now

        retry_save(report, [
            "status",
            "llm_reasoning",
            "confidence",
            "llm_model",
            "time_judged",
        ])

        # Count only after a successful save so a retry never double-counts.
        judged_ids.add(entry_id)
        if j["keep"]:
            relevant += 1
        else:
            rejected += 1

    return judged_ids, relevant, rejected, skipped, unknown
