"""Apply a previously-produced judgments file to the DB, without any LLM.

Invocation::

    python obsv/manage.py apply_judgments --judgments PATH --model MODEL [--dry-run]

A manual-repair tool for the case where the LLM classify phase produced a
``judgments_*.json`` file but the run crashed before (or during) persisting it —
for example the Neon ``AdminShutdown`` that motivated the persistent
``pipeline_runs/`` workdir. It applies that file directly, running no LLM.

``--model`` is **required**: the model label stored in ``llm_model`` is never
guessed. (``run_pipeline``'s automated resume instead reads a ``.model`` sidecar
it wrote itself.)

The command mirrors ``run_pipeline``'s persist semantics: rows are matched by the
pk-derived ``entry_id`` (``e{pk:04d}``), only ``new`` rows are touched (so a
re-run is idempotent and human edits are never clobbered), and each row is saved
one at a time via :func:`dash.judgment_persistence.retry_save`. It does not
create a ``PipelineRun`` — it is a repair tool, not a pipeline run.
"""

import json
import sys
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from dash.judgment_persistence import apply_judgments_to_db
from dash.models import IncidentReport
from pipeline import incident_classifier


class Command(BaseCommand):
    help = (
        "Apply a previously-produced judgments_*.json file to the DB without "
        "running any LLM. Requires --model."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--judgments",
            required=True,
            help="Path to a judgments_*.json file to apply.",
        )
        parser.add_argument(
            "--model",
            required=True,
            help="Model label to store in llm_model (never guessed).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would change without writing to the DB.",
        )

    def handle(self, *args, **options):
        path = Path(options["judgments"])
        model = options["model"]
        dry_run = options["dry_run"]

        # Validate the file hard (mirrors incident_classifier.classify): a
        # missing/invalid file is a failure, not a silent no-op.
        if not path.exists():
            raise CommandError(f"judgments file not found at {path}")
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise CommandError(
                f"judgments file at {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(raw, list):
            raise CommandError(
                f"judgments file at {path} top-level is {type(raw).__name__}, "
                "expected a JSON list"
            )

        judgments = incident_classifier.parse_judgments(path)

        # Map every report by its pk-derived entry_id. Building from all rows
        # (not just `new`) lets apply_judgments_to_db skip already-judged rows
        # explicitly (so re-runs are idempotent and print a skipped count),
        # while genuinely unknown entry_ids are reported separately.
        reports_by_entry_id = {
            f"e{r.pk:04d}": r for r in IncidentReport.objects.all()
        }

        if dry_run:
            would_relevant = would_rejected = would_skip = unknown = 0
            for j in judgments:
                report = reports_by_entry_id.get(j["entry_id"])
                if report is None:
                    unknown += 1
                elif report.status != IncidentReport.StatusType.new:
                    would_skip += 1
                elif j["keep"]:
                    would_relevant += 1
                else:
                    would_rejected += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f"[dry-run] would apply {would_relevant} relevant, "
                    f"{would_rejected} rejected; "
                    f"{would_skip} already-judged skipped, "
                    f"{unknown} unknown entry_id(s). "
                    "No changes written."
                )
            )
            return

        now = timezone.now()
        judged_ids, relevant, rejected, skipped, unknown = apply_judgments_to_db(
            judgments, reports_by_entry_id, model, now
        )
        if unknown:
            print(
                f"apply_judgments: {unknown} judgment(s) had an entry_id not "
                "among the `new` rows and were skipped",
                file=sys.stderr,
            )
        self.stdout.write(
            self.style.SUCCESS(
                f"Applied judgments (model={model}): "
                f"{relevant} relevant, {rejected} rejected, "
                f"{skipped} already-judged skipped, {unknown} unknown."
            )
        )
