"""Run the incident pipeline and upsert kept judgments into the database.

Invocation::

    python obsv/manage.py run_pipeline [--model provider/model] [--workdir PATH]

Collects and classifies incident data via :func:`pipeline.main.run_pipeline`,
then maps each kept, reconciled record onto an :class:`~dash.models.IncidentReport`
via ``update_or_create`` keyed on the ``(source, source_id)`` natural key so
reruns upsert rather than duplicate.
"""

import subprocess

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from dash.models import IncidentReport
from pipeline.main import run_pipeline


class Command(BaseCommand):
    help = "Run the incident pipeline and upsert kept judgments into IncidentReport."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            default=None,
            help="LLM to use, in 'provider/model' form. Defaults to the pipeline default.",
        )
        parser.add_argument(
            "--workdir",
            default=None,
            help=(
                "Directory for the pipeline's intermediate JSON files. "
                "Defaults to the current working directory."
            ),
        )

    def handle(self, *args, **options):
        try:
            records, model = run_pipeline(
                model=options["model"],
                workdir=options["workdir"],
            )
        except subprocess.CalledProcessError as exc:
            raise CommandError(f"Pipeline classification failed: {exc}") from exc

        now = timezone.now()
        created_count = 0
        updated_count = 0

        with transaction.atomic():
            for record in records:
                _, created = IncidentReport.objects.update_or_create(
                    source=record["aaiid_data_source"],
                    source_id=record["source_id"],
                    defaults={
                        "raw_data": record["json_blob"],
                        "confidence": IncidentReport.CONFIDENCE_MAP.get(
                            record["confidence"], 0
                        ),
                        "llm_reasoning": record["reasoning"],
                        "llm_model": record["model"],
                        "time_ingested": now,
                        "time_judged": now,
                    },
                )
                if created:
                    created_count += 1
                else:
                    updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Pipeline complete (model={model}): "
                f"{len(records)} kept record(s); "
                f"{created_count} created, {updated_count} updated."
            )
        )
