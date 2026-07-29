"""Run the incident pipeline: ingest, classify, then extract detail fields.

Invocation::

    python obsv/manage.py run_pipeline [--model provider/model] [--workdir PATH]

The command runs in phases, each crash-safe against the next:

1. Ingest: collect raw entries via
   :func:`pipeline.incident_fetcher.collect_data`, project each with
   :func:`~pipeline.incident_fetcher.prepare_entry`, and upsert onto
   :class:`~dash.models.IncidentReport` keyed on the ``(source, source_id)``
   natural key. Deterministic per-source columns from
   :func:`~pipeline.incident_fetcher.map_direct_fields` are written alongside
   ``raw_data``. ``status`` and the judgment/extraction fields are left to the
   model default.
2. Classify: select every row still ``new`` (including leftovers from a prior
   partial run), shell to the LLM via
   :func:`pipeline.incident_classifier.classify`, and write each validated
   judgment back. Rows the LLM never judged stay ``new`` and are retried.
3. Extract: select every ``llm_relevant`` row, shell to the LLM via
   :func:`pipeline.incident_classifier.extract_details`, and fill content
   columns that still hold their default (deterministic and human values win).
   Extracted rows advance to ``pending``. This phase is failure-isolated: an
   extraction crash never rolls back the already-committed judgment updates.
"""

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from dash.models import IncidentReport, PipelineRun, SourceIngestion
from pipeline import incident_classifier, incident_fetcher
from pipeline.main import get_data_paths


# Columns fillable by the extraction pass, mapped to their model default. An
# extracted value is written only when the column still holds this default, so
# deterministic (4a) values and human edits always win.
_FILL_DEFAULTS = {
    "title": "",
    "description": "",
    "animal_type": "unknown",
    "animal_species": "",
    "animal_count": -1,
    "harm_type": "",
    "harm_description": "",
    "ai_system": "",
    "country": "",
}


class Command(BaseCommand):
    help = "Ingest incident source rows, classify them, and extract detail fields."

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
                "Directory for the pipeline's intermediate JSON files. If given, "
                "the files are written there and left in place (debug override). "
                "If omitted, a temporary directory is used and auto-deleted."
            ),
        )

    def handle(self, *args, **options):
        now = timezone.now()

        #The - prefixing started_at means descending order.
        #Then .first() gives first item, meaning most recently completed.
        last_run = (PipelineRun.objects.filter(status=PipelineRun.Status.COMPLETED)
                    .order_by("-started_at")
                    .first())
        last_run_at = last_run.started_at if last_run else None

        run = PipelineRun.objects.create(started_at=now)

        created_count, refreshed_count = self._ingest(now, last_run_at)

        model: str | None = None
        judged_relevant = judged_llm_rejected = still_unjudged = 0
        extracted = still_unextracted = 0

        # A single workdir spans both LLM phases so --workdir debug files land
        # in one place. A tempdir auto-deletes; an explicit --workdir is left.
        if options["workdir"]:
            workdir_ctx = contextlib.nullcontext(options["workdir"])
        else:
            workdir_ctx = tempfile.TemporaryDirectory()

        try:
            with workdir_ctx as wd:
                workdir = Path(wd)

                # --- classify unjudged rows ---
                # Filter by status, not by this run's entries, so rows left `new`
                # by a previous crashed/partial run get picked up automatically.
                unjudged = list(
                    IncidentReport.objects.filter(status=IncidentReport.StatusType.new)
                )
                if not unjudged:
                    self.stdout.write("No unjudged records; skipping classification.")
                else:
                    model, judged_relevant, judged_llm_rejected, still_unjudged = (
                        self._classify(unjudged, workdir, options["model"], now)
                    )

                # --- Phase 4b: extract details from relevant rows ---
                relevant = list(
                    IncidentReport.objects.filter(
                        status=IncidentReport.StatusType.llm_rel
                    )
                )
                if not relevant:
                    self.stdout.write("No relevant records to extract; skipping extraction.")
                else:
                    # Failure isolation: an extraction crash must not abort the
                    # already-committed judgment updates. Leave rows llm_relevant
                    # (retried next run) and continue to the summary.
                    try:
                        emodel, extracted, still_unextracted = self._extract(
                            relevant, workdir, options["model"], now
                        )
                        model = model or emodel
                    except Exception as exc:
                        print(
                            f"run_pipeline: extraction phase failed ({exc!r}); "
                            f"leaving {len(relevant)} row(s) llm_relevant for retry",
                            file=sys.stderr,
                        )
                        still_unextracted = len(relevant)
            run.status = PipelineRun.Status.COMPLETED

        except Exception:
            run.status = PipelineRun.Status.FAILED
            raise

        finally:
            run.completed_at = timezone.now()
            run.save(update_fields=["status", "completed_at"])

        # --- Phase 3e/4b.3: summary ---
        self._write_summary(
            ingested_new=created_count,
            refreshed=refreshed_count,
            judged_relevant=judged_relevant,
            judged_llm_rejected=judged_llm_rejected,
            still_unjudged=still_unjudged,
            extracted=extracted,
            still_unextracted=still_unextracted,
            model=model,
        )

    # ------------------------------------------------------------------ #

    def _ingest(self, now, since=None):
        """Phase 4a/3a: upsert every collected entry; return (created, refreshed)."""
        created_count = 0
        refreshed_count = 0

        already_ingested = set(
            SourceIngestion.objects
            .filter(single_ingestion=True, ingested_at__isnull=False)
            .values_list("source", flat=True)
        )

        entries = incident_fetcher.collect_data(
            now = now,
            since = since,
            already_ingested=already_ingested
        )

        sources_seen: set[str] = set()

        with transaction.atomic():
            for entry in entries:
                source = entry.get("aaiid_data_source")
                if source:
                    sources_seen.add(source)

                prepared = incident_fetcher.prepare_entry(entry)
                if not prepared["source_id"]:
                    print(
                        "run_pipeline: skipping entry with empty source_id "
                        f"(source={prepared['aaiid_data_source']!r}); "
                        "no natural key to upsert on",
                        file=sys.stderr,
                    )
                    continue

                # raw_data plus deterministic per-source columns are refreshed
                # on both create and update so source corrections propagate;
                # time_ingested is applied only on creation (create_defaults) so
                # reruns preserve the original ingest time. status and the
                # judgment/extraction fields are never set here, so the model
                # default and any human edits survive.
                defaults = {
                    "raw_data": prepared["json_blob"],
                    **incident_fetcher.map_direct_fields(entry),
                }
                _, created = IncidentReport.objects.update_or_create(
                    source=prepared["aaiid_data_source"],
                    source_id=prepared["source_id"],
                    defaults=defaults,
                    create_defaults={**defaults, "time_ingested": now},
                )
                if created:
                    created_count += 1
                else:
                    refreshed_count += 1

        for source in sources_seen:
            SourceIngestion.objects.update_or_create(
                source=source,
                defaults={"ingested_at": now}
            )

        return created_count, refreshed_count

    def _build_llm_input(self, reports):
        """Build the LLM input list and an ``{entry_id: report}`` map.

        entry_id is derived from pk (replacing assign_entry_ids on this path)
        for stable, collision-free ids. aaiid_data_source is re-added because
        _trim_entry strips it from raw_data and the LLM prompts branch on it.
        """
        reports_by_entry_id: dict[str, IncidentReport] = {}
        input_list: list[dict] = []
        for report in reports:
            entry_id = f"e{report.pk:04d}"
            reports_by_entry_id[entry_id] = report
            input_list.append({
                "entry_id": entry_id,
                "aaiid_data_source": report.source,
                **report.raw_data,
            })
        return input_list, reports_by_entry_id

    def _classify(self, unjudged, workdir, model_opt, now):
        """Phases 3c/3d: classify then persist judgments.

        Returns ``(model, judged_relevant, judged_llm_rejected, still_unjudged)``.
        """
        input_list, reports_by_entry_id = self._build_llm_input(unjudged)

        incidents_path, judgments_path = get_data_paths(workdir)
        incident_fetcher.write_json(input_list, incidents_path)
        try:
            model, judgments = incident_classifier.classify(
                workdir, incidents_path, judgments_path, model_opt
            )
        except subprocess.CalledProcessError as exc:
            # Ingested rows are already committed as `new`, so the next run
            # resumes exactly where this one failed.
            raise CommandError(f"Pipeline classification failed: {exc}") from exc

        judged_relevant = 0
        judged_llm_rejected = 0
        judged_ids: set[str] = set()

        with transaction.atomic():
            for j in judgments:
                entry_id = j["entry_id"]
                report = reports_by_entry_id.get(entry_id)
                if report is None:
                    print(
                        "run_pipeline: judgment has unknown entry_id "
                        f"{entry_id!r} (not among {len(reports_by_entry_id)} "
                        "unjudged rows); skipping",
                        file=sys.stderr,
                    )
                    continue
                judged_ids.add(entry_id)

                if j["keep"]:
                    report.status = IncidentReport.StatusType.llm_rel
                    judged_relevant += 1
                else:
                    report.status = IncidentReport.StatusType.llm_rej
                    judged_llm_rejected += 1
                report.llm_reasoning = j["reasoning"]
                report.confidence = IncidentReport.CONFIDENCE_MAP.get(
                    j["confidence"], 0
                )
                report.llm_model = model
                report.time_judged = now
                report.save(update_fields=[
                    "status",
                    "llm_reasoning",
                    "confidence",
                    "llm_model",
                    "time_judged",
                ])

        # Rows the LLM never returned a valid judgment for stay `new`; they'll
        # be retried next run. Mirrors build_records's "missing" report.
        still_unjudged = len(reports_by_entry_id) - len(judged_ids)
        if still_unjudged:
            print(
                f"run_pipeline: {still_unjudged} unjudged row(s) received no "
                "valid judgment and remain `new` (will retry next run)",
                file=sys.stderr,
            )

        return model, judged_relevant, judged_llm_rejected, still_unjudged

    def _extract(self, relevant, workdir, model_opt, now):
        """Phase 4b: extract detail fields then persist them (fill-only-if-default).

        Returns ``(model, extracted, still_unextracted)``.
        """
        input_list, reports_by_entry_id = self._build_llm_input(relevant)

        ts = timezone.now().strftime("%Y%m%d_%H%M%S")
        input_path = workdir / f"details_input_{ts}.json"
        output_path = workdir / f"details_{ts}.json"
        incident_fetcher.write_json(input_list, input_path)

        model, details = incident_classifier.extract_details(
            workdir, input_path, output_path, model_opt
        )

        extracted = 0
        applied_ids: set[str] = set()

        with transaction.atomic():
            for d in details:
                entry_id = d["entry_id"]
                report = reports_by_entry_id.get(entry_id)
                if report is None:
                    print(
                        "run_pipeline: detail has unknown entry_id "
                        f"{entry_id!r} (not among {len(reports_by_entry_id)} "
                        "relevant rows); skipping",
                        file=sys.stderr,
                    )
                    continue
                applied_ids.add(entry_id)

                update_fields = []
                for col, default in _FILL_DEFAULTS.items():
                    # Fill only if the column still holds its default, so
                    # deterministic (4a) values and human edits always win.
                    if getattr(report, col) == default:
                        setattr(report, col, d[col])
                        update_fields.append(col)

                report.status = IncidentReport.StatusType.pending
                report.time_hydrated = now
                update_fields += ["status", "time_hydrated"]
                report.save(update_fields=update_fields)
                extracted += 1

        # Rows the LLM omitted or that failed parse_details validation stay
        # llm_relevant; log a count (mirrors the missing-judgments report).
        still_unextracted = len(reports_by_entry_id) - len(applied_ids)
        if still_unextracted:
            print(
                f"run_pipeline: {still_unextracted} relevant row(s) received no "
                "valid extraction and remain `llm_relevant` (will retry next run)",
                file=sys.stderr,
            )

        return model, extracted, still_unextracted

    def _write_summary(
        self,
        *,
        ingested_new: int,
        refreshed: int,
        judged_relevant: int,
        judged_llm_rejected: int,
        still_unjudged: int,
        extracted: int,
        still_unextracted: int,
        model: str | None = None,
    ):
        model_note = f" (model={model})" if model else ""
        self.stdout.write(
            self.style.SUCCESS(
                f"Pipeline complete{model_note}: "
                f"{ingested_new} ingested_new, {refreshed} refreshed, "
                f"{judged_relevant} judged_relevant, "
                f"{judged_llm_rejected} judged_llm_rejected, "
                f"{still_unjudged} still_unjudged, "
                f"{extracted} extracted, {still_unextracted} still_unextracted."
            )
        )
