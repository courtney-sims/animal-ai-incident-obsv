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

Crash resilience: the intermediate JSON is written to a persistent
``pipeline_runs/`` directory (inside the project tree) by default rather than an
auto-deleted tempdir, so the expensive LLM output survives a crash and a later
run can resume from it instead of re-running the LLM. Each persist loop saves one
row at a time in autocommit mode (no giant ``transaction.atomic()`` wrapper) and
reconnects-and-retries on a transient ``OperationalError`` (Neon serverless can
drop a stale connection during the long LLM call), so a mid-batch connection drop
preserves every row committed before it.
"""

import contextlib
import subprocess
import sys
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import close_old_connections, connection
from django.db.utils import OperationalError
from django.utils import timezone

from dash.judgment_persistence import apply_judgments_to_db, retry_save
from dash.models import IncidentReport, PipelineRun, SourceIngestion
from pipeline import incident_classifier, incident_fetcher
from pipeline.main import get_data_paths


# Columns fillable by the extraction pass, mapped to their model default. An
# extracted value is written only when the column still holds this default, so
# deterministic (4a) values and human edits always win.
def _drop_stale_connection():
    """Drop a possibly-stale DB connection so the next query reconnects.

    Called just before the long, DB-idle LLM subprocess so the persist phase
    opens a fresh connection (Neon serverless suspends idle endpoints and kills
    the connection). Guarded by ``in_atomic_block`` because closing a connection
    mid-transaction is unsafe — production runs these phases outside any
    transaction, while the test suite wraps each test in one.
    """
    if not connection.in_atomic_block:
        close_old_connections()


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
                "If omitted, they are kept in a persistent 'pipeline_runs/' "
                "directory in the project tree so a crash cannot delete the LLM "
                "output and a later run can resume from it."
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

        # A single workdir spans both LLM phases so intermediate files land in
        # one place. Both the explicit --workdir and the default persist (no
        # auto-delete) so the expensive LLM output survives a crash and a later
        # run can resume from it.
        #
        # The default lives *inside* the project tree (not the system default
        # /tmp) because the opencode subprocess confines its file writes to its
        # detected project/worktree. A /tmp workdir sits outside that scope, so
        # the LLM's judgments file would land in the project root instead of the
        # workdir and the reader would never find it.
        if options["workdir"]:
            workdir_ctx = contextlib.nullcontext(options["workdir"])
        else:
            default_workdir = settings.BASE_DIR.parent / "pipeline_runs"
            default_workdir.mkdir(parents=True, exist_ok=True)
            workdir_ctx = contextlib.nullcontext(default_workdir)

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
            # Persist the final run status even if the connection went stale
            # during the long LLM phase. Retry a couple times on a transient
            # OperationalError, but never mask the original exception (if any):
            # the original failure is what should propagate.
            run.completed_at = timezone.now()
            for attempt in range(3):
                try:
                    run.save(update_fields=["status", "completed_at"])
                    break
                except OperationalError as exc:
                    close_old_connections()
                    if attempt == 2:
                        print(
                            "run_pipeline: could not persist final run status "
                            f"({exc!r})",
                            file=sys.stderr,
                        )
                        break
                    time.sleep(1.0 * 2 ** attempt)

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

        # Each upsert is independent (natural key on (source, source_id)), so no
        # surrounding transaction is needed. A mid-ingest connection drop thus
        # preserves the partial upserts; the missing rows are picked up on the
        # next run. collect_data() already ran above, outside any retry, so a
        # retry never re-fetches network data.
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
            _, created = self._upsert_with_retry(
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

    def _upsert_with_retry(self, *, source, source_id, defaults, create_defaults,
                           attempts=3, base_delay=1.0):
        """update_or_create with reconnect-and-retry on transient drops.

        Mirrors :func:`dash.judgment_persistence.retry_save` for the ingest
        upsert (which is a self-contained SELECT-then-INSERT/UPDATE on the
        natural key). Only ``OperationalError`` is retried.
        """
        for attempt in range(attempts):
            try:
                return IncidentReport.objects.update_or_create(
                    source=source,
                    source_id=source_id,
                    defaults=defaults,
                    create_defaults=create_defaults,
                )
            except OperationalError:
                close_old_connections()
                if attempt == attempts - 1:
                    raise
                time.sleep(base_delay * 2 ** attempt)

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

    @staticmethod
    def _model_sidecar_path(data_path: Path) -> Path:
        """The ``.model`` sidecar path for a judgments/details JSON file."""
        return Path(str(data_path) + ".model")

    def _write_model_sidecar(self, data_path: Path, model: str) -> None:
        """Persist the model string that produced ``data_path``.

        The judgments/details JSON don't record which model produced them, so a
        later resume run reads this sidecar to persist ``llm_model`` accurately.
        """
        self._model_sidecar_path(data_path).write_text(model)

    def _read_model_sidecar(self, data_path: Path) -> str | None:
        """Return the model recorded for ``data_path``, or ``None`` if absent."""
        sidecar = self._model_sidecar_path(data_path)
        if not sidecar.exists():
            return None
        return sidecar.read_text().strip()

    def _find_reusable_judgments(self, workdir, wanted_ids):
        """Return ``(model, judgments)`` from a reusable file, or ``None``.

        Scans ``judgments_*.json`` newest-first. A file is reusable when it has
        an accompanying ``.model`` sidecar (so ``llm_model`` is never guessed)
        and its validated entry_ids are a superset of ``wanted_ids``. Resume is
        all-or-nothing: a file covering only some of the queue is ignored.
        """
        if not wanted_ids:
            return None
        for path in sorted(workdir.glob("judgments_*.json"), reverse=True):
            model = self._read_model_sidecar(path)
            if model is None:
                continue
            judgments = incident_classifier.parse_judgments(path)
            file_ids = {j["entry_id"] for j in judgments}
            if wanted_ids <= file_ids:
                return model, judgments
        return None

    def _find_reusable_details(self, workdir, wanted_ids):
        """Return ``(model, details)`` from a reusable file, or ``None``.

        Mirrors :meth:`_find_reusable_judgments` for the extraction phase. The
        glob is ``details_[0-9]*.json`` (leading digit of the timestamp) so it
        excludes the ``details_input_*.json`` input files.
        """
        if not wanted_ids:
            return None
        for path in sorted(workdir.glob("details_[0-9]*.json"), reverse=True):
            model = self._read_model_sidecar(path)
            if model is None:
                continue
            details = incident_classifier.parse_details(path)
            file_ids = {d["entry_id"] for d in details}
            if wanted_ids <= file_ids:
                return model, details
        return None

    def _classify(self, unjudged, workdir, model_opt, now):
        """Phases 3c/3d: classify then persist judgments.

        Returns ``(model, judged_relevant, judged_llm_rejected, still_unjudged)``.
        """
        input_list, reports_by_entry_id = self._build_llm_input(unjudged)
        wanted_ids = set(reports_by_entry_id)

        # Resume path: if a prior judgments file in the workdir already covers
        # this queue's entry_ids, reuse it and skip the expensive LLM call.
        resume = self._find_reusable_judgments(workdir, wanted_ids)
        if resume is not None:
            model, judgments = resume
            self.stdout.write(
                self.style.SUCCESS(
                    "Reusing existing judgments file; skipping the LLM "
                    f"classify call (model={model})."
                )
            )
        else:
            incidents_path, judgments_path = get_data_paths(workdir)
            incident_fetcher.write_json(input_list, incidents_path)
            # Drop any connection that may have gone stale before the long,
            # DB-idle LLM call so the persist phase opens a fresh one.
            _drop_stale_connection()
            try:
                model, judgments = incident_classifier.classify(
                    workdir, incidents_path, judgments_path, model_opt
                )
            except (
                subprocess.CalledProcessError,
                incident_classifier.ClassificationOutputError,
            ) as exc:
                # Ingested rows are already committed as `new`, so the next run
                # resumes exactly where this one failed. A missing or
                # structurally invalid judgments file (ClassificationOutputError)
                # is a hard failure, not a silent no-op: it propagates so the run
                # is logged `failed` rather than `completed`.
                raise CommandError(f"Pipeline classification failed: {exc}") from exc

            # Record which model produced this file so a later resume run can
            # persist llm_model accurately (the JSON itself doesn't carry it).
            self._write_model_sidecar(judgments_path, model)

        judged_ids, judged_relevant, judged_llm_rejected, _skipped, _unknown = (
            apply_judgments_to_db(judgments, reports_by_entry_id, model, now)
        )

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
        wanted_ids = set(reports_by_entry_id)

        # Resume path: reuse a prior details file that covers this queue.
        resume = self._find_reusable_details(workdir, wanted_ids)
        if resume is not None:
            model, details = resume
            self.stdout.write(
                self.style.SUCCESS(
                    "Reusing existing details file; skipping the LLM extract "
                    f"call (model={model})."
                )
            )
        else:
            ts = timezone.now().strftime("%Y%m%d_%H%M%S")
            input_path = workdir / f"details_input_{ts}.json"
            output_path = workdir / f"details_{ts}.json"
            incident_fetcher.write_json(input_list, input_path)
            # Drop any stale connection before the long, DB-idle LLM call.
            _drop_stale_connection()
            model, details = incident_classifier.extract_details(
                workdir, input_path, output_path, model_opt
            )
            self._write_model_sidecar(output_path, model)

        extracted = 0
        applied_ids: set[str] = set()

        # Per-row autocommit (no transaction wrapper): each save is an
        # independent single-row UPDATE, so a mid-batch drop preserves earlier
        # rows and retryable saves reconnect via retry_save.
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
            retry_save(report, update_fields)
            applied_ids.add(entry_id)
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
