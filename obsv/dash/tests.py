"""Tests for the ``run_pipeline`` management command.

These exercise the full ingest -> classify -> extract flow with
``incident_fetcher.collect_data``, ``incident_classifier.classify`` and
``incident_classifier.extract_details`` mocked, so no network or subprocess
work runs. The mocked classify/extract fakes read the real intermediate JSON
the command writes to its (temp) workdir, so they see the exact pk-derived
``entry_id`` values the command produced.
"""

import contextlib
from datetime import datetime, timedelta
import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.db.utils import OperationalError
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from dash.models import IncidentReport, PipelineRun
from pipeline.incident_classifier import ClassificationOutputError
from pipeline.incident_fetcher import URL


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def nhtsa_entry(report_id="N1", crash_with="Animal", **extra):
    entry = {
        "aaiid_data_source": "nhtsa_incident_report",
        "Report ID": report_id,
        "Crash With": crash_with,
        "City": "Austin",
        "Incident Date": "MAR-2026",
        "Reporting Entity": "Waymo LLC",
        "Narrative": "the vehicle struck a deer",
    }
    entry.update(extra)
    return entry


def openalex_entry(work_id="https://openalex.org/W1", **extra):
    entry = {
        "aaiid_data_source": "openalex_work",
        "id": work_id,
        "display_name": "Autonomous vehicles and deer collisions",
        "title": "raw title",
        "publication_date": "2026-06-01",
        "primary_location": {"landing_page_url": "https://land.example/paper"},
        "abstract": "about AV and deer",
    }
    entry.update(extra)
    return entry


def _default_keep(entry):
    """Keep OpenAlex works and NHTSA rows whose Crash With is Animal."""
    return (
        entry.get("aaiid_data_source") == "openalex_work"
        or entry.get("Crash With") == "Animal"
    )


def full_detail(entry_id, **overrides):
    detail = {
        "entry_id": entry_id,
        "title": "LLM title",
        "description": "LLM description",
        "animal_type": "wild",
        "animal_species": "deer",
        "animal_count": 2,
        "harm_type": "collision",
        "harm_description": "fatal impact",
        "ai_system": "Waymo Driver",
        "country": "USA",
    }
    detail.update(overrides)
    return detail


# ---------------------------------------------------------------------------
# mock factories: read the command's real incidents/input JSON, return canned data
# ---------------------------------------------------------------------------


def fake_classify(seen=None, keep_fn=_default_keep, skip_fn=None):
    def _run(workdir, incidents_path, judgments_path, model=None):
        entries = json.loads(Path(incidents_path).read_text())
        if seen is not None:
            seen.append(entries)
        judgments = []
        for e in entries:
            if skip_fn and skip_fn(e):
                continue
            judgments.append({
                "entry_id": e["entry_id"],
                "keep": bool(keep_fn(e)),
                "reasoning": "because",
                "confidence": "High",
            })
        return (model or "test/classify"), judgments

    return _run


def fake_extract(seen=None, detail_fn=None, skip_fn=None):
    def _run(workdir, input_path, output_path, model=None):
        entries = json.loads(Path(input_path).read_text())
        if seen is not None:
            seen.append(entries)
        details = []
        for e in entries:
            if skip_fn and skip_fn(e):
                continue
            details.append(
                detail_fn(e) if detail_fn else full_detail(e["entry_id"])
            )
        return (model or "test/extract"), details

    return _run


class RunPipelineTestCase(TestCase):
    def _call(self, collect_return, classify_side, extract_side, workdir=None):
        """Run the command with all external calls mocked; return (mocks, out, err).

        A ``--workdir`` is always passed so tests never touch the persistent
        ``pipeline_runs/`` directory. When ``workdir`` is omitted a throwaway
        tempdir is used; pass an explicit path to seed/inspect resume files.
        """
        out, err = StringIO(), StringIO()
        if workdir is None:
            wd_ctx = tempfile.TemporaryDirectory()
        else:
            wd_ctx = contextlib.nullcontext(str(workdir))
        with wd_ctx as wd, mock.patch(
            "pipeline.incident_fetcher.collect_data", return_value=collect_return
        ), mock.patch(
            "pipeline.incident_classifier.classify", side_effect=classify_side
        ) as m_classify, mock.patch(
            "pipeline.incident_classifier.extract_details", side_effect=extract_side
        ) as m_extract, contextlib.redirect_stderr(err):
            call_command("run_pipeline", "--workdir", wd, stdout=out)
        return m_classify, m_extract, out.getvalue(), err.getvalue()

    # -- 1. fresh ingest + judgment + extraction ---------------------------

    def test_fresh_run_ingests_judges_and_extracts(self):
        extract_seen = []
        self._call(
            [nhtsa_entry("N1"), nhtsa_entry("N2", crash_with="Passenger Car"), openalex_entry()],
            fake_classify(),
            fake_extract(seen=extract_seen),
        )

        # kept NHTSA row: new -> llm_relevant -> pending in one run
        n1 = IncidentReport.objects.get(source_id="N1")
        self.assertEqual(n1.status, IncidentReport.StatusType.pending)
        # deterministic fields set at ingest
        self.assertEqual(n1.url, URL)
        self.assertEqual(n1.city, "Austin")
        self.assertEqual(n1.ai_system_manufacturer, "Waymo LLC")
        self.assertIsNotNone(n1.time_occurred)
        # judgment fields
        self.assertEqual(n1.llm_reasoning, "because")
        self.assertEqual(n1.confidence, IncidentReport.CONFIDENCE_MAP["High"])
        self.assertEqual(n1.llm_model, "test/classify")
        self.assertIsNotNone(n1.time_judged)
        # extraction fields (title was default "" -> filled)
        self.assertEqual(n1.title, "LLM title")
        self.assertEqual(n1.description, "LLM description")
        self.assertEqual(n1.animal_type, "wild")
        self.assertEqual(n1.animal_species, "deer")
        self.assertEqual(n1.animal_count, 2)
        self.assertEqual(n1.harm_type, "collision")
        self.assertEqual(n1.ai_system, "Waymo Driver")
        self.assertEqual(n1.country, "USA")
        self.assertIsNotNone(n1.time_hydrated)  # a.k.a. "time_extracted"
        self.assertIsNotNone(n1.time_ingested)

        # rejected NHTSA row: llm_rejected, never seen by extraction
        n2 = IncidentReport.objects.get(source_id="N2")
        self.assertEqual(n2.status, IncidentReport.StatusType.llm_rej)
        self.assertIsNotNone(n2.time_ingested)

        extracted_sources = {
            e.get("Report ID") or e.get("id")
            for batch in extract_seen for e in batch
        }
        self.assertNotIn("N2", extracted_sources)

        # kept OpenAlex row: deterministic title preserved over LLM title
        o1 = IncidentReport.objects.get(source_id="https://openalex.org/W1")
        self.assertEqual(o1.status, IncidentReport.StatusType.pending)
        self.assertEqual(o1.title, "Autonomous vehicles and deer collisions")
        self.assertEqual(o1.url, "https://land.example/paper")
        self.assertIsNotNone(o1.time_reported)
        self.assertIsNotNone(o1.time_ingested)

    # -- 2. rerun idempotency ----------------------------------------------

    def test_rerun_is_idempotent(self):
        entries = [nhtsa_entry("N1"), openalex_entry()]
        self._call(entries, fake_classify(), fake_extract())

        before = {r.source_id: (r.pk, r.status, r.time_ingested)
                  for r in IncidentReport.objects.all()}
        count_before = IncidentReport.objects.count()

        # second run with a changed raw field to prove raw_data refreshes
        entries2 = [nhtsa_entry("N1", Narrative="UPDATED narrative"), openalex_entry()]
        m_classify, m_extract, _out, _err = self._call(
            entries2, fake_classify(), fake_extract()
        )

        self.assertEqual(IncidentReport.objects.count(), count_before)  # no new rows
        m_classify.assert_not_called()  # nothing left `new`
        m_extract.assert_not_called()   # nothing left `llm_relevant`

        n1 = IncidentReport.objects.get(source_id="N1")
        self.assertEqual(n1.pk, before["N1"][0])                 # same row
        self.assertEqual(n1.status, before["N1"][1])             # status unchanged
        self.assertEqual(n1.time_ingested, before["N1"][2])      # ingest time preserved
        self.assertEqual(n1.raw_data["Narrative"], "UPDATED narrative")  # raw refreshed

    # -- 3. status preservation --------------------------------------------

    def test_rerun_preserves_human_status(self):
        entries = [nhtsa_entry("N1")]
        self._call(entries, fake_classify(), fake_extract())

        n1 = IncidentReport.objects.get(source_id="N1")
        n1.status = IncidentReport.StatusType.approved
        n1.save(update_fields=["status"])

        self._call(entries, fake_classify(), fake_extract())

        n1.refresh_from_db()
        self.assertEqual(n1.status, IncidentReport.StatusType.approved)

    # -- 4. partial judgment -----------------------------------------------

    def test_partial_judgment_reprocesses_only_unjudged(self):
        entries = [nhtsa_entry("N1"), openalex_entry()]
        is_openalex = lambda e: e.get("aaiid_data_source") == "openalex_work"

        # first run: classifier omits the OpenAlex entry entirely
        classify_seen = []
        self._call(
            entries,
            fake_classify(seen=classify_seen, skip_fn=is_openalex),
            fake_extract(),
        )
        o1 = IncidentReport.objects.get(source_id="https://openalex.org/W1")
        self.assertEqual(o1.status, IncidentReport.StatusType.new)  # stayed new

        # second run: only the still-`new` OpenAlex row is sent to classify
        classify_seen2 = []
        self._call(
            entries,
            fake_classify(seen=classify_seen2),
            fake_extract(),
        )
        self.assertEqual(len(classify_seen2), 1)
        sent = classify_seen2[0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["aaiid_data_source"], "openalex_work")

        o1.refresh_from_db()
        self.assertEqual(o1.status, IncidentReport.StatusType.pending)

    # -- 5. empty-queue short-circuits -------------------------------------

    def test_empty_queues_skip_classify_and_extract(self):
        # a pre-existing row that is neither `new` nor `llm_relevant`
        IncidentReport.objects.create(
            source="nhtsa_incident_report",
            source_id="OLD",
            raw_data={"Report ID": "OLD"},
            time_ingested=timezone.now(),
            status=IncidentReport.StatusType.approved,
        )
        m_classify, m_extract, out, _err = self._call(
            [], fake_classify(), fake_extract()
        )
        m_classify.assert_not_called()
        m_extract.assert_not_called()
        self.assertIn("No unjudged records", out)
        self.assertIn("No relevant records", out)

    # -- 6. missing natural key --------------------------------------------

    def test_entry_without_natural_key_is_skipped(self):
        m_classify, _m_extract, _out, err = self._call(
            [nhtsa_entry(report_id="")], fake_classify(), fake_extract()
        )
        self.assertEqual(IncidentReport.objects.count(), 0)
        self.assertIn("empty source_id", err)
        m_classify.assert_not_called()

    # -- 7. fill-only-if-default -------------------------------------------

    def test_fill_only_if_default(self):
        def details(e):
            # LLM returns a title for every row plus some unknown-markers
            return full_detail(
                e["entry_id"],
                title="LLM would-be title",
                animal_species="",      # unknown-marker
                animal_count=-1,         # unknown-marker
                animal_type="unknown",   # unknown-marker
                description="filled description",
            )

        self._call(
            [nhtsa_entry("N1"), openalex_entry()],
            fake_classify(),
            fake_extract(detail_fn=details),
        )

        # OpenAlex row has a deterministic title -> LLM title must not win
        o1 = IncidentReport.objects.get(source_id="https://openalex.org/W1")
        self.assertEqual(o1.title, "Autonomous vehicles and deer collisions")
        # default columns get filled from the LLM
        self.assertEqual(o1.description, "filled description")
        # unknown-markers equal the defaults -> columns left untouched
        self.assertEqual(o1.animal_species, "")
        self.assertEqual(o1.animal_count, -1)
        self.assertEqual(o1.animal_type, "unknown")

        # NHTSA row has no deterministic title -> LLM title applied
        n1 = IncidentReport.objects.get(source_id="N1")
        self.assertEqual(n1.title, "LLM would-be title")

    # -- 8. extraction failure isolation -----------------------------------

    def test_extraction_failure_does_not_roll_back_judgments(self):
        def boom(*args, **kwargs):
            raise RuntimeError("extraction exploded")

        # command must exit cleanly despite the extraction crash
        _m_classify, _m_extract, out, err = self._call(
            [nhtsa_entry("N1")], fake_classify(), boom
        )
        n1 = IncidentReport.objects.get(source_id="N1")
        # judgment from the same run is committed...
        self.assertEqual(n1.status, IncidentReport.StatusType.llm_rel)
        self.assertIsNotNone(n1.time_judged)
        # ...but extraction did not run
        self.assertIsNone(n1.time_hydrated)
        self.assertIn("extraction phase failed", err)

        # next run re-sends the row to a working extractor
        extract_seen = []
        self._call([nhtsa_entry("N1")], fake_classify(), fake_extract(seen=extract_seen))
        n1.refresh_from_db()
        self.assertEqual(n1.status, IncidentReport.StatusType.pending)
        resent = {e.get("Report ID") for batch in extract_seen for e in batch}
        self.assertIn("N1", resent)

    # -- 9. partial extraction ---------------------------------------------

    def test_partial_extraction_leaves_omitted_row_relevant(self):
        is_openalex = lambda e: e.get("aaiid_data_source") == "openalex_work"
        self._call(
            [nhtsa_entry("N1"), openalex_entry()],
            fake_classify(),
            fake_extract(skip_fn=is_openalex),
        )
        n1 = IncidentReport.objects.get(source_id="N1")
        o1 = IncidentReport.objects.get(source_id="https://openalex.org/W1")
        self.assertEqual(n1.status, IncidentReport.StatusType.pending)
        self.assertEqual(o1.status, IncidentReport.StatusType.llm_rel)

    # -- 10. missing judgments file is a hard failure ----------------------

    def test_missing_judgments_file_marks_run_failed(self):
        def missing_file(workdir, incidents_path, judgments_path, model=None):
            # Mirrors classify() when the LLM never produced the judgments
            # file: a hard failure, not a silent no-op.
            raise ClassificationOutputError(
                f"classify: judgments file was not produced at {judgments_path}"
            )

        with self.assertRaises(CommandError):
            self._call([nhtsa_entry("N1")], missing_file, fake_extract())

        # the run is logged failed, not completed
        run = PipelineRun.objects.order_by("-started_at").first()
        self.assertEqual(run.status, PipelineRun.Status.FAILED)
        self.assertIsNotNone(run.completed_at)

        # the ingested row stays `new` so the next run retries it
        n1 = IncidentReport.objects.get(source_id="N1")
        self.assertEqual(n1.status, IncidentReport.StatusType.new)

    # -- 11. structurally-invalid judgments file is a hard failure ---------

    def test_invalid_judgments_file_marks_run_failed(self):
        def invalid_file(workdir, incidents_path, judgments_path, model=None):
            # Mirrors classify() when the LLM wrote a structurally invalid file
            # (malformed JSON / non-list top-level): a hard failure.
            raise ClassificationOutputError(
                f"classify: judgments file at {judgments_path} is not valid JSON"
            )

        with self.assertRaises(CommandError):
            self._call([nhtsa_entry("N1")], invalid_file, fake_extract())

        run = PipelineRun.objects.order_by("-started_at").first()
        self.assertEqual(run.status, PipelineRun.Status.FAILED)
        self.assertIsNotNone(run.completed_at)

        n1 = IncidentReport.objects.get(source_id="N1")
        self.assertEqual(n1.status, IncidentReport.StatusType.new)

    # -- 12. resume from a pre-existing judgments file skips the LLM --------

    def test_resume_from_judgments_file_skips_classify(self):
        with tempfile.TemporaryDirectory() as wd:
            wd = Path(wd)
            r1 = _new_report("nhtsa_incident_report", "R1", {"Report ID": "R1"})
            r2 = _new_report("openalex_work", "R2", {"id": "R2"})

            judgments = [
                {"entry_id": f"e{r1.pk:04d}", "keep": True,
                 "reasoning": "kept", "confidence": "High"},
                {"entry_id": f"e{r2.pk:04d}", "keep": False,
                 "reasoning": "dropped", "confidence": "Low"},
            ]
            jpath = wd / "judgments_20260101_000000.json"
            jpath.write_text(json.dumps(judgments))
            (wd / "judgments_20260101_000000.json.model").write_text("seed/model")

            def boom_classify(*args, **kwargs):
                raise AssertionError("classify must not be called on resume")

            m_classify, _m_extract, out, _err = self._call(
                [], boom_classify, fake_extract(), workdir=wd
            )

        m_classify.assert_not_called()
        self.assertIn("Reusing existing judgments file", out)

        r1.refresh_from_db()
        r2.refresh_from_db()
        # kept row was judged from the file (model from the sidecar) and then
        # extracted to `pending`.
        self.assertEqual(r1.status, IncidentReport.StatusType.pending)
        self.assertEqual(r1.llm_model, "seed/model")
        self.assertEqual(r1.llm_reasoning, "kept")
        self.assertIsNotNone(r1.time_judged)
        # rejected row stays llm_rejected and is never extracted
        self.assertEqual(r2.status, IncidentReport.StatusType.llm_rej)
        self.assertIsNone(r2.time_hydrated)

    def test_resume_ignored_without_model_sidecar(self):
        # A judgments file with no .model sidecar must NOT be reused (llm_model
        # is never guessed); the LLM classify path runs instead.
        with tempfile.TemporaryDirectory() as wd:
            wd = Path(wd)
            r1 = _new_report("nhtsa_incident_report", "R1", {"Report ID": "R1"})
            jpath = wd / "judgments_20260101_000000.json"
            jpath.write_text(json.dumps([
                {"entry_id": f"e{r1.pk:04d}", "keep": True,
                 "reasoning": "kept", "confidence": "High"},
            ]))
            # no sidecar written

            m_classify, _m_extract, out, _err = self._call(
                [], fake_classify(), fake_extract(), workdir=wd
            )

        m_classify.assert_called_once()
        self.assertNotIn("Reusing existing judgments file", out)


# ---------------------------------------------------------------------------
# apply_judgments command + crash-resilient persistence
# ---------------------------------------------------------------------------


def _new_report(source, source_id, raw_data):
    return IncidentReport.objects.create(
        source=source,
        source_id=source_id,
        raw_data=raw_data,
        time_ingested=timezone.now(),
        status=IncidentReport.StatusType.new,
    )


def _write_judgments(path, judgments):
    Path(path).write_text(json.dumps(judgments))


class ApplyJudgmentsTestCase(TestCase):
    def _judgment(self, report, keep=True, reasoning="because", confidence="High"):
        return {
            "entry_id": f"e{report.pk:04d}",
            "keep": keep,
            "reasoning": reasoning,
            "confidence": confidence,
        }

    # -- 1. applies a valid file -------------------------------------------

    def test_applies_valid_file(self):
        r1 = _new_report("nhtsa_incident_report", "N1", {"Report ID": "N1"})
        r2 = _new_report("nhtsa_incident_report", "N2", {"Report ID": "N2"})
        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "judgments_x.json"
            _write_judgments(path, [
                self._judgment(r1, keep=True, confidence="High"),
                self._judgment(r2, keep=False, confidence="Low"),
            ])
            out = StringIO()
            call_command(
                "apply_judgments", judgments=str(path),
                model="test/model", stdout=out,
            )

        r1.refresh_from_db()
        r2.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.llm_rel)
        self.assertEqual(r1.llm_reasoning, "because")
        self.assertEqual(r1.confidence, IncidentReport.CONFIDENCE_MAP["High"])
        self.assertEqual(r1.llm_model, "test/model")
        self.assertIsNotNone(r1.time_judged)

        self.assertEqual(r2.status, IncidentReport.StatusType.llm_rej)
        self.assertEqual(r2.confidence, IncidentReport.CONFIDENCE_MAP["Low"])
        self.assertIsNone(r2.time_hydrated)

    # -- 2. idempotent second run ------------------------------------------

    def test_second_run_is_idempotent(self):
        r1 = _new_report("nhtsa_incident_report", "N1", {"Report ID": "N1"})
        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "judgments_x.json"
            _write_judgments(path, [self._judgment(r1, keep=True)])
            call_command("apply_judgments", judgments=str(path), model="m1")
            r1.refresh_from_db()
            first_judged = r1.time_judged

            out = StringIO()
            call_command(
                "apply_judgments", judgments=str(path), model="m2", stdout=out,
            )

        r1.refresh_from_db()
        # already-judged row was skipped; not re-judged / not re-modelled.
        self.assertEqual(r1.llm_model, "m1")
        self.assertEqual(r1.time_judged, first_judged)
        self.assertIn("1 already-judged skipped", out.getvalue())

    # -- 3. unknown entry_id skipped with warning --------------------------

    def test_unknown_entry_id_skipped(self):
        r1 = _new_report("nhtsa_incident_report", "N1", {"Report ID": "N1"})
        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "judgments_x.json"
            _write_judgments(path, [
                self._judgment(r1, keep=True),
                {"entry_id": "e9999", "keep": True,
                 "reasoning": "ghost", "confidence": "High"},
            ])
            out, err = StringIO(), StringIO()
            with contextlib.redirect_stderr(err):
                call_command(
                    "apply_judgments", judgments=str(path),
                    model="m", stdout=out,
                )

        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.llm_rel)
        self.assertIn("1 unknown", out.getvalue())
        self.assertIn("unknown entry_id", err.getvalue())

    # -- 4. missing file -> CommandError -----------------------------------

    def test_missing_file_raises(self):
        with self.assertRaises(CommandError):
            call_command(
                "apply_judgments",
                judgments="/nonexistent/judgments.json",
                model="m",
            )

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "bad.json"
            path.write_text("{not json")
            with self.assertRaises(CommandError):
                call_command("apply_judgments", judgments=str(path), model="m")

    # -- 5. dry-run writes nothing -----------------------------------------

    def test_dry_run_writes_nothing(self):
        r1 = _new_report("nhtsa_incident_report", "N1", {"Report ID": "N1"})
        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "judgments_x.json"
            _write_judgments(path, [self._judgment(r1, keep=True)])
            out = StringIO()
            call_command(
                "apply_judgments", judgments=str(path),
                model="m", dry_run=True, stdout=out,
            )

        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.new)
        self.assertIsNone(r1.time_judged)
        self.assertIn("dry-run", out.getvalue())

    # -- 6. partial-progress preservation on a mid-loop drop ---------------

    def test_partial_progress_preserved_on_operational_error(self):
        from dash import judgment_persistence

        r1 = _new_report("nhtsa_incident_report", "N1", {"Report ID": "N1"})
        r2 = _new_report("nhtsa_incident_report", "N2", {"Report ID": "N2"})
        r3 = _new_report("nhtsa_incident_report", "N3", {"Report ID": "N3"})

        real_retry_save = judgment_persistence.retry_save
        state = {"n": 0}

        def flaky(report, update_fields, **kwargs):
            state["n"] += 1
            if state["n"] == 3:
                # simulate a connection drop that even retries can't recover
                raise OperationalError("connection dropped")
            return real_retry_save(report, update_fields)

        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "judgments_x.json"
            _write_judgments(path, [
                self._judgment(r1, keep=True),
                self._judgment(r2, keep=False),
                self._judgment(r3, keep=True),
            ])
            with mock.patch(
                "dash.judgment_persistence.retry_save", side_effect=flaky
            ):
                with self.assertRaises(OperationalError):
                    call_command(
                        "apply_judgments", judgments=str(path), model="m"
                    )

        r1.refresh_from_db()
        r2.refresh_from_db()
        r3.refresh_from_db()
        # the two rows saved before the drop are committed...
        self.assertEqual(r1.status, IncidentReport.StatusType.llm_rel)
        self.assertIsNotNone(r1.time_judged)
        self.assertEqual(r2.status, IncidentReport.StatusType.llm_rej)
        self.assertIsNotNone(r2.time_judged)
        # ...the row whose save failed remains `new` for the next run to retry
        self.assertEqual(r3.status, IncidentReport.StatusType.new)
        self.assertIsNone(r3.time_judged)


# ---------------------------------------------------------------------------
# public frontend: incident list + detail views
# ---------------------------------------------------------------------------


def make_incident(source_id, status=IncidentReport.StatusType.approved, **fields):
    """Create an IncidentReport with sensible defaults for the view tests."""
    data = {
        "source": IncidentReport.SourceType.NHTSA,
        "source_id": source_id,
        "raw_data": {},
        "time_ingested": timezone.now(),
        "status": status,
    }
    data.update(fields)
    return IncidentReport.objects.create(**data)


class IncidentListViewTests(TestCase):
    def setUp(self):
        self.url = reverse("incident_list")

    def test_default_shows_only_approved(self):
        approved = make_incident("A1", title="Approved incident")
        make_incident("N1", status=IncidentReport.StatusType.new, title="New incident")
        make_incident(
            "P1", status=IncidentReport.StatusType.pending, title="Pending incident"
        )

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        titles = [r["incident"].pk for r in resp.context["rows"]]
        self.assertEqual(titles, [approved.pk])
        self.assertContains(resp, "Approved incident")
        self.assertNotContains(resp, "New incident")

    def test_status_param_overrides_default(self):
        make_incident("A1", title="Approved incident")
        new = make_incident(
            "N1", status=IncidentReport.StatusType.new, title="New incident"
        )

        with override_settings(DEBUG=True):
            resp = self.client.get(
                self.url, {"status": IncidentReport.StatusType.new}
            )
        self.assertEqual(resp.status_code, 200)
        pks = [r["incident"].pk for r in resp.context["rows"]]
        self.assertEqual(pks, [new.pk])
        self.assertContains(resp, "New incident")
        self.assertNotContains(resp, "Approved incident")

    def test_status_field_dev_only(self):
        make_incident("A1", title="Approved incident")
        with override_settings(DEBUG=True):
            resp_dev = self.client.get(self.url)
        self.assertContains(resp_dev, 'name="status"')  # visible in dev

        resp_prod = self.client.get(self.url)
        self.assertNotContains(resp_prod, 'name="status"')

    def test_status_param_ignored_in_prod(self):
        approved = make_incident("A1", title="Approved incident")
        make_incident(
            "N1", status=IncidentReport.StatusType.new, title="New incident"
        )

        resp = self.client.get(self.url, {"status": IncidentReport.StatusType.new})
        self.assertEqual(resp.status_code, 200)
        pks = [r["incident"].pk for r in resp.context["rows"]]
        self.assertEqual(pks, [approved.pk])
        self.assertContains(resp, "Approved incident")
        self.assertNotContains(resp, "New incident")

    def test_count_line_shows_metrics_in_dev_only(self):
        make_incident("A1", title="Approved incident")
        make_incident("N1", status=IncidentReport.StatusType.new, title="New incident")

        with override_settings(DEBUG=True):
            resp_dev = self.client.get(self.url)
        self.assertContains(resp_dev, "1 approved of 2 reports · 1 shown")

        resp_prod = self.client.get(self.url)
        self.assertContains(resp_prod, "Showing 1 incident")
        self.assertNotContains(resp_prod, "approved of")

    def test_q_matches_title_description_and_narrative(self):
        by_title = make_incident("T1", title="Elephant collision")
        by_desc = make_incident("D1", title="Row two", description="a rare pangolin")
        by_narr = make_incident(
            "R1", title="Row three", raw_data={"Narrative": "struck a wombat"}
        )
        make_incident("X1", title="Unrelated", description="nothing here")

        for term, expected in [
            ("elephant", by_title),
            ("pangolin", by_desc),
            ("wombat", by_narr),
        ]:
            resp = self.client.get(self.url, {"q": term})
            pks = [r["incident"].pk for r in resp.context["rows"]]
            self.assertEqual(pks, [expected.pk], term)

    def test_source_filter(self):
        nhtsa = make_incident("N1", source=IncidentReport.SourceType.NHTSA)
        make_incident("O1", source=IncidentReport.SourceType.OPENALEX)

        resp = self.client.get(
            self.url, {"source": IncidentReport.SourceType.NHTSA}
        )
        pks = [r["incident"].pk for r in resp.context["rows"]]
        self.assertEqual(pks, [nhtsa.pk])

    def test_animal_type_filter(self):
        wild = make_incident("W1", animal_type=IncidentReport.AnimalType.wild)
        make_incident("F1", animal_type=IncidentReport.AnimalType.farmed)

        resp = self.client.get(
            self.url, {"animal_type": IncidentReport.AnimalType.wild}
        )
        pks = [r["incident"].pk for r in resp.context["rows"]]
        self.assertEqual(pks, [wild.pk])

    def test_date_range_filter(self):
        early = make_incident(
            "E1", time_occurred=timezone.make_aware(datetime(2026, 1, 15))
        )
        mid = make_incident(
            "M1", time_occurred=timezone.make_aware(datetime(2026, 3, 15))
        )
        late = make_incident(
            "L1", time_occurred=timezone.make_aware(datetime(2026, 6, 15))
        )

        resp = self.client.get(
            self.url, {"date_from": "2026-02-01", "date_to": "2026-04-01"}
        )
        pks = {r["incident"].pk for r in resp.context["rows"]}
        self.assertEqual(pks, {mid.pk})
        self.assertNotIn(early.pk, pks)
        self.assertNotIn(late.pk, pks)

    def test_pagination_second_page_and_out_of_range(self):
        for i in range(30):
            make_incident(f"P{i:02d}", title=f"Incident {i:02d}")

        resp = self.client.get(self.url)
        self.assertEqual(len(resp.context["rows"]), 25)

        resp2 = self.client.get(self.url, {"page": 2})
        self.assertEqual(len(resp2.context["rows"]), 5)
        self.assertEqual(resp2.context["page_obj"].number, 2)

        # out-of-range -> clamp to last page
        resp3 = self.client.get(self.url, {"page": 999})
        self.assertEqual(resp3.context["page_obj"].number, 2)

    def test_pagination_links_preserve_query_params(self):
        for i in range(30):
            make_incident(
                f"S{i:02d}",
                source=IncidentReport.SourceType.NHTSA,
                title=f"Incident {i:02d}",
            )

        resp = self.client.get(
            self.url, {"source": IncidentReport.SourceType.NHTSA}
        )
        content = resp.content.decode()
        self.assertIn("source=nhtsa_incident_report&page=2", content)

    def test_empty_state(self):
        resp = self.client.get(self.url)
        self.assertContains(resp, "No approved incidents yet")


class IncidentDetailViewTests(TestCase):
    def test_detail_renders_for_approved(self):
        inc = make_incident(
            "A1",
            title="Approved incident",
            description="details here",
            url="https://example.com/report",
        )
        resp = self.client.get(reverse("incident_detail", args=[inc.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Approved incident")
        self.assertContains(resp, "details here")
        self.assertContains(resp, "https://example.com/report")

    def test_detail_404_for_missing_pk(self):
        resp = self.client.get(reverse("incident_detail", args=[999999]))
        self.assertEqual(resp.status_code, 404)

    def test_detail_404_for_non_approved(self):
        inc = make_incident("P1", status=IncidentReport.StatusType.pending)
        resp = self.client.get(reverse("incident_detail", args=[inc.pk]))
        self.assertEqual(resp.status_code, 404)

    def test_detail_raw_data_fallbacks(self):
        inc = make_incident(
            "R1",
            title="",
            description="",
            raw_data={
                "Crash With": "Deer",
                "City": "Austin",
                "Narrative": "vehicle struck a deer",
                "Reporting Entity": "Waymo LLC",
            },
        )
        resp = self.client.get(reverse("incident_detail", args=[inc.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Deer in Austin")
        self.assertContains(resp, "vehicle struck a deer")


# ---------------------------------------------------------------------------
# admin: incident review (approve / reject)
# ---------------------------------------------------------------------------


from django.contrib.auth import get_user_model  # noqa: E402


class IncidentReviewAdminTests(TestCase):
    CHANGELIST = "/admin/dash/incidentreport/"

    def setUp(self):
        self.user = get_user_model().objects.create_superuser(
            username="reviewer", email="r@example.com", password="pw12345"
        )
        self.client.force_login(self.user)

    def _pending(self, source_id="P1", **fields):
        return make_incident(
            source_id, status=IncidentReport.StatusType.pending, **fields
        )

    def _review_url(self, pk):
        return f"/admin/dash/incidentreport/{pk}/review/"

    # -- changelist scope --------------------------------------------------

    def test_changelist_lists_only_pending(self):
        pending = self._pending("P1", title="Pending one")
        make_incident("A1", title="Approved one")  # approved by default
        make_incident("N1", status=IncidentReport.StatusType.new, title="New one")

        resp = self.client.get(self.CHANGELIST)
        self.assertEqual(resp.status_code, 200)
        cl = resp.context["cl"]
        pks = {obj.pk for obj in cl.result_list}
        self.assertEqual(pks, {pending.pk})

    # -- bulk actions ------------------------------------------------------

    def test_bulk_approve(self):
        r1 = self._pending("P1")
        r2 = self._pending("P2")
        resp = self.client.post(
            self.CHANGELIST,
            {"action": "approve_selected", "_selected_action": [r1.pk, r2.pk]},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        for r in (r1, r2):
            r.refresh_from_db()
            self.assertEqual(r.status, IncidentReport.StatusType.approved)
            self.assertEqual(r.reviewer, self.user)
            self.assertIsNotNone(r.time_reviewed)

    def test_bulk_reject(self):
        r1 = self._pending("P1")
        self.client.post(
            self.CHANGELIST,
            {"action": "reject_selected", "_selected_action": [r1.pk]},
            follow=True,
        )
        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.rejected)
        self.assertEqual(r1.reviewer, self.user)
        self.assertIsNotNone(r1.time_reviewed)

    # -- per-row review page -----------------------------------------------

    def test_review_get_renders_for_pending(self):
        r1 = self._pending("P1", title="Pending one")
        resp = self.client.get(self._review_url(r1.pk))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Pending one")
        self.assertContains(resp, "csrfmiddlewaretoken")

    def test_review_get_404_for_non_pending(self):
        approved = make_incident("A1")
        resp = self.client.get(self._review_url(approved.pk))
        self.assertEqual(resp.status_code, 404)

    def _review_form_data(self, incident, **overrides):
        data = {
            "title": incident.title,
            "description": incident.description,
            "url": incident.url,
            "animal_type": incident.animal_type,
            "animal_species": incident.animal_species,
            "animal_count": incident.animal_count,
            "harm_type": incident.harm_type,
            "harm_description": incident.harm_description,
            "ai_system": incident.ai_system,
            "ai_system_manufacturer": incident.ai_system_manufacturer,
            "city": incident.city,
            "country": incident.country,
            "status": incident.status,
            "confidence": incident.confidence,
            "llm_reasoning": incident.llm_reasoning,
            "llm_model": incident.llm_model,
            "reviewer": "",
        }
        data.update(overrides)
        return data

    def test_review_approve(self):
        r1 = self._pending("P1", title="Pending one")
        data = self._review_form_data(r1, title="Edited title", approve="Approve")
        resp = self.client.post(self._review_url(r1.pk), data)
        self.assertRedirects(resp, self.CHANGELIST)
        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.approved)
        self.assertEqual(r1.title, "Edited title")
        self.assertEqual(r1.reviewer, self.user)
        self.assertIsNotNone(r1.time_reviewed)

    def test_review_reject(self):
        r1 = self._pending("P1")
        data = self._review_form_data(r1, reject="Reject")
        resp = self.client.post(self._review_url(r1.pk), data)
        self.assertRedirects(resp, self.CHANGELIST)
        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.rejected)
        self.assertEqual(r1.reviewer, self.user)
        self.assertIsNotNone(r1.time_reviewed)

    def test_review_save_keeps_pending(self):
        r1 = self._pending("P1", title="Pending one")
        data = self._review_form_data(r1, title="Just edited", save="Save")
        resp = self.client.post(self._review_url(r1.pk), data)
        self.assertRedirects(resp, self.CHANGELIST)
        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.pending)
        self.assertEqual(r1.title, "Just edited")
        # a plain save must not stamp reviewer/time_reviewed
        self.assertIsNone(r1.reviewer)
        self.assertIsNone(r1.time_reviewed)

    # -- permissions -------------------------------------------------------

    def test_anonymous_denied(self):
        self.client.logout()
        r1 = self._pending("P1")
        for url in (self.CHANGELIST, self._review_url(r1.pk)):
            resp = self.client.get(url)
            self.assertIn(resp.status_code, (301, 302))
            self.assertIn("/admin/login/", resp.url)

    def test_non_staff_denied(self):
        get_user_model().objects.create_user(
            username="plain", password="pw12345"
        )
        self.client.logout()
        self.client.login(username="plain", password="pw12345")
        r1 = self._pending("P1")
        resp = self.client.get(self._review_url(r1.pk))
        self.assertIn(resp.status_code, (301, 302))
        self.assertIn("/admin/login/", resp.url)

    # -- persistence vs pipeline rerun -------------------------------------

    def test_admin_edits_survive_pipeline_rerun(self):
        # An approved, human-edited row must not be reverted by a rerun: the
        # pipeline preserves human status and fills only default fields.
        r1 = self._pending("N1", raw_data={"Report ID": "N1"})
        data = self._review_form_data(r1, title="Human title", approve="Approve")
        self.client.post(self._review_url(r1.pk), data)
        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.approved)

        # simulate a pipeline rerun over the same natural key
        with tempfile.TemporaryDirectory() as wd, mock.patch(
            "pipeline.incident_fetcher.collect_data",
            return_value=[nhtsa_entry("N1")],
        ), mock.patch(
            "pipeline.incident_classifier.classify", side_effect=fake_classify()
        ), mock.patch(
            "pipeline.incident_classifier.extract_details",
            side_effect=fake_extract(),
        ):
            call_command("run_pipeline", "--workdir", wd, stdout=StringIO())

        r1.refresh_from_db()
        self.assertEqual(r1.status, IncidentReport.StatusType.approved)
        self.assertEqual(r1.title, "Human title")
