"""Tests for the ``run_pipeline`` management command.

These exercise the full ingest -> classify -> extract flow with
``incident_fetcher.collect_data``, ``incident_classifier.classify`` and
``incident_classifier.extract_details`` mocked, so no network or subprocess
work runs. The mocked classify/extract fakes read the real intermediate JSON
the command writes to its (temp) workdir, so they see the exact pk-derived
``entry_id`` values the command produced.
"""

import contextlib
import json
from io import StringIO
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from dash.models import IncidentReport
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
    def _call(self, collect_return, classify_side, extract_side):
        """Run the command with all external calls mocked; return (mocks, out, err)."""
        out, err = StringIO(), StringIO()
        with mock.patch(
            "pipeline.incident_fetcher.collect_data", return_value=collect_return
        ), mock.patch(
            "pipeline.incident_classifier.classify", side_effect=classify_side
        ) as m_classify, mock.patch(
            "pipeline.incident_classifier.extract_details", side_effect=extract_side
        ) as m_extract, contextlib.redirect_stderr(err):
            call_command("run_pipeline", stdout=out)
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
