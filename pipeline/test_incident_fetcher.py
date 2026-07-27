import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline.incident_fetcher import (
    OPENALEX_KEEP_FIELDS,
    _get_dotted,
    _set_dotted,
    assign_entry_ids,
    build_records,
    collect_data,
    fetch_csv,
    filter_by_date,
    parse_csv,
    query_openalex,
    reconstruct_abstract,
    trim_nhtsa_fields,
    trim_openalex_fields,
    write_json,
)


NHTSA_KEEP = [
    "Report ID", "Report Version", "Reporting Entity",
    "Crash With", "Narrative", "City", "State",
    "Incident Date", "SV Precrash Speed (MPH)",
    "Highest Injury Severity Alleged",
    "CP Pre-Crash Movement", "SV Pre-Crash Movement",
    "Within ODD?",
]


# ---------- trimming ----------


def test_trim_nhtsa_fields():
    row = {
        "Report ID": "123",
        "Report Version": "1",
        "Reporting Entity": "Waymo LLC",
        "Crash With": "Animal",
        "Narrative": "Hit a duck",
        "City": "Austin",
        "State": "TX",
        "Incident Date": "MAR-2026",
        "SV Precrash Speed (MPH)": "15",
        "Highest Injury Severity Alleged": "Property Damage",
        "CP Pre-Crash Movement": "Stopped",
        "SV Pre-Crash Movement": "Proceeding Straight",
        "Within ODD?": "Yes",
        "aaiid_data_source": "nhtsa_incident_report",
        "VIN": "SOMEVIN",
        "Weather - Clear": "Y",
    }
    result = trim_nhtsa_fields([row])
    assert len(result) == 1
    assert list(result[0].keys()) == NHTSA_KEEP
    assert "VIN" not in result[0]
    assert "aaiid_data_source" not in result[0]


def test_trim_nhtsa_fields_empty():
    assert trim_nhtsa_fields([]) == []


def test_get_dotted_reads_nested():
    d = {"a": {"b": {"c": 1}}}
    assert _get_dotted(d, "a.b.c") == 1
    assert _get_dotted(d, "a.b") == {"c": 1}


def test_get_dotted_missing_returns_default():
    assert _get_dotted({}, "a.b", default="X") == "X"
    assert _get_dotted({"a": 5}, "a.b", default=None) is None  # a is not a dict
    assert _get_dotted({"a": {"b": None}}, "a.b") is None  # present but None


def test_set_dotted_creates_intermediate_dicts():
    d: dict = {}
    _set_dotted(d, "a.b.c", 42)
    assert d == {"a": {"b": {"c": 42}}}


def test_set_dotted_overwrites_non_dict_intermediate():
    d = {"a": "not-a-dict"}
    _set_dotted(d, "a.b", 1)
    assert d == {"a": {"b": 1}}


def test_trim_openalex_fields_preserves_nesting():
    row = {
        "id": "https://openalex.org/W1",
        "title": "A paper",
        "display_name": "A paper",
        "publication_year": 2026,
        "publication_date": "2026-06-01",
        "abstract": "text",
        "primary_location": {
            "landing_page_url": "https://x.example/y",
            "pdf_url": "https://x.example/z.pdf",
            "irrelevant_field": "gone",
        },
        "doi": "should-be-dropped",
        "aaiid_data_source": "openalex_work",
    }
    trimmed = trim_openalex_fields([row])[0]
    assert trimmed["id"] == "https://openalex.org/W1"
    assert trimmed["title"] == "A paper"
    assert trimmed["primary_location"] == {
        "landing_page_url": "https://x.example/y",
        "pdf_url": "https://x.example/z.pdf",
    }
    assert "doi" not in trimmed
    assert "aaiid_data_source" not in trimmed


def test_trim_openalex_fields_omits_missing_paths():
    row = {"id": "W1", "title": "A paper"}
    trimmed = trim_openalex_fields([row])[0]
    assert trimmed == {"id": "W1", "title": "A paper"}
    assert "primary_location" not in trimmed


def test_trim_openalex_fields_empty():
    assert trim_openalex_fields([]) == []


# ---------- parse_csv / write_json / fetch_csv ----------


def test_parse_csv_empty():
    assert parse_csv("") == []


def test_parse_csv_with_data():
    csv_text = "name,age\nAlice,30\nBob,25\n"
    result = parse_csv(csv_text)
    assert result == [
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "25"},
    ]


def test_write_json_creates_file(tmp_path):
    data = [{"name": "Alice", "age": "30"}]
    output = tmp_path / "out.json"
    write_json(data, output)
    assert output.exists()
    with open(output) as f:
        assert json.load(f) == data


@patch("pipeline.incident_fetcher.requests.get")
def test_fetch_csv_returns_text(mock_get):
    mock_response = mock_get.return_value
    mock_response.text = "a,b\n1,2\n"
    result = fetch_csv("http://example.com/data.csv")
    assert result == "a,b\n1,2\n"
    mock_get.assert_called_once_with("http://example.com/data.csv")


@patch("pipeline.incident_fetcher.requests.get")
def test_fetch_csv_raises_on_http_error(mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error"
    )
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_csv("http://example.com/bad.csv")


# ---------- filter_by_date ----------


def test_filter_by_date_keeps_current_month_only():
    data = [
        {"Incident Date": "JUN-2026", "id": "1"},
        {"Incident Date": "MAY-2026", "id": "2"},
        {"Incident Date": "APR-2026", "id": "3"},
    ]
    result = filter_by_date(data, months_back=1, now=datetime(2026, 6, 25))
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_by_date_drops_missing_date():
    data = [
        {"Incident Date": "JUN-2026", "id": "1"},
        {"id": "2"},
    ]
    result = filter_by_date(data, months_back=1, now=datetime(2026, 6, 25))
    assert [r["id"] for r in result] == ["1"]


def test_filter_by_date_drops_blank_date():
    data = [{"Incident Date": "   ", "id": "1"}, {"Incident Date": "", "id": "2"}]
    result = filter_by_date(data, months_back=1, now=datetime(2026, 6, 25))
    assert result == []


def test_filter_by_date_drops_malformed_date_without_error():
    data = [
        {"Incident Date": "not-a-date", "id": "1"},
        {"Incident Date": "2026-06", "id": "2"},
        {"Incident Date": "JUN-2026", "id": "3"},
    ]
    result = filter_by_date(data, months_back=1, now=datetime(2026, 6, 25))
    assert [r["id"] for r in result] == ["3"]


def test_filter_by_date_months_back_wider_window():
    data = [
        {"Incident Date": "JUN-2026", "id": "jun"},
        {"Incident Date": "MAY-2026", "id": "may"},
        {"Incident Date": "APR-2026", "id": "apr"},
        {"Incident Date": "MAR-2026", "id": "mar"},
    ]
    result = filter_by_date(data, months_back=3, now=datetime(2026, 6, 25))
    assert [r["id"] for r in result] == ["jun", "may", "apr"]


# ---------- reconstruct_abstract ----------


def test_reconstruct_abstract():
    inv = {"the": [0, 3], "cat": [1], "and": [2], "dog": [4]}
    assert reconstruct_abstract(inv) == "the cat and the dog"


def test_reconstruct_abstract_none_returns_none():
    assert reconstruct_abstract(None) is None


def test_reconstruct_abstract_empty_dict_returns_none():
    assert reconstruct_abstract({}) is None


# ---------- collect_data ----------


@patch("pipeline.incident_fetcher.requests.get")
def test_query_openalex_returns_papers(mock_get):
    mock_response = mock_get.return_value
    mock_response.json.return_value = {
        "meta": {"count": 1, "page": 1, "per_page": 25},
        "results": [{"id": "https://openalex.org/W123", "title": "Animal cognition"}],
    }
    result = query_openalex({"per_page": 25})
    assert len(result) == 1
    assert result[0]["id"] == "https://openalex.org/W123"


# TODO: re-enable once date filtering of the NHTSA CSV is settled. This test
# assumes collect_data() date-filters CSV rows (expects MAY-2026 dropped), but
# that filter is currently commented out in incident_fetcher.collect_data
# because the CSV is a fixed Jun 2025-May 2026 dataset. Commented out for now.
# @patch("pipeline.incident_fetcher.requests.get")
# def test_collect_data_returns_combined_sources(mock_get):
#     csv_response = MagicMock()
#     csv_response.text = "Incident Date,id,name\nJUN-2026,1,Alice\nMAY-2026,2,Bob\n"
#     oa_response = MagicMock()
#     oa_response.json.return_value = {
#         "meta": {"count": 1},
#         "results": [{"id": "W1", "title": "Paper on animal behavior"}],
#     }
#     mock_get.side_effect = [csv_response, oa_response]
#
#     result = collect_data(now=datetime(2026, 6, 25))
#
#     csv_records = [r for r in result if r.get("aaiid_data_source") == "nhtsa_incident_report"]
#     oa_records = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]
#     assert len(csv_records) == 1
#     assert csv_records[0]["id"] == "1"
#     assert len(oa_records) == 1
#     assert oa_records[0]["id"] == "W1"


@patch("pipeline.incident_fetcher.requests.get")
def test_collect_data_reconstructs_openalex_abstract(mock_get):
    csv_response = MagicMock()
    csv_response.text = "Incident Date\n"
    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [
            {
                "id": "W1",
                "title": "A paper",
                "abstract_inverted_index": {
                    "Autonomous": [0], "vehicles": [1], "and": [2], "deer": [3]
                },
            }
        ],
    }
    mock_get.side_effect = [csv_response, oa_response]

    result = collect_data(now=datetime(2026, 6, 25))

    papers = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]
    assert papers[0]["abstract"] == "Autonomous vehicles and deer"
    assert "abstract_inverted_index" not in papers[0]


@patch("pipeline.incident_fetcher.requests.get")
def test_collect_data_handles_missing_openalex_abstract(mock_get):
    csv_response = MagicMock()
    csv_response.text = "Incident Date\n"
    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [{"id": "W1", "title": "A paper"}],
    }
    mock_get.side_effect = [csv_response, oa_response]

    result = collect_data(now=datetime(2026, 6, 25))

    papers = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]
    assert papers[0]["abstract"] is None


# ---------- assign_entry_ids ----------


def test_assign_entry_ids_adds_padded_ids():
    entries = [{"a": 1}, {"a": 2}, {"a": 3}]
    assign_entry_ids(entries)
    assert [e["entry_id"] for e in entries] == ["e0000", "e0001", "e0002"]


def test_assign_entry_ids_is_idempotent():
    entries = [{"entry_id": "e9999", "a": 1}, {"a": 2}]
    assign_entry_ids(entries)
    assert entries[0]["entry_id"] == "e9999"  # untouched
    assert entries[1]["entry_id"] == "e0000"  # newly assigned


# ---------- build_records ----------


def _judgment(entry_id, keep=True, reasoning="r", confidence="High"):
    return {
        "entry_id": entry_id,
        "keep": keep,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def test_build_records_empty_inputs_returns_empty():
    assert build_records([], [], model="m/x") == []


def test_build_records_returns_kept_rows_only():
    entries = [
        {"entry_id": "e0", "aaiid_data_source": "nhtsa_incident_report",
         "Report ID": "A1", "Crash With": "Animal", "Narrative": "duck"},
        {"entry_id": "e1", "aaiid_data_source": "nhtsa_incident_report",
         "Report ID": "A2", "Crash With": "Passenger Car"},
    ]
    judgments = [
        _judgment("e0", keep=True, reasoning="duck", confidence="High"),
        _judgment("e1", keep=False, reasoning="car crash", confidence="Low"),
    ]
    records = build_records(entries, judgments, model="m/x")
    assert len(records) == 1
    record = records[0]
    assert record["aaiid_data_source"] == "nhtsa_incident_report"
    assert record["entry_id"] == "e0"
    assert record["source_id"] == "A1"
    assert record["reasoning"] == "duck"
    assert record["confidence"] == "High"
    assert record["model"] == "m/x"
    blob = record["json_blob"]
    assert blob["Report ID"] == "A1"
    assert "aaiid_data_source" not in blob
    assert "entry_id" not in blob


def test_build_records_source_id_for_openalex_uses_id():
    entries = [{
        "entry_id": "e0",
        "aaiid_data_source": "openalex_work",
        "id": "https://openalex.org/W1",
        "title": "AV and deer",
        "abstract": "text",
        "primary_location": {
            "landing_page_url": "u1",
            "pdf_url": "u2",
            "junk": "x",
        },
        "doi": "dropped",
    }]
    judgments = [_judgment("e0")]
    records = build_records(entries, judgments, model="m/x")
    assert records[0]["source_id"] == "https://openalex.org/W1"
    blob = records[0]["json_blob"]
    assert blob["primary_location"] == {"landing_page_url": "u1", "pdf_url": "u2"}
    assert "doi" not in blob
    assert "junk" not in blob.get("primary_location", {})


def test_build_records_skips_judgments_with_no_matching_entry(capsys):
    entries = [{"entry_id": "e0", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A1"}]
    judgments = [_judgment("ghost"), _judgment("e0")]
    records = build_records(entries, judgments, model="m/x")
    assert len(records) == 1
    assert records[0]["entry_id"] == "e0"
    assert "unknown entry_id 'ghost'" in capsys.readouterr().out


def test_build_records_logs_unknown_entry_id_even_when_keep_false(capsys):
    """Membership check runs above the keep filter so hallucinated ids are
    always logged, even if the LLM said keep=False."""
    entries = [{"entry_id": "e0", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A1"}]
    judgments = [
        _judgment("ghost", keep=False, reasoning="not relevant", confidence="Low"),
    ]
    build_records(entries, judgments, model="m/x")
    assert "unknown entry_id 'ghost'" in capsys.readouterr().out


def test_build_records_logs_missing_entry_ids_report(capsys):
    """Entries with no matching judgment are summarized at the end."""
    entries = [
        {"entry_id": "e0", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A1"},
        {"entry_id": "e1", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A2"},
        {"entry_id": "e2", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A3"},
    ]
    # only e0 is judged; e1 and e2 are silently missing from the LLM output
    judgments = [_judgment("e0")]
    build_records(entries, judgments, model="m/x")
    out = capsys.readouterr().out
    assert "2 entry_id(s) had no valid judgment" in out
    assert "e1" in out
    assert "e2" in out


def test_build_records_no_missing_report_when_all_entries_judged(capsys):
    entries = [
        {"entry_id": "e0", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A1"},
        {"entry_id": "e1", "aaiid_data_source": "nhtsa_incident_report", "Report ID": "A2"},
    ]
    judgments = [_judgment("e0"), _judgment("e1", keep=False)]
    build_records(entries, judgments, model="m/x")
    out = capsys.readouterr().out
    assert "had no valid judgment" not in out


def test_build_records_preserves_json_blob_values():
    entries = [{
        "entry_id": "e0",
        "aaiid_data_source": "nhtsa_incident_report",
        "Report ID": "A1",
        "Narrative": 'She said "hi", then left.',
    }]
    judgments = [_judgment("e0")]
    records = build_records(entries, judgments, model="m/x")
    assert records[0]["json_blob"]["Narrative"] == 'She said "hi", then left.'






