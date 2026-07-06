import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from incident_fetcher import (
    parse_csv,
    write_json,
    fetch_csv,
    main,
    query_openalex,
    collect_data,
    filter_by_date,
    trim_nhtsa_fields,
    generate_prompt,
    run_classification,
    write_timestamped_csv,
    reconstruct_abstract,
)


NHTSA_KEEP = [
    "Report ID", "Report Version", "Reporting Entity",
    "Crash With", "Narrative", "City", "State",
    "Incident Date", "SV Precrash Speed (MPH)",
    "Highest Injury Severity Alleged",
    "CP Pre-Crash Movement", "SV Pre-Crash Movement",
    "Within ODD?", "aaiid_data_source",
]


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
        "Roadway-No Unusual Conditions": "Y",
        "Any Air Bags Deployed?": "No",
        "Latitude": "[REDACTED]",
    }
    result = trim_nhtsa_fields([row])
    assert len(result) == 1
    assert list(result[0].keys()) == NHTSA_KEEP
    assert "VIN" not in result[0]


def test_trim_nhtsa_fields_empty():
    assert trim_nhtsa_fields([]) == []


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
    # tmp_path is a built-in pytest fixture that creates a unique temp directory per test run
    output = tmp_path / "out.json"
    write_json(data, output)
    assert output.exists()
    with open(output) as f:
        assert json.load(f) == data


@patch("incident_fetcher.requests.get")
def test_fetch_csv_returns_text(mock_get):
    mock_response = mock_get.return_value
    mock_response.text = "a,b\n1,2\n"
    result = fetch_csv("http://example.com/data.csv")
    assert result == "a,b\n1,2\n"
    mock_get.assert_called_once_with("http://example.com/data.csv")


@patch("incident_fetcher.requests.get")
def test_fetch_csv_raises_on_http_error(mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Client Error"
    )
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_csv("http://example.com/bad.csv")

@patch("incident_fetcher.subprocess.run")
def test_run_classification_calls_opencode_with_filename(mock_subprocess_run, tmp_path):
    incidents_file = tmp_path / "my_incidents.json"
    incidents_file.write_text("[]")
    prompt = generate_prompt(incidents_file.name)

    result_path = run_classification(incidents_file, prompt)

    assert isinstance(result_path, Path)
    assert mock_subprocess_run.called
    args = mock_subprocess_run.call_args[0][0]
    assert args[0] == "opencode"
    assert args[1] == "run"
    assert incidents_file.name in args[2]
    assert "animal_incidents_" in args[2]


def test_write_timestamped_csv_creates_file(tmp_path):
    csv_content = "a,b\n1,2\n"
    result_path = write_timestamped_csv(csv_content, tmp_path)
    assert isinstance(result_path, Path)
    assert result_path.parent == tmp_path
    assert result_path.name.startswith("animal_incidents_")
    assert result_path.suffix == ".csv"
    assert result_path.exists()
    assert result_path.read_text() == csv_content


@patch("incident_fetcher.subprocess.run")
def test_run_classification_returns_known_path(mock_subprocess_run, tmp_path):
    incidents_file = tmp_path / "incidents.json"
    incidents_file.write_text("[]")
    prompt = "test prompt"

    result_path = run_classification(
        incidents_file, prompt, now=datetime(2026, 6, 25, 12, 0, 0)
    )

    assert result_path == tmp_path / "animal_incidents_20260625_120000.csv"


@patch("incident_fetcher.requests.get")
def test_query_openalex_returns_papers(mock_get):
    mock_response = mock_get.return_value
    mock_response.json.return_value = {
        "meta": {"count": 1, "page": 1, "per_page": 25},
        "results": [
            {
                "id": "https://openalex.org/W123",
                "title": "Animal cognition in autonomous systems",
            }
        ],
    }
    result = query_openalex({"per_page": 25})
    assert len(result) == 1
    assert result[0]["id"] == "https://openalex.org/W123"
    assert result[0]["title"] == "Animal cognition in autonomous systems"


@patch("incident_fetcher.requests.get")
def test_collect_data_returns_combined_sources(mock_get):
    csv_response = MagicMock()
    csv_response.text = "Incident Date,id,name\nJUN-2026,1,Alice\nMAY-2026,2,Bob\nAPR-2026,3,Charlie\n"

    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [{"id": "W1", "title": "Paper on animal behavior"}],
    }

    mock_get.side_effect = [csv_response, oa_response]

    result = collect_data(now=datetime(2026, 6, 25))

    csv_records = [r for r in result if r.get("aaiid_data_source") == "nhtsa_incident_report"]
    oa_records = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]

    assert len(csv_records) == 1
    assert csv_records[0]["id"] == "1"
    assert csv_records[0]["name"] == "Alice"
    assert len(oa_records) == 1
    assert oa_records[0]["id"] == "W1"
    assert result[-1]["aaiid_data_source"] == "openalex_work"


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
    data = [
        {"Incident Date": "   ", "id": "1"},
        {"Incident Date": "", "id": "2"},
    ]
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


def test_reconstruct_abstract():
    inv = {"the": [0, 3], "cat": [1], "and": [2], "dog": [4]}
    assert reconstruct_abstract(inv) == "the cat and the dog"


def test_reconstruct_abstract_none_returns_none():
    assert reconstruct_abstract(None) is None


def test_reconstruct_abstract_empty_dict_returns_none():
    assert reconstruct_abstract({}) is None


@patch("incident_fetcher.requests.get")
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
    assert len(papers) == 1
    assert papers[0]["abstract"] == "Autonomous vehicles and deer"
    # inverted index is removed after reconstruction
    assert "abstract_inverted_index" not in papers[0]


@patch("incident_fetcher.requests.get")
def test_collect_data_handles_missing_openalex_abstract(mock_get):
    csv_response = MagicMock()
    csv_response.text = "Incident Date\n"
    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [{"id": "W1", "title": "A paper"}],  # no abstract field
    }
    mock_get.side_effect = [csv_response, oa_response]

    result = collect_data(now=datetime(2026, 6, 25))

    papers = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]
    assert papers[0]["abstract"] is None
