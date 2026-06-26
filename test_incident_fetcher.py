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


@patch("incident_fetcher.requests.get")
@patch("incident_fetcher.datetime")
def test_main_writes_combined_output(mock_dt_module, mock_get, tmp_path, monkeypatch):
    mock_dt_module.now.return_value = datetime(2026, 6, 25)
    mock_dt_module.strptime = datetime.strptime

    csv_response = MagicMock()
    csv_response.text = "Incident Date,id,name\nJUN-2026,1,Alice\nMAY-2026,2,Bob\nAPR-2026,3,Charlie\n"

    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [{"id": "W1", "title": "Paper on animal behavior"}],
    }

    mock_get.side_effect = [csv_response, oa_response]

    output_path = tmp_path / "out.json"
    monkeypatch.setattr("incident_fetcher.OUTPUT_PATH", str(output_path))

    main()

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)

    csv_records = [r for r in data if r.get("aaiid_data_source") == "nhtsa_incident_report"]
    oa_records = [r for r in data if r.get("aaiid_data_source") == "openalex_work"]

    assert len(csv_records) == 1
    assert csv_records[0]["id"] == "1"
    assert csv_records[0]["name"] == "Alice"
    assert oa_records[0]["title"] == "Paper on animal behavior"


@patch("incident_fetcher.requests.get")
@patch("incident_fetcher.datetime")
def test_collect_data_passes_expanded_select_to_openalex(mock_dt_module, mock_get):
    mock_dt_module.now.return_value = datetime(2026, 6, 25)
    mock_dt_module.strptime = datetime.strptime

    csv_response = MagicMock()
    csv_response.text = "Incident Date\nJUN-2026\n"
    oa_response = MagicMock()
    oa_response.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.side_effect = [csv_response, oa_response]

    collect_data()

    oa_call = mock_get.call_args_list[1]
    params = oa_call[1]["params"]
    select_val = params.get("select", "")
    fields = select_val.split(",")
    assert "id" in fields
    assert "title" in fields
    assert "abstract_inverted_index" in fields
    assert "concepts" in fields
    assert "primary_location" in fields

@patch("incident_fetcher.subprocess.run")
def test_run_classification_calls_opencode_with_filename(mock_subprocess_run, tmp_path):
    incidents_file = tmp_path / "my_incidents.json"
    incidents_file.write_text("[]")
    prompt = generate_prompt(incidents_file.name)

    result_path = run_classification(str(incidents_file), prompt)

    assert mock_subprocess_run.called
    args = mock_subprocess_run.call_args[0][0]
    assert args[0] == "opencode"
    assert args[1] == "run"
    assert incidents_file.name in args[2]
    assert "animal_incidents_" in args[2]


def test_write_timestamped_csv_creates_file(tmp_path):
    csv_content = "a,b\n1,2\n"
    result_path = write_timestamped_csv(csv_content, str(tmp_path))
    assert result_path.startswith(str(tmp_path))
    assert "animal_incidents_" in result_path
    assert result_path.endswith(".csv")
    assert Path(result_path).exists()
    assert Path(result_path).read_text() == csv_content


@patch("incident_fetcher.subprocess.run")
@patch("incident_fetcher.datetime")
def test_run_classification_returns_known_path(mock_dt_module, mock_subprocess_run, tmp_path):
    mock_dt_module.now.return_value = datetime(2026, 6, 25, 12, 0, 0)

    incidents_file = tmp_path / "incidents.json"
    incidents_file.write_text("[]")
    prompt = "test prompt"

    result_path = run_classification(str(incidents_file), prompt)

    assert result_path == str(tmp_path / "animal_incidents_20260625_120000.csv")


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
@patch("incident_fetcher.datetime")
def test_collect_data_returns_combined_sources(mock_dt_module, mock_get):
    mock_dt_module.now.return_value = datetime(2026, 6, 25)
    mock_dt_module.strptime = datetime.strptime

    csv_response = MagicMock()
    csv_response.text = "Incident Date,id,name\nJUN-2026,1,Alice\nMAY-2026,2,Bob\nAPR-2026,3,Charlie\n"

    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [{"id": "W1", "title": "Paper on animal behavior"}],
    }

    mock_get.side_effect = [csv_response, oa_response]

    result = collect_data()

    csv_records = [r for r in result if r.get("aaiid_data_source") == "nhtsa_incident_report"]
    oa_records = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]

    assert len(csv_records) == 1
    assert csv_records[0]["id"] == "1"
    assert csv_records[0]["name"] == "Alice"
    assert len(oa_records) == 1
    assert oa_records[0]["id"] == "W1"
    assert result[-1]["aaiid_data_source"] == "openalex_work"


@patch("incident_fetcher.datetime")
def test_filter_by_date_keeps_current_month_only(mock_dt_module):
    mock_dt_module.now.return_value = datetime(2026, 6, 25)
    mock_dt_module.strptime = datetime.strptime
    data = [
        {"Incident Date": "JUN-2026", "id": "1"},
        {"Incident Date": "MAY-2026", "id": "2"},
        {"Incident Date": "APR-2026", "id": "3"},
    ]
    result = filter_by_date(data, months_back=1)
    assert len(result) == 1
    assert result[0]["id"] == "1"
