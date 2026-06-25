import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from incident_fetcher import parse_csv, write_json, fetch_csv, main, query_openalex, collect_data


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
def test_main_writes_combined_output(mock_get, tmp_path, monkeypatch):
    csv_response = MagicMock()
    csv_response.text = "id,name\n1,Alice\n2,Bob\n"

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

    assert len(csv_records) == 2
    assert len(oa_records) == 1
    assert csv_records[0]["id"] == "1"
    assert oa_records[0]["title"] == "Paper on animal behavior"


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
    csv_response.text = "id,name\n1,Alice\n2,Bob\n"

    oa_response = MagicMock()
    oa_response.json.return_value = {
        "meta": {"count": 1},
        "results": [{"id": "W1", "title": "Paper on animal behavior"}],
    }

    mock_get.side_effect = [csv_response, oa_response]

    result = collect_data()

    csv_records = [r for r in result if r.get("aaiid_data_source") == "nhtsa_incident_report"]
    oa_records = [r for r in result if r.get("aaiid_data_source") == "openalex_work"]

    assert len(csv_records) == 2
    assert csv_records[0]["id"] == "1"
    assert csv_records[0]["name"] == "Alice"
    assert len(oa_records) == 1
    assert oa_records[0]["id"] == "W1"
    assert result[-1]["aaiid_data_source"] == "openalex_work"
