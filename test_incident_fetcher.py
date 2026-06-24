import json
from unittest.mock import patch

import pytest
import requests

from incident_fetcher import parse_csv, write_json, fetch_csv, main


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


@patch("incident_fetcher.fetch_csv")
@patch("incident_fetcher.parse_csv")
@patch("incident_fetcher.write_json")
def test_main_integration(mock_write, mock_parse, mock_fetch):
    mock_fetch.return_value = "a,b\n1,2\n"
    mock_parse.return_value = [{"a": "1", "b": "2"}]
    main()
    mock_fetch.assert_called_once()
    mock_parse.assert_called_once_with("a,b\n1,2\n")
    mock_write.assert_called_once()
