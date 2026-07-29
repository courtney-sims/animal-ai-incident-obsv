from unittest.mock import patch

import pytest

from pipeline.main import main, parse_cli_args

@patch("pipeline.main.incident_fetcher.write_json")
@patch("pipeline.main.run_pipeline")
def test_main_forwards_model_from_cli(mock_pipeline, mock_write):
    mock_pipeline.return_value = ([], "foo/bar")
    main(["model=foo/bar"])
    assert mock_pipeline.call_count == 1
    assert mock_pipeline.call_args.kwargs["model"] == "foo/bar"

@patch("pipeline.main.incident_fetcher.write_json")
@patch("pipeline.main.run_pipeline")
def test_main_defaults_model_to_none_when_absent(mock_pipeline, mock_write):
    mock_pipeline.return_value = ([], "model")
    main([])
    assert mock_pipeline.call_count == 1
    assert mock_pipeline.call_args.kwargs["model"] is None

@patch("pipeline.main.incident_fetcher.write_json")
@patch("pipeline.main.run_pipeline")
def test_main_writes_records_file(mock_pipeline, mock_write):
    records = [{"entry_id": "e0001"}]
    mock_pipeline.return_value = (records, "foo/bar")
    main([])
    # main writes the built records to records_<timestamp>.json in the workdir.
    assert mock_write.call_count == 1
    written_records, written_path = mock_write.call_args.args
    assert written_records == records
    assert written_path.name.startswith("records_")
    assert written_path.suffix == ".json"

def test_parse_cli_args_empty():
    assert parse_cli_args([]) == {}


def test_parse_cli_args_model():
    assert parse_cli_args(["model=foo/bar"]) == {"model": "foo/bar"}


def test_parse_cli_args_unknown_key_exits():
    with pytest.raises(SystemExit):
        parse_cli_args(["colour=blue"])


def test_parse_cli_args_missing_equals_exits():
    with pytest.raises(SystemExit):
        parse_cli_args(["model"])