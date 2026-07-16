from unittest.mock import patch

import pytest

from main import main, parse_cli_args

@patch("main.run_pipeline")
def test_main_forwards_model_from_cli(mock_pipeline):
    main(["model=foo/bar"])
    mock_pipeline.assert_called_once_with(model="foo/bar")

@patch("main.run_pipeline")
def test_main_defaults_model_to_none_when_absent(mock_pipeline):
    main([])
    mock_pipeline.assert_called_once_with(model=None)

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