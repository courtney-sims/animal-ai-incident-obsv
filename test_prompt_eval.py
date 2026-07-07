import json
from pathlib import Path
from unittest.mock import patch

from incident_fetcher import LLM_MODEL_DEFAULT
from prompt_eval import build_test_data, score_output, run_eval


def test_score_output_perfect_match():
    generated = (
        "aaiid_data_source,json_blob,reasoning,confidence_score\n"
        'nhtsa_incident_report,{"Report ID":"A1"},the duck,High\n'
    )
    expected = {"A1"}
    result = score_output(generated, expected)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


@patch("prompt_eval.run_classification")
def test_run_eval_uses_default_model_and_reports_it(mock_run, tmp_path):
    fake_csv = tmp_path / "out.csv"
    fake_csv.write_text("aaiid_data_source,json_blob\n")
    mock_run.return_value = fake_csv

    metrics = run_eval(output_dir=str(tmp_path))

    assert mock_run.call_args.kwargs["model"] == LLM_MODEL_DEFAULT
    assert metrics["model"] == LLM_MODEL_DEFAULT


@patch("prompt_eval.run_classification")
def test_run_eval_passes_explicit_model(mock_run, tmp_path):
    fake_csv = tmp_path / "out.csv"
    fake_csv.write_text("aaiid_data_source,json_blob\n")
    mock_run.return_value = fake_csv

    metrics = run_eval(output_dir=str(tmp_path), model="foo/bar")

    assert mock_run.call_args.kwargs["model"] == "foo/bar"
    assert metrics["model"] == "foo/bar"