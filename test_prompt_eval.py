import json
from unittest.mock import patch

from prompt_eval import build_test_data, score_output, run_eval


def test_build_test_data_returns_incidents_and_ground_truth():
    incidents, expected = build_test_data()
    assert len(incidents) >= 6
    assert isinstance(expected, set)
    assert all(isinstance(rid, str) for rid in expected)


def test_build_test_data_includes_animal_and_non_animal():
    incidents, expected = build_test_data()
    nhtsa = [i for i in incidents if i.get("aaiid_data_source") == "nhtsa_incident_report"]
    oa = [i for i in incidents if i.get("aaiid_data_source") == "openalex_work"]
    assert len(nhtsa) >= 3
    assert len(oa) >= 1
    assert len(expected) >= 2
    assert len(expected) < len(incidents)  # some should be excluded


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
def test_run_eval_returns_metrics(mock_classify, tmp_path):
    output_csv = tmp_path / "animal_incidents_20260625_120000.csv"
    output_csv.write_text(
        "aaiid_data_source,json_blob,reasoning,confidence_score\n"
        'nhtsa_incident_report,{"Report ID":"EVAL-001"},duck,High\n'
        'nhtsa_incident_report,{"Report ID":"EVAL-003"},not animal,High\n'
    )
    mock_classify.return_value = str(output_csv)

    result = run_eval(output_dir=str(tmp_path))

    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert result["precision"] > 0
    assert result["recall"] > 0


def test_score_output_partial_match():
    generated = (
        "aaiid_data_source,json_blob,reasoning,confidence_score\n"
        'nhtsa_incident_report,{"Report ID":"A1"},duck,High\n'
        'nhtsa_incident_report,{"Report ID":"A2"},cat,High\n'
    )
    expected = {"A1", "A3"}
    result = score_output(generated, expected)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5

