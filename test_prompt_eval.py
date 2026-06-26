import json
from unittest.mock import patch

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