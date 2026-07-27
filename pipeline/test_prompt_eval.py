from unittest.mock import patch

from pipeline.incident_classifier import LLM_MODEL_DEFAULT
from pipeline.prompt_eval import build_test_data, run_eval, score_output


def test_score_output_perfect_match():
    records = [
        {"source_id": "A1", "json_blob": {"Report ID": "A1"}, "reasoning": "the duck",
         "confidence": "High"},
    ]
    result = score_output(records, {"A1"})
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_score_output_empty_records():
    result = score_output([], {"A1"})
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["false_negatives"] == ["A1"]


def test_build_test_data_ids_and_expected():
    incidents, expected = build_test_data()
    assert len(incidents) == 6
    assert expected == {"EVAL-001", "EVAL-002", "W-EVAL-001"}


@patch("pipeline.prompt_eval.classify")
def test_run_eval_uses_default_model_and_reports_it(mock_classify, tmp_path):
    # classify returns (resolved_model, judgments)
    mock_classify.return_value = (LLM_MODEL_DEFAULT, [])
    metrics = run_eval(output_dir=str(tmp_path))
    assert mock_classify.call_args.kwargs["model"] == LLM_MODEL_DEFAULT
    assert metrics["model"] == LLM_MODEL_DEFAULT


@patch("pipeline.prompt_eval.classify")
def test_run_eval_passes_explicit_model(mock_classify, tmp_path):
    mock_classify.return_value = ("foo/bar", [])
    metrics = run_eval(output_dir=str(tmp_path), model="foo/bar")
    assert mock_classify.call_args.kwargs["model"] == "foo/bar"
    assert metrics["model"] == "foo/bar"


@patch("pipeline.prompt_eval.classify")
def test_run_eval_scores_correctly_with_mocked_judgments(mock_classify, tmp_path):
    # incidents get entry_ids e0000..e0005 (built in order)
    # EVAL-001 → e0000 ; EVAL-002 → e0001 ; EVAL-003 → e0002 ; EVAL-004 → e0003
    # W-EVAL-001 → e0004 ; W-EVAL-002 → e0005
    mock_classify.return_value = (
        LLM_MODEL_DEFAULT,
        [
            {"entry_id": "e0000", "keep": True, "reasoning": "duck", "confidence": "High"},
            {"entry_id": "e0001", "keep": True, "reasoning": "dog", "confidence": "High"},
            {"entry_id": "e0004", "keep": True, "reasoning": "av+animals", "confidence": "Medium"},
        ],
    )
    metrics = run_eval(output_dir=str(tmp_path))
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
