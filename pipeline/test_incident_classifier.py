import json

from pathlib import Path
from unittest.mock import MagicMock, patch

from incident_classifier import (
    LLM_MODEL_DEFAULT,
    classify,
    generate_prompt,
    parse_judgments,
)


def _write_judgments(path: Path, judgments) -> None:
    path.write_text(json.dumps(judgments))


def _make_fake_subprocess(judgments, judgments_path: Path):
    """Return a subprocess.run side_effect that writes ``judgments`` to ``judgments_path``.

    Used by classify tests to simulate the LLM writing its JSON output to a
    path Python controls.
    """
    def fake_run(*args, **kwargs):
        _write_judgments(judgments_path, judgments)
        return MagicMock(returncode=0)
    return fake_run


# ---------- generate_prompt ----------


def test_generate_prompt_mentions_both_filenames():
    prompt = generate_prompt("my_incidents.json", "my_judgments.json")
    assert "my_incidents.json" in prompt
    assert "my_judgments.json" in prompt
    assert "entry_id" in prompt


# ---------- classify (LLM invocation + return shape) ----------


@patch("incident_classifier.subprocess.run")
def test_classify_returns_model_and_judgments(mock_run, tmp_path):
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess(
        [{"entry_id": "e0000", "keep": True, "reasoning": "duck", "confidence": "High"}],
        judgments_path,
    )

    result = classify(tmp_path, incidents_path, judgments_path, model="m/x")

    assert result == (
        "m/x",
        [{"entry_id": "e0000", "keep": True, "reasoning": "duck", "confidence": "High"}],
    )


@patch("incident_classifier.subprocess.run")
def test_classify_passes_model_flag(mock_run, tmp_path):
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess([], judgments_path)

    classify(tmp_path, incidents_path, judgments_path, model="foo/bar")

    args = mock_run.call_args[0][0]
    assert "-m" in args
    assert args[args.index("-m") + 1] == "foo/bar"


@patch("incident_classifier.subprocess.run")
def test_classify_uses_default_model_when_none(mock_run, tmp_path):
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess([], judgments_path)

    model, _ = classify(tmp_path, incidents_path, judgments_path)

    assert model == LLM_MODEL_DEFAULT
    args = mock_run.call_args[0][0]
    assert args[args.index("-m") + 1] == LLM_MODEL_DEFAULT


# ---------- parse_judgments: file / JSON / top-level shape ----------


def test_parse_judgments_returns_empty_when_file_missing(tmp_path, capsys):
    result = parse_judgments(tmp_path / "does_not_exist.json")
    assert result == []
    assert "judgments file missing" in capsys.readouterr().out


def test_parse_judgments_returns_empty_on_invalid_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("this is not json{")
    result = parse_judgments(path)
    assert result == []
    assert "not valid JSON" in capsys.readouterr().out


def test_parse_judgments_returns_empty_when_top_level_not_list(tmp_path, capsys):
    path = tmp_path / "obj.json"
    path.write_text('{"judgments": []}')
    result = parse_judgments(path)
    assert result == []
    assert "top-level is dict" in capsys.readouterr().out


# ---------- parse_judgments: per-record shape ----------


def test_parse_judgments_appends_valid_records(tmp_path):
    path = tmp_path / "j.json"
    _write_judgments(path, [
        {"entry_id": "e0", "keep": True,  "reasoning": "duck", "confidence": "High"},
        {"entry_id": "e1", "keep": False, "reasoning": "no",   "confidence": "Low"},
    ])
    result = parse_judgments(path)
    assert result == [
        {"entry_id": "e0", "keep": True,  "reasoning": "duck", "confidence": "High"},
        {"entry_id": "e1", "keep": False, "reasoning": "no",   "confidence": "Low"},
    ]


def test_parse_judgments_skips_non_dict_element(tmp_path, capsys):
    path = tmp_path / "j.json"
    _write_judgments(path, [
        "not a dict",
        {"entry_id": "e0", "keep": True, "reasoning": "r", "confidence": "High"},
    ])
    result = parse_judgments(path)
    assert [j["entry_id"] for j in result] == ["e0"]
    assert "expected dict" in capsys.readouterr().out


def test_parse_judgments_skips_records_with_bad_shape(tmp_path, capsys):
    path = tmp_path / "j.json"
    _write_judgments(path, [
        {"entry_id": "e0", "keep": "yes", "reasoning": "r", "confidence": "High"},   # keep is str
        {"entry_id": "e1", "keep": True,  "reasoning": "r", "confidence": "SUPER"},  # bad enum
        {"entry_id": "e2", "keep": True,  "reasoning": "r", "confidence": "High"},   # valid
    ])
    result = parse_judgments(path)
    assert [j["entry_id"] for j in result] == ["e2"]
    out = capsys.readouterr().out
    assert "keep must be bool" in out
    assert "confidence must be one of" in out


# ---------- parse_judgments: dedupe (first wins) ----------


def test_parse_judgments_dedupes_repeated_entry_ids_first_wins(tmp_path, capsys):
    path = tmp_path / "j.json"
    _write_judgments(path, [
        {"entry_id": "e0", "keep": True,  "reasoning": "first",  "confidence": "High"},
        {"entry_id": "e0", "keep": False, "reasoning": "second", "confidence": "Low"},
    ])
    result = parse_judgments(path)
    assert len(result) == 1
    assert result[0]["reasoning"] == "first"


def test_parse_judgments_dedupe_prints_both_records(tmp_path, capsys):
    path = tmp_path / "j.json"
    _write_judgments(path, [
        {"entry_id": "e0", "keep": True,  "reasoning": "first",  "confidence": "High"},
        {"entry_id": "e0", "keep": False, "reasoning": "second", "confidence": "Low"},
    ])
    parse_judgments(path)
    out = capsys.readouterr().out
    assert "duplicate entry_id" in out
    # Both kept and discarded records should be visible in the log for
    # debugging.
    assert "first" in out
    assert "second" in out
    assert "kept:" in out
    assert "discarded:" in out
