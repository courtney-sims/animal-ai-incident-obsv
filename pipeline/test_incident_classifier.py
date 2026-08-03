import json

from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.incident_classifier import (
    ClassificationOutputError,
    LLM_MODEL_DEFAULT,
    classify,
    extract_details,
    generate_extraction_prompt,
    generate_prompt,
    parse_details,
    parse_judgments,
)


def _valid_detail(entry_id="e0", **overrides):
    detail = {
        "entry_id": entry_id,
        "title": "AV strikes deer",
        "description": "An autonomous vehicle struck a deer.",
        "animal_type": "wild",
        "animal_species": "deer",
        "animal_count": 1,
        "harm_type": "collision",
        "harm_description": "fatal impact",
        "ai_system": "Waymo Driver",
        "country": "USA",
    }
    detail.update(overrides)
    return detail


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


@patch("pipeline.incident_classifier.subprocess.run")
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


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_passes_model_flag(mock_run, tmp_path):
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess([], judgments_path)

    classify(tmp_path, incidents_path, judgments_path, model="foo/bar")

    args = mock_run.call_args[0][0]
    assert "-m" in args
    assert args[args.index("-m") + 1] == "foo/bar"


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_uses_default_model_when_none(mock_run, tmp_path):
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess([], judgments_path)

    model, _ = classify(tmp_path, incidents_path, judgments_path)

    assert model == LLM_MODEL_DEFAULT
    args = mock_run.call_args[0][0]
    assert args[args.index("-m") + 1] == LLM_MODEL_DEFAULT


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_raises_when_judgments_file_missing(mock_run, tmp_path):
    # LLM subprocess "succeeds" but never writes the judgments file. This is a
    # hard failure (not zero valid judgments), so classify must raise rather
    # than silently return no judgments.
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.return_value = MagicMock(returncode=0)

    try:
        classify(tmp_path, incidents_path, judgments_path, model="m/x")
    except ClassificationOutputError as exc:
        assert str(judgments_path) in str(exc)
    else:
        raise AssertionError(
            "expected ClassificationOutputError when judgments file is missing"
        )


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_raises_when_judgments_file_is_malformed_json(mock_run, tmp_path):
    # File exists but is not valid JSON -> structural failure, not empty.
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")

    def fake_run(*args, **kwargs):
        judgments_path.write_text("this is not json{")
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_run

    try:
        classify(tmp_path, incidents_path, judgments_path, model="m/x")
    except ClassificationOutputError as exc:
        assert "not valid JSON" in str(exc)
    else:
        raise AssertionError("expected ClassificationOutputError on malformed JSON")


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_raises_when_judgments_top_level_not_list(mock_run, tmp_path):
    # Valid JSON but top-level is an object, not a list -> structural failure.
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")

    def fake_run(*args, **kwargs):
        judgments_path.write_text('{"judgments": []}')
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_run

    try:
        classify(tmp_path, incidents_path, judgments_path, model="m/x")
    except ClassificationOutputError as exc:
        assert "expected a JSON list" in str(exc)
    else:
        raise AssertionError("expected ClassificationOutputError on non-list top-level")


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_allows_valid_empty_list(mock_run, tmp_path):
    # A structurally-valid empty list is a genuine "zero judgments" result and
    # must NOT raise.
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess([], judgments_path)

    assert classify(tmp_path, incidents_path, judgments_path, model="m/x") == ("m/x", [])


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_tolerates_all_records_invalid(mock_run, tmp_path):
    # Valid JSON list whose records all fail per-record validation is NOT a
    # structural failure: the bad records are skipped (retried next run) and
    # classify returns an empty list rather than raising.
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess(
        [{"keep": True}, {"entry_id": 5, "keep": "yes"}],
        judgments_path,
    )

    assert classify(tmp_path, incidents_path, judgments_path, model="m/x") == ("m/x", [])


@patch("pipeline.incident_classifier.subprocess.run")
def test_classify_passes_absolute_paths_to_prompt(mock_run, tmp_path):
    # Writer/reader must agree on an unambiguous absolute path, independent of
    # the subprocess cwd. The prompt (first positional arg after "run") should
    # contain the absolute incidents and judgments paths.
    incidents_path = tmp_path / "incidents.json"
    judgments_path = tmp_path / "judgments.json"
    incidents_path.write_text("[]")
    mock_run.side_effect = _make_fake_subprocess([], judgments_path)

    classify(tmp_path, incidents_path, judgments_path, model="m/x")

    prompt = mock_run.call_args[0][0][2]
    assert str(incidents_path) in prompt
    assert str(judgments_path) in prompt


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


# ---------- generate_extraction_prompt ----------


def test_generate_extraction_prompt_mentions_filenames_and_fields():
    prompt = generate_extraction_prompt("in.json", "out.json")
    assert "in.json" in prompt
    assert "out.json" in prompt
    assert "entry_id" in prompt
    assert "animal_type" in prompt
    assert "animal_count" in prompt


# ---------- extract_details (LLM invocation + return shape) ----------


@patch("pipeline.incident_classifier.subprocess.run")
def test_extract_details_returns_model_and_details(mock_run, tmp_path):
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    input_path.write_text("[]")

    def fake_run(*args, **kwargs):
        output_path.write_text(json.dumps([_valid_detail("e0000")]))
        return MagicMock(returncode=0)

    mock_run.side_effect = fake_run

    model, details = extract_details(tmp_path, input_path, output_path, model="m/x")
    assert model == "m/x"
    assert details == [_valid_detail("e0000")]


@patch("pipeline.incident_classifier.subprocess.run")
def test_extract_details_uses_default_model_when_none(mock_run, tmp_path):
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    input_path.write_text("[]")
    mock_run.side_effect = lambda *a, **k: output_path.write_text("[]")

    model, _ = extract_details(tmp_path, input_path, output_path)
    assert model == LLM_MODEL_DEFAULT
    args = mock_run.call_args[0][0]
    assert args[args.index("-m") + 1] == LLM_MODEL_DEFAULT


# ---------- parse_details: file / JSON / top-level shape ----------


def test_parse_details_returns_empty_when_file_missing(tmp_path, capsys):
    result = parse_details(tmp_path / "does_not_exist.json")
    assert result == []
    assert "details file missing" in capsys.readouterr().out


def test_parse_details_returns_empty_on_invalid_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("not json{")
    result = parse_details(path)
    assert result == []
    assert "not valid JSON" in capsys.readouterr().out


def test_parse_details_returns_empty_when_top_level_not_list(tmp_path, capsys):
    path = tmp_path / "obj.json"
    path.write_text('{"details": []}')
    result = parse_details(path)
    assert result == []
    assert "top-level is dict" in capsys.readouterr().out


# ---------- parse_details: per-record shape ----------


def test_parse_details_appends_valid_record(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([_valid_detail("e0")]))
    result = parse_details(path)
    assert result == [_valid_detail("e0")]


def test_parse_details_skips_non_dict_element(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(["nope", _valid_detail("e0")]))
    result = parse_details(path)
    assert [d["entry_id"] for d in result] == ["e0"]
    assert "expected dict" in capsys.readouterr().out


def test_parse_details_skips_bad_animal_type_enum(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([
        _valid_detail("e0", animal_type="dinosaur"),
        _valid_detail("e1"),
    ]))
    result = parse_details(path)
    assert [d["entry_id"] for d in result] == ["e1"]
    assert "animal_type must be one of" in capsys.readouterr().out


def test_parse_details_skips_non_int_animal_count(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([
        _valid_detail("e0", animal_count="three"),
        _valid_detail("e1"),
    ]))
    result = parse_details(path)
    assert [d["entry_id"] for d in result] == ["e1"]
    assert "animal_count must be int" in capsys.readouterr().out


def test_parse_details_rejects_bool_animal_count(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([_valid_detail("e0", animal_count=True)]))
    result = parse_details(path)
    assert result == []
    assert "animal_count must be int" in capsys.readouterr().out


def test_parse_details_skips_missing_entry_id(tmp_path, capsys):
    path = tmp_path / "d.json"
    bad = _valid_detail()
    del bad["entry_id"]
    path.write_text(json.dumps([bad, _valid_detail("e1")]))
    result = parse_details(path)
    assert [d["entry_id"] for d in result] == ["e1"]
    assert "entry_id must be str" in capsys.readouterr().out


def test_parse_details_skips_non_str_field(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([
        _valid_detail("e0", title=123),
        _valid_detail("e1"),
    ]))
    result = parse_details(path)
    assert [d["entry_id"] for d in result] == ["e1"]
    assert "title must be str" in capsys.readouterr().out


# ---------- parse_details: dedupe (first wins) ----------


def test_parse_details_dedupes_repeated_entry_ids_first_wins(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([
        _valid_detail("e0", title="first"),
        _valid_detail("e0", title="second"),
    ]))
    result = parse_details(path)
    assert len(result) == 1
    assert result[0]["title"] == "first"
    out = capsys.readouterr().out
    assert "duplicate entry_id" in out
