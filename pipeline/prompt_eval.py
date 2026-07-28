import tempfile
from pathlib import Path

from pipeline.incident_classifier import LLM_MODEL_DEFAULT, classify
from pipeline.incident_fetcher import (
    assign_entry_ids,
    build_records,
    write_json,
)


def build_test_data() -> tuple[list[dict], set[str]]:
    incidents = [
        {
            "aaiid_data_source": "nhtsa_incident_report",
            "Report ID": "EVAL-001",
            "Crash With": "Animal",
            "Narrative": "The AV made contact with a duck.",
            "City": "Austin",
            "State": "TX",
        },
        {
            "aaiid_data_source": "nhtsa_incident_report",
            "Report ID": "EVAL-002",
            "Crash With": "Animal",
            "Narrative": "A dog ran into the roadway and the AV braked.",
            "City": "Phoenix",
            "State": "AZ",
        },
        {
            "aaiid_data_source": "nhtsa_incident_report",
            "Report ID": "EVAL-003",
            "Crash With": "Passenger Car",
            "Narrative": "The AV was rear-ended at a stop sign.",
            "City": "San Francisco",
            "State": "CA",
        },
        {
            "aaiid_data_source": "nhtsa_incident_report",
            "Report ID": "EVAL-004",
            "Crash With": "Passenger Car",
            "Narrative": "The AV slowed for a HAWK pedestrian beacon and was hit from behind.",
            "City": "Phoenix",
            "State": "AZ",
        },
        {
            "aaiid_data_source": "openalex_work",
            "id": "https://openalex.org/W-EVAL-001",
            "title": "Autonomous vehicle interactions with farm animals",
            "display_name": "Autonomous vehicle interactions with farm animals",
            "primary_location": {"landing_page_url": None, "pdf_url": None},
        },
        {
            "aaiid_data_source": "openalex_work",
            "id": "https://openalex.org/W-EVAL-002",
            "title": "Canine diabetes management in veterinary practice",
            "display_name": "Canine diabetes management in veterinary practice",
            "primary_location": {"landing_page_url": None, "pdf_url": None},
        },
    ]

    expected: set[str] = {"EVAL-001", "EVAL-002", "W-EVAL-001"}
    return incidents, expected


def _extract_id(source_id: str) -> str:
    return source_id.replace("https://openalex.org/", "")


def run_eval(output_dir: str | None = None, model: str | None = None) -> dict:
    incidents, expected_ids = build_test_data()
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    workdir = Path(output_dir)

    assign_entry_ids(incidents)
    resolved_model = model or LLM_MODEL_DEFAULT

    incidents_path = workdir / "eval_incidents.json"
    judgments_path = workdir / "eval_judgments.json"
    write_json(incidents, incidents_path)

    resolved_model, judgments = classify(
        workdir, incidents_path, judgments_path, model=resolved_model
    )
    records = build_records(incidents, judgments, resolved_model)

    metrics = score_output(records, expected_ids)
    metrics["model"] = resolved_model
    return metrics


def score_output(records: list[dict], expected_ids: set[str]) -> dict:
    found_ids = set()
    for record in records:
        found_ids.add(_extract_id(record["source_id"]))

    true_positives = found_ids & expected_ids
    false_positives = found_ids - expected_ids
    false_negatives = expected_ids - found_ids

    precision = len(true_positives) / len(found_ids) if found_ids else 0.0
    recall = len(true_positives) / len(expected_ids) if expected_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
    }
