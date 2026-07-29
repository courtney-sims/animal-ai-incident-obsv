import json
from pathlib import Path
import subprocess

from typing import Tuple

from pipeline.classifier_types import Judgment, VALID_CONFIDENCES

# Default LLM used for relevance classification.
LLM_MODEL_DEFAULT = "opencode/deepseek-v4-flash-free"

# Allowed values for the extracted ``animal_type`` field (mirrors the model's
# AnimalType choices).
VALID_ANIMAL_TYPES = ("farmed", "wild", "companion", "other", "unknown")

def generate_prompt(incidents_filename: str, judgments_filename: str) -> str:
    return f"""\
You are a classifier for the Animal AI Incident Observatory. You will receive
a JSON array of entries in the file {incidents_filename}. Each entry has an
"entry_id" field.

For every entry, decide whether it describes an AI system directly harming a
non-human animal, or (for openalex_work entries) a paper either related to that topic or whose research generated those harms.

Rules for aaiid_data_source == "nhtsa_incident_report":
- Keep if "Crash With" is "Animal".
- Keep if the "Narrative" mentions an animal (dog, cat, raccoon, bird, duck,
  flock, deer, wildlife, etc.).
- Do NOT keep narratives whose only animal-like token is "HAWK" — that is a
  pedestrian beacon acronym, not a bird.

Rules for aaiid_data_source == "openalex_work":
- Keep only if the title/abstract are actually about AI or autonomous vehicles
  interacting with animals.
- Drop general animal health, animal law, animal testing, or agriculture
  unless AI is central.

Output: write a JSON array to {judgments_filename} in the current working
directory. Include exactly one object per input entry. Each object must have:
- entry_id: string, matching the input exactly
- keep: boolean (true or false, JSON literal — not a string)
- reasoning: short string explaining the decision
- confidence: exactly one of "High", "Medium", "Low"

Confidence guidance:
- "High" when the entry directly and unambiguously matches (e.g. Crash With ==
  "Animal").
- "Medium" for narrative-only NHTSA matches or clearly-relevant OpenAlex
  papers.
- "Low" for ambiguous cases you're still leaning toward keep.

Write only the JSON file. Do not emit any other output.
"""

def parse_judgments(judgments_path) -> list[Judgment]:
    """Load a judgments JSON file and drop any records that don't validate.

    Handles all validation that does not require knowledge of the input
    entries: file/JSON/top-level shape, per-record shape and enum, and
    first-wins deduplication of ``entry_id``. Membership checks (is this entry_id known?) and the missing-ids report
    live downstream in ``assemble_csv``, where the entries list is available.
    """
    if not judgments_path.exists():
        print(f"classify: judgments file missing at {judgments_path}; returning no judgments")
        return []

    try:
        raw = json.loads(judgments_path.read_text())
    except json.JSONDecodeError as e:
        print(
            f"classify: judgments file at {judgments_path} is not valid JSON: {e}; "
            "returning no judgments"
        )
        return []

    if not isinstance(raw, list):
        print(
            f"classify: judgments file top-level is {type(raw).__name__}, "
            "expected list; returning no judgments"
        )
        return []

    validated: list[Judgment] = []
    kept_by_id: dict[str, Judgment] = {}

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            print(
                f"classify: judgment[{i}] is {type(item).__name__}, "
                # truncates value just in case it's super long
                f"expected dict; skipping. value={item!r:.200}"
            )
            continue

        entry_id = item.get("entry_id")
        keep = item.get("keep")
        reasoning = item.get("reasoning")
        confidence = item.get("confidence")

        problems = []
        if not isinstance(entry_id, str):
            problems.append(f"entry_id must be str, got {type(entry_id).__name__}={entry_id!r}")
        if not isinstance(keep, bool):
            problems.append(f"keep must be bool, got {type(keep).__name__}={keep!r}")
        if not isinstance(reasoning, str):
            problems.append(f"reasoning must be str, got {type(reasoning).__name__}={reasoning!r}")
        if not isinstance(confidence, str) or confidence not in VALID_CONFIDENCES:
            problems.append(
                f"confidence must be one of {list(VALID_CONFIDENCES)}, got {confidence!r}"
            )

        if problems:
            print(
                f"classify: judgment[{i}] (entry_id={entry_id!r}) invalid — "
                f"{'; '.join(problems)}; skipping"
            )
            continue

        normalized: Judgment = {
            "entry_id": entry_id,
            "keep": keep,
            "reasoning": reasoning,
            "confidence": confidence,
        }

        if entry_id in kept_by_id:
            print(
                f"classify: judgment[{i}] duplicate entry_id {entry_id!r}; "
                "keeping first, discarding second.\n"
                f"  kept:      {kept_by_id[entry_id]!r}\n"
                f"  discarded: {normalized!r}"
            )
            continue

        kept_by_id[entry_id] = normalized
        validated.append(normalized)

    return validated

def generate_extraction_prompt(input_filename: str, output_filename: str) -> str:
    return f"""\
You are a detail extractor for the Animal AI Incident Observatory. You will
receive a JSON array of entries in the file {input_filename}. Each entry has an
"entry_id" field and describes an AI system harming (or a paper about AI
harming) a non-human animal.

For every entry, extract the following fields from the entry's content. When a
value cannot be determined from the entry, use the unknown-marker shown.

Output: write a JSON array to {output_filename} in the current working
directory. Include exactly one object per input entry. Each object must have:
- entry_id: string, matching the input exactly
- title: short descriptive title of the incident (string; "" if unknown)
- description: one or two sentence summary of the incident (string; "" if unknown)
- animal_type: exactly one of "farmed", "wild", "companion", "other", "unknown"
- animal_species: the species involved (string; "" if unknown)
- animal_count: integer count of animals impacted (-1 if unknown)
- harm_type: short category of harm (string; "" if unknown)
- harm_description: longer description of the harm (string; "" if unknown)
- ai_system: the AI system involved, e.g. "Waymo", "ChatGPT" (string; "" if unknown)
- country: country where the incident occurred (string; "" if unknown)

Unknown values: use "" for strings, -1 for animal_count, "unknown" for
animal_type. animal_count must be a JSON integer literal, not a string.

Write only the JSON file. Do not emit any other output.
"""


def parse_details(output_path) -> list[dict]:
    """Load an extracted-details JSON file and drop records that don't validate.

    Mirrors :func:`parse_judgments`: handles file/JSON/top-level shape,
    per-record type checks, the ``animal_type`` enum, the ``animal_count`` int
    check, and first-wins deduplication of ``entry_id``. Invalid records are
    logged and skipped; the pipeline does not raise on validation problems.
    """
    if not output_path.exists():
        print(f"extract: details file missing at {output_path}; returning no details")
        return []

    try:
        raw = json.loads(output_path.read_text())
    except json.JSONDecodeError as e:
        print(
            f"extract: details file at {output_path} is not valid JSON: {e}; "
            "returning no details"
        )
        return []

    if not isinstance(raw, list):
        print(
            f"extract: details file top-level is {type(raw).__name__}, "
            "expected list; returning no details"
        )
        return []

    _STR_FIELDS = (
        "title",
        "description",
        "animal_species",
        "harm_type",
        "harm_description",
        "ai_system",
        "country",
    )

    validated: list[dict] = []
    kept_by_id: dict[str, dict] = {}

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            print(
                f"extract: detail[{i}] is {type(item).__name__}, "
                f"expected dict; skipping. value={item!r:.200}"
            )
            continue

        entry_id = item.get("entry_id")
        animal_type = item.get("animal_type")
        animal_count = item.get("animal_count")

        problems = []
        if not isinstance(entry_id, str):
            problems.append(f"entry_id must be str, got {type(entry_id).__name__}={entry_id!r}")
        for field in _STR_FIELDS:
            value = item.get(field)
            if not isinstance(value, str):
                problems.append(f"{field} must be str, got {type(value).__name__}={value!r}")
        if not isinstance(animal_type, str) or animal_type not in VALID_ANIMAL_TYPES:
            problems.append(
                f"animal_type must be one of {list(VALID_ANIMAL_TYPES)}, got {animal_type!r}"
            )
        # bool is a subclass of int; reject it explicitly.
        if not isinstance(animal_count, int) or isinstance(animal_count, bool):
            problems.append(
                f"animal_count must be int, got {type(animal_count).__name__}={animal_count!r}"
            )

        if problems:
            print(
                f"extract: detail[{i}] (entry_id={entry_id!r}) invalid — "
                f"{'; '.join(problems)}; skipping"
            )
            continue

        normalized = {
            "entry_id": entry_id,
            "title": item["title"],
            "description": item["description"],
            "animal_type": animal_type,
            "animal_species": item["animal_species"],
            "animal_count": animal_count,
            "harm_type": item["harm_type"],
            "harm_description": item["harm_description"],
            "ai_system": item["ai_system"],
            "country": item["country"],
        }

        if entry_id in kept_by_id:
            print(
                f"extract: detail[{i}] duplicate entry_id {entry_id!r}; "
                "keeping first, discarding second.\n"
                f"  kept:      {kept_by_id[entry_id]!r}\n"
                f"  discarded: {normalized!r}"
            )
            continue

        kept_by_id[entry_id] = normalized
        validated.append(normalized)

    return validated


def extract_details(
    workdir: Path,
    input_path: Path,
    output_path: Path,
    model: str | None = None,
) -> Tuple[str, list[dict]]:
    """Ask the LLM to extract per-entry detail fields; return validated details.

    Mirrors :func:`classify`: writes are done by the caller, shells to
    ``opencode run`` with an extraction-oriented prompt, then reads and
    validates the details file the LLM produced. Records that fail validation
    are logged and skipped; surviving details are returned.
    """
    resolved_model = model or LLM_MODEL_DEFAULT

    prompt = generate_extraction_prompt(input_path.name, output_path.name)
    subprocess.run(
        [
            "opencode", "run", prompt,
            "-f", str(input_path),
            "-m", resolved_model,
        ],
        cwd=str(workdir),
        check=True,
    )

    details = parse_details(output_path)

    return resolved_model, details


def classify(
    workdir: Path,
    incidents_path: Path,
    judgments_path: Path,
    model: str | None = None,
) -> Tuple[str,list[Judgment]]:
    """Ask the LLM to classify each entry; return validated judgments.

    Writes the input entries as JSON, shells to ``opencode run`` with a
    judgment-oriented prompt, then reads and validates the judgments file the
    LLM produced. Records that fail validation are logged and skipped;
    surviving judgments are returned. The pipeline does not raise on
    validation problems.
    """
    resolved_model = model or LLM_MODEL_DEFAULT

    prompt = generate_prompt(incidents_path.name, judgments_path.name)
    subprocess.run(
        [
            "opencode", "run", prompt,
            "-f", str(incidents_path),
            "-m", resolved_model,
        ],
        cwd=str(workdir),
        check=True,
    )

    judgments = parse_judgments(judgments_path)
    
    return resolved_model, judgments

