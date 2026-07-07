import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
import tablib


URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"
OUTPUT_PATH = Path("incidents.json")
OPENALEX_BASE_URL = "https://api.openalex.org/works"

# Default LLM used for relevance classification.
LLM_MODEL_DEFAULT = "opencode/deepseek-v4-flash-free"


def fetch_csv(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def query_openalex(params: dict) -> list[dict]:
    # Eventually, log and skip the source on error. Right now, just bubble it up for dev
    response = requests.get(OPENALEX_BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [])


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild plain-text abstract from OpenAlex ``abstract_inverted_index``.

    OpenAlex returns abstracts as  word-frequency map with positions ``{word: [positions, ...]}``.
    Ex:
        {
            "the": [0, 3],
            "cat": [1],
            "and": [2],
            "dog": [4]
        } == "the cat and the dog"

    It's possible this isn't needed and an LLM could parse the original form equally well. Could be worth investigating.

    Returns ``None`` when the input is missing, empty, or not a dict. Malformed
    entries (non-list values, non-int positions) are skipped rather than
    raising.
    """
    if not inverted_index:
        print("reconstruct_abstract: skipping — inverted_index is empty or None")
        return None
    if not isinstance(inverted_index, dict):
        print(
            f"reconstruct_abstract: skipping — expected dict, got "
            f"{type(inverted_index).__name__}: {inverted_index!r:.200}"
        )
        return None

    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            print(
                f"reconstruct_abstract: skipping word {word!r} — expected list of "
                f"positions, got {type(positions).__name__}: {positions!r:.100}"
            )
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, word))
            else:
                print(
                    f"reconstruct_abstract: skipping position for word {word!r} — "
                    f"expected int, got {type(pos).__name__}: {pos!r:.100}"
                )

    if not positioned:
        print(
            f"reconstruct_abstract: no valid positions found in inverted_index "
            f"with {len(inverted_index)} key(s)"
        )
        return None

    positioned.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positioned)


NHTSA_KEEP_FIELDS = [
    "Report ID", "Report Version", "Reporting Entity",
    "Crash With", "Narrative", "City", "State",
    "Incident Date", "SV Precrash Speed (MPH)",
    "Highest Injury Severity Alleged",
    "CP Pre-Crash Movement", "SV Pre-Crash Movement",
    "Within ODD?", "aaiid_data_source",
]


def trim_nhtsa_fields(rows: list[dict]) -> list[dict]:
    return [{k: row[k] for k in NHTSA_KEEP_FIELDS if k in row} for row in rows]


def filter_by_date(
    data: list[dict],
    months_back: int = 1,
    now: datetime = datetime.now(),
) -> list[dict]:
    """Return rows where ``Incident Date`` is within the last ``months_back`` months.

    ``Incident Date`` is expected in ``"%b-%Y"`` form (e.g. ``"JUN-2026"``). Rows
    with a missing, blank, or unparseable date are dropped and logged.
    """
    result = []
    for row in data:
        date_str = row.get("Incident Date", "")
        if not date_str or not date_str.strip():
            print(f'Could not get incident date from {row}')
            continue
        try:
            dt = datetime.strptime(date_str.strip(), "%b-%Y")
        except ValueError:
            print(f'Could not parse month and year from {date_str}')
            continue
        months_diff = (now.year - dt.year) * 12 + (now.month - dt.month)
        if months_diff < months_back:
            result.append(row)
    return result


def parse_csv(text: str) -> list[dict]:
    if not text.strip():
        return []
    data = tablib.Dataset()
    data.load(text, format="csv")
    return data.dict


def write_json(data: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def collect_data(trim: bool = False, now: datetime = datetime.now()) -> list[dict]:
    csv_text = fetch_csv(URL)
    data = parse_csv(csv_text)
    data = filter_by_date(data, now=now)
    for row in data:
        row["aaiid_data_source"] = "nhtsa_incident_report"
    if trim:
        data = trim_nhtsa_fields(data)

    thirty_days_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "sort": "publication_date:desc",
        "per_page": 25,
        "search": "animal",
        "filter": f"from_publication_date:{thirty_days_ago}",
        "select": "id,title,display_name,publication_year,publication_date,doi,abstract_inverted_index,primary_location,keywords,referenced_works,concepts",
    }
    papers = query_openalex(params)
    for paper in papers:
        paper["aaiid_data_source"] = "openalex_work"
        paper["abstract"] = reconstruct_abstract(
            paper.pop("abstract_inverted_index", None)
        )

    return data + papers


def write_timestamped_csv(
    content: str,
    directory: Path = Path("."),
    now: datetime = datetime.now(),
) -> Path:
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    path = Path(directory) / f"animal_incidents_{timestamp}.csv"
    path.write_text(content)
    return path


def append_llm_meta_column(csv_path: Path, model: str) -> None:
    """Append an ``llm_meta`` column to ``csv_path`` in place to store data like the model used.

    This will expand to having python write all the data from
    the LLM provided response.
    If an
    ``llm_meta`` column already exists, its values are overwritten instead of
    duplicating the column. This way if an incident is reprocessed, it will have the latest meta.
    """
    text = Path(csv_path).read_text()
    if not text.strip():
        return

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return

    header = rows[0]
    meta_value = json.dumps({"model": model}, separators=(",", ":"))

    if "llm_meta" in header:
        idx = header.index("llm_meta")
        for row in rows[1:]:
            # pad short rows so idx is addressable
            while len(row) <= idx:
                row.append("")
            row[idx] = meta_value
    else:
        header.append("llm_meta")
        for row in rows[1:]:
            row.append(meta_value)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    Path(csv_path).write_text(buf.getvalue())


def run_classification(
    incidents_path: Path,
    prompt: str,
    now: datetime = datetime.now(),
    model: str | None = None,
) -> Path:
    incidents_path = Path(incidents_path)
    workdir = incidents_path.parent
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_name = f"animal_incidents_{timestamp}.csv"
    output_path = workdir / output_name
    resolved_model = model or LLM_MODEL_DEFAULT

    full_prompt = f"{prompt}\n\n6. Name the output file {output_name}."
    subprocess.run(
        [
            "opencode", "run", full_prompt,
            "-f", str(incidents_path),
            "-m", resolved_model,
        ],
        cwd=str(workdir),
        check=True,
    )

    if output_path.exists():
        append_llm_meta_column(output_path, resolved_model)

    return output_path


def generate_prompt(incidents_filename: str = "incidents.json") -> str:
    return f"""\
Review the file {incidents_filename} and produce a CSV of animal-related AI incidents.

1. Load the JSON file (it's a JSON array of objects).

2. For each object with "aaiid_data_source": "nhtsa_incident_report", include it if:
   - "Crash With" is "Animal", OR
   - The "Narrative" field mentions an animal keyword (e.g. dog, cat, raccoon, bird, duck, flock, domestic animal, etc.)
   - Exclude false positives: "HAWK" (pedestrian beacon acronym, not a bird) should not count.

3. For each object with "aaiid_data_source": "openalex_work", read its title and "abstract" field (already reconstructed to plain text). Keep it only if it's actually about AI or autonomous vehicles interacting with animals. Drop general animal health, animal law, animal testing, agriculture, or other unrelated topics.

4. For every kept entry, trim the JSON blob to only these relevant fields:
   - NHTSA entries: Report ID, Report Version, Reporting Entity, Crash With, Narrative, City, State, Incident Date, SV Precrash Speed (MPH), Highest Injury Severity Alleged
   - OpenAlex entries: id, title, display_name, publication_year, publication_date, abstract, primary_location.landing_page_url, primary_location.pdf_url

5. Write the CSV to the current directory with columns: aaiid_data_source, json_blob, reasoning, confidence_score. The json_blob column should contain the trimmed JSON (compact, no extra whitespace). The reasoning column should explain why the entry was included. Use confidence_score of "High" for entries with "Crash With": "Animal", "Medium" for narrative-only matches, and "Low" for OpenAlex entries where relevance is uncertain.
"""


def run_pipeline(model: str | None = None) -> Path:
    data = collect_data(trim=True)
    write_json(data, OUTPUT_PATH)
    prompt = generate_prompt(str(OUTPUT_PATH))
    output_path = run_classification(OUTPUT_PATH, prompt, model=model)
    return output_path


def parse_cli_args(argv: list[str]) -> dict:
    """Parse ``key=value`` positional args from the CLI.

    Currently accepts ``model=<provider/model>``. Unknown keys raise SystemExit
    with a usage message.
    """
    known = {"model"}
    parsed: dict = {}
    for arg in argv:
        if "=" not in arg:
            sys.exit(f"unrecognized argument {arg!r}; expected key=value form (e.g. model=foo/bar)")
        key, _, value = arg.partition("=")
        if key not in known:
            sys.exit(f"unknown key {key!r}; known keys: {known}")
        parsed[key] = value
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv if argv is not None else sys.argv[1:])
    run_pipeline(model=args.get("model"))


if __name__ == "__main__":
    main()
