import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import tablib

from pipeline.classifier_types import Judgment


URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"
OUTPUT_PATH = Path("incidents.json")
OPENALEX_BASE_URL = "https://api.openalex.org/works"



# ---------- HTTP fetching ----------


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


# ---------- Trimming ----------


NHTSA_KEEP_FIELDS = [
    "Report ID", "Report Version", "Reporting Entity",
    "Crash With", "Narrative", "City", "State",
    "Incident Date", "SV Precrash Speed (MPH)",
    "Highest Injury Severity Alleged",
    "CP Pre-Crash Movement", "SV Pre-Crash Movement",
    "Within ODD?",
]


OPENALEX_KEEP_FIELDS = [
    "id",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "abstract",
    "primary_location.landing_page_url",
    "primary_location.pdf_url",
]


_MISSING = object()


def _get_dotted(d: dict, path: str, default=None):
    """Return the value at ``path`` (dotted keys) inside nested dicts, else default."""
    current: object = d
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set_dotted(d: dict, path: str, value) -> None:
    """Assign ``value`` at ``path`` inside ``d``, creating intermediate dicts."""
    parts = path.split(".")
    current = d
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def trim_nhtsa_fields(rows: list[dict]) -> list[dict]:
    return [{k: row[k] for k in NHTSA_KEEP_FIELDS if k in row} for row in rows]


def trim_openalex_fields(rows: list[dict]) -> list[dict]:
    """Trim OpenAlex work dicts to only the keep-list fields, preserving nesting.

    Dotted keep-list entries (e.g. ``"primary_location.landing_page_url"``) are
    read from the source at that nested path and re-emitted as nested keys in
    the trimmed dict, not flattened.
    """
    result = []
    for row in rows:
        trimmed: dict = {}
        for path in OPENALEX_KEEP_FIELDS:
            value = _get_dotted(row, path, _MISSING)
            if value is _MISSING:
                continue
            _set_dotted(trimmed, path, value)
        result.append(trimmed)
    return result


# ---------- Date filtering, CSV parsing, JSON writing ----------


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


# ---------- Collection ----------


def collect_data(now: datetime = datetime.now()) -> list[dict]:
    """Fetch and tag entries from all configured sources.

    Returns untrimmed entries with an ``aaiid_data_source`` field. Trimming
    happens later in ``assemble_csv``.
    """
    csv_text = fetch_csv(URL)
    data = parse_csv(csv_text)
    # Might be relevant later, but for now the csv we have is a constant data source of Jun 2025-May 2026
    # data = filter_by_date(data, now=now)
    for row in data:
        row["aaiid_data_source"] = "nhtsa_incident_report"

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


# ---------- Entry ids ----------


def assign_entry_ids(entries: list[dict], start: int = 0) -> None:
    """Mutate entries in place, adding ``entry_id`` fields like ``"e0000"``.

    Idempotent: entries that already have an ``entry_id`` are left untouched.
    The counter advances only for entries that need one.
    """
    counter = start
    for entry in entries:
        if "entry_id" not in entry:
            entry["entry_id"] = f"e{counter:04d}"
            counter += 1


# ---------- Record assembly ----------


# Fields to strip from the trimmed json_blob because they are internal
# bookkeeping, not source data.
_JSON_BLOB_STRIP = {"aaiid_data_source", "entry_id"}


def _trim_entry(entry: dict) -> dict:
    source = entry.get("aaiid_data_source")
    if source == "nhtsa_incident_report":
        trimmed = trim_nhtsa_fields([entry])[0]
    elif source == "openalex_work":
        trimmed = trim_openalex_fields([entry])[0]
    else:
        # unknown source: pass through as-is minus internal fields
        trimmed = dict(entry)
    return {k: v for k, v in trimmed.items() if k not in _JSON_BLOB_STRIP}


def _source_id(entry: dict) -> str:
    """Return the source's natural key for ``entry``.

    NHTSA rows are keyed by ``Report ID``; OpenAlex works by ``id``. Falls back
    to an empty string when the expected key is absent.
    """
    source = entry.get("aaiid_data_source")
    if source == "nhtsa_incident_report":
        return entry.get("Report ID", "") or ""
    if source == "openalex_work":
        return entry.get("id", "") or ""
    return ""


def build_records(
    entries: list[dict],
    judgments: list[Judgment],
    model: str,
) -> list[dict]:
    """Join entries with kept judgments and return structured records.

    Performs the entry-aware reconciliation:
      - Unknown ``entry_id`` values (not present in ``entries``) are logged and
        skipped.
      - Judgments with ``keep`` falsey are dropped.
      - A summary of ``entry_id`` values that received no valid judgment is
        printed at the end.

    Each returned record is a dict with the trimmed source data under
    ``json_blob`` plus the judgment fields, ready to be persisted or scored.
    """
    entries_by_id = {e["entry_id"]: e for e in entries if "entry_id" in e}

    records: list[dict] = []
    seen_ids: set[str] = set()

    for j in judgments:
        entry_id = j["entry_id"]
        entry = entries_by_id.get(entry_id)
        # Membership check runs regardless of keep so unknown ids are always
        # logged.
        if entry is None:
            print(
                f"build_records: judgment has unknown entry_id {entry_id!r} "
                f"(not among {len(entries_by_id)} input entries); skipping"
            )
            continue
        seen_ids.add(entry_id)
        if not j.get("keep"):
            continue
        records.append({
            "aaiid_data_source": entry.get("aaiid_data_source", ""),
            "entry_id": entry_id,
            "source_id": _source_id(entry),
            "json_blob": _trim_entry(entry),
            "reasoning": j.get("reasoning", ""),
            "confidence": j.get("confidence", ""),
            "model": model,
        })

    missing = set(entries_by_id) - seen_ids
    if missing:
        sample = sorted(missing)[:5]
        print(
            f"build_records: {len(missing)} entry_id(s) had no valid judgment "
            f"(will be omitted); sample: {sample}"
        )

    return records







