import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
import tablib

from pipeline.classifier_types import Judgment


OUTPUT_PATH = Path("incidents.json")
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO")
# OpenAlex meters the API by a daily USD budget: $1/day with a (free) key vs
# only $0.01/day without one. Our query is a search op (~$0.001/request), so an
# unauthenticated run can't even complete one paginated pass (~10 req/day cap).
# Get a free key at https://openalex.org/settings/api and set OPENALEX_API_KEY.
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")

# ---------- Sources ----------


OPEN_ALEX = "openalex_work"
OPENALEX_BASE_URL = "https://api.openalex.org/works"
# Rolling window (in days) back from ``now`` used to filter OpenAlex works by
# publication date. A window (rather than "since last run") absorbs OpenAlex
# indexing lag: works often appear days-to-weeks after their publication date,
# so re-scanning the recent past on every run catches late-indexed papers.
# Measured lag on a real sample: median ~6d, p99 ~33d, max ~87d — 90 days keeps
# lag-driven misses at ~0%. Re-fetching is harmless — ingestion upserts on
# (source, source_id) and never resets status, so already-judged rows just get
# their raw_data refreshed.
OPENALEX_LOOKBACK_DAYS = 90
# Cap on cursor-paginated pages per run (per_page=100 => up to 3000 works) as a
# safety valve against an unexpectedly huge result set and runaway LLM load.
# The scoped query below yields ~2.3k works at the 90-day window, so 3000 leaves
# headroom without silently dropping results. per_page is 100 because that is
# OpenAlex's documented maximum.
OPENALEX_MAX_PAGES = 30
OPENALEX_PER_PAGE = 100

# Scoped OpenAlex query terms. The observatory targets works where AI/autonomous
# systems harm non-human animals, so we require a term from BOTH lists. Using
# OpenAlex's pipe (OR-within-a-filter) and repeated-filter (AND-across-filters)
# syntax instead of inline boolean operators avoids the free-tier boolean-
# operator limit (which returns 429). `search=animal` alone matched ~38k works
# (19x over the fetch cap); this pair keeps the 90-day window near ~2.3k.
# These are the recall/precision lever — tune here.
OPENALEX_ANIMAL_TERMS = [
    "animal", "animals", "wildlife", "livestock", "deer", "bird", "cattle",
]
OPENALEX_TECH_TERMS = [
    "autonomous vehicle", "self-driving", "artificial intelligence",
    "machine learning", "drone", "robot",
]

NHTSA = "nhtsa_incident_report"
URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"


# ---------- HTTP fetching ----------


def fetch_csv(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


_REDACT_PARAMS = ("api_key", "mailto")


def _redact_url(url: str) -> str:
    """Return ``url`` with sensitive query params masked.

    Keeps the key/email out of logs and tracebacks (the error path prints the
    request URL). Values for :data:`_REDACT_PARAMS` become ``REDACTED``.
    """
    parts = urlsplit(url)
    if not parts.query:
        return url
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    redacted = [
        (k, "REDACTED" if k in _REDACT_PARAMS else v) for k, v in pairs
    ]
    return urlunsplit(parts._replace(query=urlencode(redacted)))


def query_openalex(params: dict) -> list[dict]:
    """Fetch all OpenAlex works for ``params``, following cursor pagination.

    Starts at ``cursor=*`` and follows ``meta.next_cursor`` until it is empty,
    accumulating ``results`` across pages. Stops early at
    :data:`OPENALEX_MAX_PAGES` (logging when the cap is hit) to bound result
    volume and LLM load. ``per_page`` is forced to :data:`OPENALEX_PER_PAGE`.

    Authentication: ``OPENALEX_API_KEY`` (if set) is sent as the ``api_key``
    query param, which OpenAlex requires for its $1/day free budget. The key is
    redacted from any URL included in a raised error so it never lands in logs.

    On a non-OK response the raised error includes the HTTP status and a snippet
    of the response body, so policy/plan/budget errors (which OpenAlex may serve
    as a 429 with a descriptive body) are not mistaken for plain rate limiting.
    """
    params = dict(params)
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    params["per_page"] = OPENALEX_PER_PAGE

    results: list[dict] = []
    cursor = "*"
    for page in range(OPENALEX_MAX_PAGES):
        # Fresh dict per request so each call gets its own cursor value.
        page_params = {**params, "cursor": cursor}
        # Eventually, log and skip the source on error. Right now, bubble it up
        # for dev — but with the body attached so the cause is diagnosable.
        response = requests.get(OPENALEX_BASE_URL, params=page_params)
        if not response.ok:
            snippet = (response.text or "")[:500]
            raise requests.exceptions.HTTPError(
                f"OpenAlex request failed: {response.status_code} "
                f"{response.reason} for url {_redact_url(response.url)} — "
                f"body: {snippet}",
                response=response,
            )
        data = response.json()
        results.extend(data.get("results", []))

        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    else:
        print(
            f"query_openalex: hit OPENALEX_MAX_PAGES={OPENALEX_MAX_PAGES} "
            f"(per_page={OPENALEX_PER_PAGE}); results beyond "
            f"{OPENALEX_MAX_PAGES * OPENALEX_PER_PAGE} works were not fetched"
        )

    return results


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


# ---------- CSV parsing and JSON writing ----------


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


def collect_data(
    now: datetime = datetime.now(),
    since: datetime | None = None,
    already_ingested: set[str] | None = None,
) -> list[dict]:
    """Fetch and tag entries from all configured sources.

    Returns untrimmed entries with an ``aaiid_data_source`` field.
    """
    already_ingested = already_ingested or set()
    data: list[dict] = []

    if NHTSA not in already_ingested:
        csv_text = fetch_csv(URL)
        data = parse_csv(csv_text)
        for row in data:
            row["aaiid_data_source"] = NHTSA
    else:
        print(f"collect_data: {NHTSA} already ingested; skipping")

    # Always scan a fixed rolling window rather than "since last run": OpenAlex
    # indexes many works after their publication_date, so a work published just
    # before the last run may only appear now. The window re-catches those; the
    # `since` arg is intentionally ignored for OpenAlex (upserts make re-fetching
    # harmless). from_publication_date is used because from_created_date is now
    # gated behind a paid OpenAlex plan (returns 429 "Plan upgrade required").
    from_date = (now - timedelta(days=OPENALEX_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    # (animal OR animals OR ...) AND (autonomous vehicle OR ...) expressed with
    # pipe-OR and repeated-filter-AND so it stays within the free-tier boolean
    # limit. See OPENALEX_ANIMAL_TERMS / OPENALEX_TECH_TERMS for the term lists.
    animal_clause = "|".join(OPENALEX_ANIMAL_TERMS)
    tech_clause = "|".join(OPENALEX_TECH_TERMS)
    openalex_filter = (
        f"from_publication_date:{from_date},"
        f"title_and_abstract.search:{animal_clause},"
        f"title_and_abstract.search:{tech_clause}"
    )
    params = {
        "sort": "publication_date:desc",
        "filter": openalex_filter,
        "select": "id,title,display_name,publication_year,publication_date,doi,abstract_inverted_index,primary_location,keywords,referenced_works,concepts",
    }
    # Failure isolation: an OpenAlex outage or a spent daily budget (429) must
    # not abort the whole ingest. Log and skip the source so NHTSA rows and
    # downstream classification/extraction of already-ingested rows still run.
    try:
        papers = query_openalex(params)
    except requests.exceptions.RequestException as exc:
        print(f"collect_data: OpenAlex fetch failed ({exc}); skipping source")
        papers = []

    for paper in papers:
        paper["aaiid_data_source"] = OPEN_ALEX
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
    if source == NHTSA:
        trimmed = trim_nhtsa_fields([entry])[0]
    elif source == OPEN_ALEX:
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
    if source == NHTSA:
        return entry.get("Report ID", "") or ""
    if source == OPEN_ALEX:
        return entry.get("id", "") or ""
    return ""


def map_direct_fields(entry: dict) -> dict:
    """Return model-column values derivable directly from source data.

    Sparse by design: each source returns only the fields it can provide.
    Only keys with non-empty, successfully-parsed values are included; unknown
    sources return ``{}``.
    """
    source = entry.get("aaiid_data_source")
    fields: dict = {}

    if source == NHTSA:
        # No per-report public URL exists; the bulk SGO CSV is the source.
        fields["url"] = URL
        city = entry.get("City", "")
        if city:
            fields["city"] = city
        entity = entry.get("Reporting Entity", "")
        if entity:
            fields["ai_system_manufacturer"] = entity
        raw_date = entry.get("Incident Date", "")
        if raw_date:
            try:
                fields["time_occurred"] = datetime.strptime(raw_date, "%b-%Y").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                print(
                    f"map_direct_fields: could not parse Incident Date "
                    f"{raw_date!r} as '%b-%Y'; omitting time_occurred"
                )
    elif source == OPEN_ALEX:
        url = (
            _get_dotted(entry, "primary_location.landing_page_url")
            or _get_dotted(entry, "primary_location.pdf_url")
            or entry.get("id")
        )
        if url:
            fields["url"] = url
        title = entry.get("display_name") or entry.get("title")
        if title:
            fields["title"] = title
        pub_date = entry.get("publication_date", "")
        if pub_date:
            try:
                fields["time_reported"] = datetime.strptime(
                    pub_date, "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                print(
                    f"map_direct_fields: could not parse publication_date "
                    f"{pub_date!r} as '%Y-%m-%d'; omitting time_reported"
                )
    return fields


def prepare_entry(entry: dict) -> dict:
    """Return the DB-ready projection of a raw entry.

    Keys: aaiid_data_source, entry_id (may be absent), source_id, json_blob.
    """
    return {
        "aaiid_data_source": entry.get("aaiid_data_source", ""),
        "entry_id": entry.get("entry_id", ""),
        "source_id": _source_id(entry),
        "json_blob": _trim_entry(entry),
    }


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
            **prepare_entry(entry),
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







