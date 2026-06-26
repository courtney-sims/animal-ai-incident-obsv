import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import requests
import tablib


URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"
OUTPUT_PATH = "incidents.json"
OPENALEX_BASE_URL = "https://api.openalex.org/works"


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


def filter_by_date(data: list[dict], months_back: int = 1) -> list[dict]:
    now = datetime.now()
    result = []
    for row in data:
        date_str = row.get("Incident Date", "")
        if not date_str or not date_str.strip():
            result.append(row)
            print(f'Could not get incident date from {row}')
            continue
        try:
            dt = datetime.strptime(date_str.strip(), "%b-%Y")
            months_diff = (now.year - dt.year) * 12 + (now.month - dt.month)
            if months_diff < months_back:
                result.append(row)
        except ValueError:
            print(f'Could not parse month and year from {dt}')
    return result


def parse_csv(text: str) -> list[dict]:
    if not text.strip():
        return []
    data = tablib.Dataset()
    data.load(text, format="csv")
    return list(data.dict)


def write_json(data: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def collect_data(trim: bool = False) -> list[dict]:
    csv_text = fetch_csv(URL)
    data = parse_csv(csv_text)
    data = filter_by_date(data)
    for row in data:
        row["aaiid_data_source"] = "nhtsa_incident_report"
    if trim:
        data = trim_nhtsa_fields(data)

    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
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

    return data + papers


def write_timestamped_csv(content: str, directory: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(directory) / f"animal_incidents_{timestamp}.csv"
    path.write_text(content)
    return str(path)


def run_classification(incidents_path: str, prompt: str) -> str:
    workdir = Path(incidents_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"animal_incidents_{timestamp}.csv"
    output_path = str(workdir / output_name)

    full_prompt = f"{prompt}\n\n6. Name the output file {output_name}."
    subprocess.run(
        ["opencode", "run", full_prompt, "-f", incidents_path],
        cwd=str(workdir),
        check=True,
    )

    return output_path


def generate_prompt() -> str:
    return """\
Review the file incidents.json and produce a CSV of animal-related AI incidents.

1. Load the JSON file (it's a JSON array of objects).

2. For each object with "aaiid_data_source": "nhtsa_incident_report", include it if:
   - "Crash With" is "Animal", OR
   - The "Narrative" field mentions an animal keyword (e.g. dog, cat, raccoon, bird, duck, flock, domestic animal, etc.)
   - Exclude false positives: "HAWK" (pedestrian beacon acronym, not a bird) should not count.

3. For each object with "aaiid_data_source": "openalex_work", read its title and abstract. Keep it only if it's actually about AI or autonomous vehicles interacting with animals. Drop general animal health, animal law, animal testing, agriculture, or other unrelated topics.

4. For every kept entry, trim the JSON blob to only these relevant fields:
   - NHTSA entries: Report ID, Report Version, Reporting Entity, Crash With, Narrative, City, State, Incident Date, SV Precrash Speed (MPH), Highest Injury Severity Alleged
   - OpenAlex entries: id, title, display_name, publication_year, publication_date, primary_location.landing_page_url, primary_location.pdf_url

5. Write the CSV to the current directory with columns: aaiid_data_source, json_blob, reasoning, confidence_score. The json_blob column should contain the trimmed JSON (compact, no extra whitespace). The reasoning column should explain why the entry was included. Use confidence_score of "High" for entries with "Crash With": "Animal", "Medium" for narrative-only matches, and "Low" for OpenAlex entries where relevance is uncertain.
"""


def run_pipeline() -> str:
    data = collect_data(trim=True)
    write_json(data, OUTPUT_PATH)
    prompt = generate_prompt()
    output_path = run_classification(OUTPUT_PATH, prompt)
    return output_path


def main() -> None:
    write_json(collect_data(), OUTPUT_PATH)


if __name__ == "__main__":
    main()
