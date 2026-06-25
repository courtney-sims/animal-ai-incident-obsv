import json
from datetime import datetime, timedelta

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


def parse_csv(text: str) -> list[dict]:
    if not text.strip():
        return []
    data = tablib.Dataset()
    data.load(text, format="csv")
    return list(data.dict)


def write_json(data: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def collect_data() -> list[dict]:
    csv_text = fetch_csv(URL)
    data = parse_csv(csv_text)
    for row in data:
        row["aaiid_data_source"] = "nhtsa_incident_report"

    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "sort": "publication_date:desc",
        "per_page": 25,
        "search": "animal",
        "filter": f"from_publication_date:{thirty_days_ago}",
        "select": "id",
    }
    papers = query_openalex(params)
    for paper in papers:
        paper["aaiid_data_source"] = "openalex_work"

    return data + papers


def main() -> None:
    write_json(collect_data(), OUTPUT_PATH)


if __name__ == "__main__":
    main()
