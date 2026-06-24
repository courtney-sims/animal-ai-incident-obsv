import json

import requests
import tablib


URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"
OUTPUT_PATH = "incidents.json"


def fetch_csv(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_csv(text: str) -> list[dict]:
    if not text.strip():
        return []
    data = tablib.Dataset()
    data.load(text, format="csv")
    return list(data.dict)


def write_json(data: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    csv_text = fetch_csv(URL)
    data = parse_csv(csv_text)
    write_json(data, OUTPUT_PATH)


if __name__ == "__main__":
    main()
