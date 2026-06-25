# Animal-AI Incident Observer

Identify NHTSA autonomous-vehicle crash reports involving non-human animals, using an LLM pipeline via `opencode run`.

## How it works

Two data sources are combined into `incidents.json`:

1. **NHTSA SGO-2021-01** — Standing General Order crash reports from AV operators (Waymo, Tesla, Zoox, Avride, etc.). ~80 raw CSV columns are trimmed to a focused set (Report ID, Crash With, Narrative, City, State, etc.) for LLM-friendly input.

2. **OpenAlex** — Academic works matching a search for `"animal"`, filtered to the last 30 days. Includes title, abstract, and metadata for relevance judgment.

An LLM agent (via `opencode run`) reads the combined JSON, classifies each entry as animal-related or not, and writes `animal_incidents_<TIMESTAMP>.csv` with columns:
- `aaiid_data_source` — `nhtsa_incident_report` or `openalex_work`
- `json_blob` — trimmed JSON of the entry
- `reasoning` — why the entry was included
- `confidence_score` — High / Medium / Low

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires [opencode](https://opencode.ai) available on `PATH`.

## Usage

### Collect data only

```bash
python incident_fetcher.py   # writes incidents.json (full fields)
```

### Full pipeline (collect + classify)

```python
from incident_fetcher import run_pipeline

csv_path = run_pipeline()           # produces animal_incidents_<TS>.csv
```

### Interactive classification

```bash
# After running `python incident_fetcher.py`:
opencode run "$(python -c 'from incident_fetcher import generate_prompt; print(generate_prompt())')" -f incidents.json
```

## Prompt eval

Evaluate how well the prompt performs against a small labeled test set:

```python
from prompt_eval import run_eval

metrics = run_eval()
print(metrics)  # {"precision": 1.0, "recall": 0.67, "f1": 0.8, ...}
```

The test set has 6 entries (4 NHTSA + 2 OpenAlex), with 3 expected as animal-related. The eval shells to `opencode run`, then scores precision, recall, and F1.

## Tests

```bash
source .venv/bin/activate
python -m pytest -v
```

## Project structure

```
incident_fetcher.py     — data collection, field trimming, prompt generation, opencode runner
prompt_eval.py          — test data builder, scoring metrics, eval orchestrator
test_incident_fetcher.py
test_prompt_eval.py
incidents.json.bak      — example combined output
requirements.txt
```
