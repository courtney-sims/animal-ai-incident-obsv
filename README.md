# Animal-AI Incident Observer

The Animal AI Incident Observatory is a public platform for tracking, cataloguing, and disseminating incidents in which AI systems caused direct harm to non-human animals.

This is just a POC so far for programmatically pulling data from a couple of sources and using LLM relevance evaluation to filter for AI->Animal direct harm incidents. The output is a csv listing each incident deemed relevant along with their source, the LLM's reasoning for considering that incident relevant, and a qualitative confidence score.

## How it works

Data sources currently included:
1. **NHTSA SGO-2021-01** — Standing General Order crash reports from AV operators (Waymo, Tesla, Zoox, Avride, etc.).
2. **OpenAlex** — Academic works matching a search for `"animal"`, filtered to the last 30 days.

An LLM agent (via `opencode run`) reads the combined JSON of entries pulled from the data sources, classifies each entry as animal&AI-related or not, and writes relevant entries to `animal_incidents_<TIMESTAMP>.csv` with columns:
- `aaiid_data_source` — `nhtsa_incident_report` or `openalex_work`
- `json_blob` — trimmed JSON of the entry
- `reasoning` — why the entry was included
- `confidence_score` — High / Medium / Low
- `llm_meta` — provenance JSON added by the pipeline (currently `{"model": "..."}`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires [opencode](https://opencode.ai) available on `PATH`.

## Usage

```bash
python incident_fetcher.py
```

### Choosing a model

The classification step runs `opencode run -m <model>`. By default the pipeline
uses `opencode/deepseek-v4-flash-free` — a free-tier model bundled with
opencode. Override on the command line with a `model=<provider/model>` arg:

```bash
python incident_fetcher.py model=anthropic/claude-sonnet-4-5
```

You must have the corresponding provider authenticated (`opencode auth login <provider>`) and the
model available (`opencode models` to list). The model used for each run is recorded in the `llm_meta` column of every output row.

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
requirements.txt
```
