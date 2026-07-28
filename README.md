# Animal-AI Incident Observer

The Animal AI Incident Observatory is a public platform for tracking, cataloguing, and disseminating incidents in which AI systems caused direct harm to non-human animals.

This is just a POC so far for programmatically pulling data from a couple of sources and using LLM relevance evaluation to filter for AI->Animal direct harm incidents. The output is a csv listing each incident deemed relevant along with their source, the LLM's reasoning for considering that incident relevant, and a qualitative confidence score.

## How it works

Data sources currently included:
1. **NHTSA SGO-2021-01** — Standing General Order crash reports from AV operators (Waymo, Tesla, Zoox, Avride, etc.).
2. **OpenAlex** — Academic works matching a search for `"animal"`, filtered to the last 30 days.

The pipeline runs in three steps:

1. **Collect.** Python fetches from NHTSA and OpenAlex, tags each entry with its source, and assigns a synthetic `entry_id`.
2. **Classify.** An LLM agent (via `opencode run`) reads the combined JSON and writes a JSON array of judgments — one per input entry — with fields `entry_id`, `keep`, `reasoning`, `confidence`. The LLM only classifies; it does not write the final CSV.
3. **Assemble.** Python joins entries with judgments, trims fields per source, and writes `animal_incidents_<TIMESTAMP>.csv` with columns:
   - `aaiid_data_source` — `nhtsa_incident_report` or `openalex_work`
   - `entry_id` — synthetic ID linking back to the input
   - `json_blob` — trimmed JSON of the entry
   - `reasoning` — why the entry was included
   - `confidence` — `High` / `Medium` / `Low`
   - `llm_meta` — provenance JSON, currently `{"model": "..."}`

Judgments that fail validation (bad shape, unknown `entry_id`, duplicates, missing fields) are logged to stdout and skipped. The pipeline does not abort on validation problems, but affected entries simply won't appear in the CSV.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires [opencode](https://opencode.ai) available on `PATH`.

### Local database setup

#### Installation
```bash
sudo apt install postgresql
sudo apt install postgresql-client-common
```

#### Configuration
```bash
sudo -u postgres psql
```

```psql
CREATE DATABASE animal_ai_obsv;
CREATE USER admin WITH PASSWORD 'test';
GRANT ALL PRIVILEGES ON DATABASE animal_ai_obsv TO admin;
```

```bash
python manage.py migrate
```

## Usage

### Observatory App

```bash
python obsv/manage.py runserver
```

Go to http://127.0.0.1:8000/ in browser.

#### Django API

```bash
python manage.py shell
```

### Data Pipeline

```bash
python pipeline/main.py
```

#### Choosing a model

The classification step runs `opencode run -m <model>`. By default the pipeline
uses `opencode/deepseek-v4-flash-free` — a free-tier model bundled with
opencode. Override on the command line with a `model=<provider/model>` arg:

```bash
python incident_fetcher.py model=anthropic/claude-sonnet-4-5
```

You must have the corresponding provider authenticated (`opencode auth login <provider>`) and the
model available (`opencode models` to list). The model used for each run is recorded in the `llm_meta` column of every output row.

### Database
From the obsv/ directory:

```bash
sudo -u postgres psql
```

#### Create Migrations

```bash
python manage.py makemigrations dash
```

#### Review Migrations
```bash
python manage.py sqlmigrate dash insert-mig-num-here
python manage.py check
```

#### Apply Migrations
```bash
python manage.py migrate
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

