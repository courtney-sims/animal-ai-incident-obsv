# Animal-AI Incident Observer

The Animal AI Incident Observatory is a public platform for tracking, cataloguing, and disseminating incidents in which AI systems caused direct harm to non-human animals.

This is just a POC so far for programmatically pulling data from a couple of sources and using LLM relevance evaluation to filter for AI->Animal direct harm incidents. The output is a csv listing each incident deemed relevant along with their source, the LLM's reasoning for considering that incident relevant, and a qualitative confidence score.

## How it works

Data sources currently included:
1. **NHTSA SGO-2021-01** — Standing General Order crash reports from AV operators (Waymo, Tesla, Zoox, Avride, etc.).
2. **OpenAlex** — Academic works matching a search for `"animal"`, filtered to the last 30 days.

The pipeline runs in three steps:

1. **Collect.** Python fetches from data sources and adds each entry as "new" to the db.
2. **Classify.** An LLM agent (via `opencode run`) looks at all "new" entries and determines whether or not they are relevant.
3. **Hydrate.** An LLM agent (via `opencode run`) looks at all "llm_relevant" entries and fills out additional data fields based on the incident as described from the source and marks hydrated entries as "pending".

Human intervention is needed at the final step:
4. **Review.** Humans can mark "pending" entries as either "approved" or "rejected".

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
   python obsv/manage.py run_pipeline [--model provider/model] [--workdir PATH]
```

This django command fetches data from all sources and populates the database with LLM judgments. `--workdir` preserves intermediate JSON files for debugging.

For django-free, database-free debugging, run:

```bash
python pipeline/main.py
```

#### Choosing a model

The classification step runs `opencode run -m <model>`. By default the pipeline
uses `opencode/deepseek-v4-flash-free` — a free-tier model bundled with
opencode. Override on the command line with a `model=<provider/model>` arg:

```bash
python -m pipeline/main.py model=anthropic/claude-sonnet-4-5
```

You must have the corresponding provider authenticated (`opencode auth login <provider>`) and the
model available (`opencode models` to list).

#### LLM integration: opencode + files (for now)

The pipeline invokes the LLM by shelling out to `opencode run`, passing input
entries and receiving judgments via JSON files as the IPC channel. This was
chosen deliberately for the POC: opencode handles provider auth and gives
access to a free-tier model, and having the agent write a file produces a
clean, validatable artifact. The file I/O overhead is negligible next to LLM
latency.

If/when the project is funded, the plan is to switch to direct provider API
calls with structured (JSON-mode) output — fully in-memory, schema-enforced,
and without the subprocess boundary.

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
from pipeline.prompt_eval import run_eval

metrics = run_eval()
print(metrics)  # {"precision": 1.0, "recall": 0.67, "f1": 0.8, ...}
```

The test set has 6 entries (4 NHTSA + 2 OpenAlex), with 3 expected as animal-related. The eval shells to `opencode run`, then scores precision, recall, and F1.

## Tests

```bash
source .venv/bin/activate
python -m pytest -v
```

