"""Debug CLI for the incident pipeline.

This module is a **file-only debug harness**: it fetches, classifies, and writes
the intermediate/result JSON files to disk so a run can be inspected by hand. It
**never touches the database**. The production path — the one that persists to
:class:`~dash.models.IncidentReport` — is the ``run_pipeline`` *management
command* (``obsv/dash/management/commands/run_pipeline.py``).
"""

from datetime import datetime
from pathlib import Path
import sys

from pipeline import incident_classifier
from pipeline import incident_fetcher


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

def get_data_paths(
        workdir: Path,
        now: datetime | None = None,
    ):
    if now is None:
        now = datetime.now()
    if workdir is None:
        workdir = Path(".")
    workdir = Path(workdir)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    incidents_path = workdir / f"incidents_{timestamp}.json"
    judgments_path = workdir / f"judgments_{timestamp}.json"

    return incidents_path, judgments_path

def run_pipeline(
    model: str | None = None,
    now: datetime | None = None,
    workdir: Path | None = None,
) -> tuple[list[dict], str]:
    """Collect and classify data; return the kept records and resolved model.

    Debug CLI only — never touches the database. Production path is the
    ``run_pipeline`` management command.

    Writes the intermediate ``incidents``/``judgments`` JSON files into
    ``workdir`` (these are the IPC channel with the ``opencode`` subprocess) and
    returns ``(records, model)`` where ``records`` are the reconciled, trimmed
    kept judgments as produced by :func:`incident_fetcher.build_records`.
    """
    if now is None:
        now = datetime.now()
    if workdir is None:
        workdir = Path(".")
    workdir = Path(workdir)

    entries = incident_fetcher.collect_data(now=now)
    incident_fetcher.assign_entry_ids(entries)
    incidents_path, judgments_path = get_data_paths(workdir, now=now)
    incident_fetcher.write_json(entries, incidents_path)
    model, judgments = incident_classifier.classify(workdir, incidents_path, judgments_path, model)
    records = incident_fetcher.build_records(entries, judgments, model=model)
    return records, model

def main(argv: list[str] | None = None) -> None:
    """Run the debug pipeline and write its outputs to disk.

    Debug CLI only — never touches the database; the production path is the
    ``run_pipeline`` management command. In addition to the intermediate
    ``incidents``/``judgments`` files written by :func:`run_pipeline`, the built
    records are written to ``records_<timestamp>.json`` in the workdir so the
    run's output is inspectable.
    """
    args = parse_cli_args(argv if argv is not None else sys.argv[1:])

    now = datetime.now()
    workdir = Path(".")
    records, _model = run_pipeline(model=args.get("model"), now=now, workdir=workdir)

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    records_path = workdir / f"records_{timestamp}.json"
    incident_fetcher.write_json(records, records_path)

if __name__ == "__main__":
    main()

