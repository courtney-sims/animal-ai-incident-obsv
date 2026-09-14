"""Shared display helpers for the public views.

Turns an ``IncidentReport`` into a display-ready dict, preferring structured
fields but falling back to NHTSA ``raw_data`` keys so unhydrated rows still
render usefully.
"""

from ..models import IncidentReport

PAGE_SIZE = 25
SUMMARY_TRUNCATE = 280

SOURCE_LABELS = {
    IncidentReport.SourceType.NHTSA: "NHTSA",
    IncidentReport.SourceType.OPENALEX: "OpenAlex",
    IncidentReport.SourceType.USER: "User Submitted",
}


def _first_nonempty(*values):
    for value in values:
        if value not in (None, "", -1):
            return value
    return None


def row_context(incident):
    """Build a display dict for an incident with raw-data fallbacks.

    Structured fields win when populated; otherwise we fall back to the NHTSA
    ``raw_data`` keys so the (currently unhydrated) rows still render usefully.
    """
    raw = incident.raw_data or {}

    # --- title ---
    crash_with = raw.get("Crash With")
    city = raw.get("City")
    if incident.title:
        title = incident.title
    elif crash_with and city:
        title = f"{crash_with} in {city}"
    elif crash_with:
        title = crash_with
    else:
        title = incident.source_id

    # --- summary ---
    summary = incident.description or raw.get("Narrative") or ""
    truncated = False
    if len(summary) > SUMMARY_TRUNCATE:
        summary_short = summary[:SUMMARY_TRUNCATE].rstrip() + "\u2026"
        truncated = True
    else:
        summary_short = summary

    # --- location ---
    location_parts = [
        incident.city or raw.get("City"),
        raw.get("State"),
        incident.country,
    ]
    location = ", ".join(p for p in location_parts if p)

    # --- ai system ---
    ai_system = _first_nonempty(incident.ai_system, raw.get("Reporting Entity")) or ""

    # --- animal / impact ---
    has_structured_animal = (
        incident.animal_type != IncidentReport.AnimalType.unknown
        or incident.animal_species
        or incident.animal_count not in (-1, None)
    )
    if has_structured_animal:
        parts = []
        if incident.animal_type != IncidentReport.AnimalType.unknown:
            parts.append(incident.get_animal_type_display().lower())
        species = incident.animal_species
        if species and incident.animal_count not in (-1, None):
            parts.append(f"{species} ({incident.animal_count})")
        elif species:
            parts.append(species)
        elif incident.animal_count not in (-1, None):
            parts.append(f"({incident.animal_count})")
        animal = " \u00b7 ".join(parts)
    else:
        animal = crash_with or "Animal"

    # --- date ---
    date = _first_nonempty(incident.time_occurred, incident.time_reported)

    return {
        "title": title,
        "summary": summary,
        "summary_short": summary_short,
        "summary_truncated": truncated,
        "location": location,
        "ai_system": ai_system,
        "animal": animal,
        "date": date,
        "source_label": SOURCE_LABELS.get(incident.source, incident.source),
    }
