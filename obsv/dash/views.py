"""Public-facing views: a filterable, paginated incident table and a citable
per-incident detail page.

Only ``approved`` incidents are public. The list view defaults to approved rows
but lets an explicit ``?status=`` query param override that base when
``settings.DEBUG`` is on (handy for browsing other statuses during development);
in production the base filter is fixed. The detail view is hard-scoped to
approved rows so non-approved pks 404.
"""

from django.conf import settings
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render

from .filters import IncidentFilter
from .models import IncidentReport

PAGE_SIZE = 25
SUMMARY_TRUNCATE = 280

SOURCE_LABELS = {
    IncidentReport.SourceType.NHTSA: "NHTSA",
    IncidentReport.SourceType.OPENALEX: "OpenAlex",
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


def incident_list(request):
    query_params = request.GET.copy()

    base_qs = IncidentReport.objects.all()
    # Default to approved. The ``?status=`` override is a dev-only convenience;
    # in production the public list always shows approved rows and the status
    # query param is ignored.
    if settings.DEBUG and query_params.get("status"):
        pass
    else:
        query_params.pop("status", None)
        base_qs = base_qs.filter(status=IncidentReport.StatusType.approved)
    # Newest occurrences first, nulls last; fall back to ingest order.
    base_qs = base_qs.order_by("-time_occurred", "-time_ingested")

    f = IncidentFilter(query_params, queryset=base_qs)

    paginator = Paginator(f.qs, PAGE_SIZE)
    page_number = request.GET.get("page") or 1
    try:
        page_obj = paginator.page(page_number)
    except Exception:
        page_obj = paginator.page(paginator.num_pages)

    rows = [
        {"incident": incident, "display": row_context(incident)}
        for incident in page_obj
    ]

    # Preserve the current filter query params (minus ``page``) for pagination.
    querydict = query_params
    querydict.pop("page", None)
    base_query = querydict.urlencode()

    total_count = IncidentReport.objects.count()
    approved_count = IncidentReport.objects.filter(
        status=IncidentReport.StatusType.approved
    ).count()

    context = {
        "filter": f,
        "page_obj": page_obj,
        "paginator": paginator,
        "rows": rows,
        "base_query": base_query,
        "filtered_count": paginator.count,
        "total_count": total_count,
        "approved_count": approved_count,
        "is_dev": settings.DEBUG,
    }
    return render(request, "dash/incident_list.html", context)


def incident_detail(request, pk):
    incident = get_object_or_404(
        IncidentReport, pk=pk, status=IncidentReport.StatusType.approved
    )
    display = row_context(incident)

    confidence_label = None
    if incident.confidence:
        confidence_label = IncidentReport.ConfidenceLevel(incident.confidence).label

    context = {
        "incident": incident,
        "display": display,
        "confidence_label": confidence_label,
        "raw_items": sorted((incident.raw_data or {}).items()),
    }
    return render(request, "dash/incident_detail.html", context)
