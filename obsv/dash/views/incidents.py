"""The public incident table and per-incident detail page.

Only ``approved`` incidents are public. The list view defaults to approved rows
but lets an explicit ``?status=`` query param override that base when
``settings.DEBUG`` is on (handy for browsing other statuses during development);
in production the base filter is fixed. The detail view is hard-scoped to
approved rows so non-approved pks 404.
"""

from django.conf import settings
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render

from ..filters import IncidentFilter
from ..models import IncidentReport
from .helpers import PAGE_SIZE, row_context


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
