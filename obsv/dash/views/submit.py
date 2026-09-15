"""Public incident submission -> a ``pending`` user-submitted report."""

import uuid
from datetime import datetime

from django.shortcuts import render
from django.utils import timezone

from ..forms import IncidentSubmitForm
from ..models import IncidentReport


def incident_submit(request):
    """Public incident submission -> a ``pending`` user-submitted report.

    Honeypot-filled POSTs are dropped. Valid submissions land in the admin review queue like any other
    pending row, tagged ``source=user_submitted``.
    """
    if request.method == "POST":
        form = IncidentSubmitForm(request.POST)
        if form.is_valid():
            # Honeypot tripped: pretend success, save nothing.
            if form.cleaned_data.get("website"):
                return render(
                    request, "dash/submit.html", {"form": IncidentSubmitForm(), "success": True}
                )
            _create_user_incident(form.cleaned_data)
            return render(
                request, "dash/submit.html", {"form": IncidentSubmitForm(), "success": True}
            )
    else:
        form = IncidentSubmitForm()
    return render(request, "dash/submit.html", {"form": form, "success": False})


def _create_user_incident(data):
    date_occurred = data.get("date_occurred")
    animal_type = data.get("animal_type") or IncidentReport.AnimalType.unknown
    animal_count = data.get("animal_count")

    # raw_data must stay JSON-serializable: serialize the date to ISO.
    raw_data = {
        "url": data.get("url", ""),
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "date_occurred": date_occurred.isoformat() if date_occurred else "",
        "animal_type": animal_type,
        "animal_species": data.get("animal_species", ""),
        "animal_count": animal_count if animal_count is not None else "",
        "harm_description": data.get("harm_description", ""),
        "ai_system": data.get("ai_system", ""),
        "city": data.get("city", ""),
        "country": data.get("country", ""),
        "submitter_email": data.get("submitter_email", ""),
    }

    return IncidentReport.objects.create(
        source=IncidentReport.SourceType.USER,
        source_id=uuid.uuid4().hex,
        raw_data=raw_data,
        title=data.get("title", "") or "",
        description=data.get("description", "") or "",
        url=data.get("url", "") or "",
        animal_type=animal_type,
        animal_species=data.get("animal_species", "") or "",
        animal_count=animal_count if animal_count is not None else -1,
        harm_description=data.get("harm_description", "") or "",
        ai_system=data.get("ai_system", "") or "",
        city=data.get("city", "") or "",
        country=data.get("country", "") or "",
        time_occurred=(
            timezone.make_aware(
                datetime(
                    date_occurred.year, date_occurred.month, date_occurred.day
                )
            )
            if date_occurred
            else None
        ),
        time_ingested=timezone.now(),
        status=IncidentReport.StatusType.pending,
    )
