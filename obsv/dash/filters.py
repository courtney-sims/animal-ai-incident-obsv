"""Server-side filtering for the public incident list.

``IncidentFilter`` drives the incident table: a free-text ``q`` search across the
structured content fields plus the NHTSA ``Narrative`` raw-data key, and discrete
filters for source, animal type, status, and an occurred-date range.
"""

import django_filters
from django.db.models import Q

from .models import IncidentReport


class IncidentFilter(django_filters.FilterSet):
    # Free-text search OR'd across the human-facing content fields. The
    # ``raw_data__Narrative`` JSON key lookup covers NHTSA rows whose structured
    # fields are still empty; on non-NHTSA rows the key is simply absent and the
    # lookup is a safe no-op (no error) in Postgres.
    q = django_filters.CharFilter(
        method="filter_q",
        label="Search",
    )
    source = django_filters.ChoiceFilter(
        choices=IncidentReport.SourceType.choices,
        label="Source",
    )
    animal_type = django_filters.ChoiceFilter(
        choices=IncidentReport.AnimalType.choices,
        label="Animal type",
    )
    status = django_filters.ChoiceFilter(
        choices=IncidentReport.StatusType.choices,
        label="Status",
    )
    date_from = django_filters.DateFilter(
        field_name="time_occurred",
        lookup_expr="gte",
        label="From",
    )
    date_to = django_filters.DateFilter(
        field_name="time_occurred",
        lookup_expr="lte",
        label="To",
    )

    class Meta:
        model = IncidentReport
        fields = ["q", "source", "animal_type", "status", "date_from", "date_to"]

    def filter_q(self, queryset, name, value):
        value = (value or "").strip()
        if not value:
            return queryset
        lookups = (
            "title",
            "description",
            "harm_description",
            "animal_species",
            "ai_system",
            "ai_system_manufacturer",
            "city",
            "country",
            "raw_data__Narrative",
        )
        query = Q()
        for field in lookups:
            query |= Q(**{f"{field}__icontains": value})
        return queryset.filter(query)
