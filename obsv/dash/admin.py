"""Admin as the human-review step of the pipeline.

Staff review the ``pending`` queue (the status the pipeline leaves rows in
after LLM extraction) and can view, edit, approve, or reject each incident.
Approve/reject records the reviewer and timestamp and — because the public
list shows only ``approved`` rows — approving a row is what publishes it.

Scope note: :meth:`IncidentReportAdmin.get_queryset` filters to ``pending``,
so approved/rejected rows disappear from the admin entirely. Corrections to an
already-approved incident must be made via the shell for now (see the plan's
§7). ``reviewer``/``time_reviewed`` are set *only* by the approve/reject paths,
never by a plain save.
"""

from django.contrib import admin
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html

from .models import IncidentReport


@admin.register(IncidentReport)
class IncidentReportAdmin(admin.ModelAdmin):
    list_display = (
        "display_title",
        "source",
        "animal_type",
        "city",
        "time_occurred",
        "time_hydrated",
        "reviewer",
        "review_link",
    )
    list_filter = ("source", "animal_type", "confidence")
    # Plain text fields only — no JSON lookups in search_fields.
    search_fields = (
        "source_id",
        "title",
        "description",
        "city",
        "country",
        "ai_system",
        "ai_system_manufacturer",
        "animal_species",
    )
    # Oldest waiting incidents surface first so the queue is drained in order.
    ordering = ("time_hydrated",)

    readonly_fields = (
        "source",
        "source_id",
        "raw_data",
        "time_occurred",
        "time_reported",
        "time_ingested",
        "time_judged",
        "time_hydrated",
        "time_reviewed",
    )

    fieldsets = (
        ("Content", {
            "fields": ("title", "description", "url"),
        }),
        ("Animal", {
            "fields": ("animal_type", "animal_species", "animal_count"),
        }),
        ("Harm", {
            "fields": ("harm_type", "harm_description"),
        }),
        ("AI system", {
            "fields": ("ai_system", "ai_system_manufacturer"),
        }),
        ("Location", {
            "fields": ("city", "country"),
        }),
        ("Judgment / review", {
            "fields": (
                "status",
                "confidence",
                "llm_reasoning",
                "llm_model",
                "reviewer",
                "time_reviewed",
            ),
        }),
        ("Provenance (read-only)", {
            "classes": ("collapse",),
            "fields": (
                "source",
                "source_id",
                "raw_data",
                "time_occurred",
                "time_reported",
                "time_ingested",
                "time_judged",
                "time_hydrated",
            ),
        }),
    )

    actions = ("approve_selected", "reject_selected")

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        # The FK is null=True but not blank=True, so its form field is required
        # by default. A pending row legitimately has no reviewer yet (it's set
        # by approve/reject), so make it optional here.
        if "reviewer" in form.base_fields:
            form.base_fields["reviewer"].required = False
        return form

    def get_queryset(self, request):
        # Restrict the admin to the pending review queue. As a result,
        # approved/rejected rows become invisible here (see module docstring).
        qs = super().get_queryset(request)
        return qs.filter(status=IncidentReport.StatusType.pending)

    # --- display helpers ---------------------------------------------------

    @admin.display(description="Title")
    def display_title(self, obj):
        """Non-empty title, else a raw-data fallback, else the source id."""
        if obj.title:
            return obj.title
        raw = obj.raw_data or {}
        crash_with = raw.get("Crash With")
        city = raw.get("City")
        if crash_with and city:
            return f"{crash_with} in {city}"
        if crash_with:
            return crash_with
        return obj.source_id

    @admin.display(description="Review")
    def review_link(self, obj):
        url = reverse("admin:dash_incidentreport_review", args=[obj.pk])
        return format_html('<a href="{}">Review</a>', url)

    # --- bulk actions ------------------------------------------------------

    def _bulk_set_status(self, request, queryset, status):
        now = timezone.now()
        with transaction.atomic():
            count = queryset.update(
                status=status,
                reviewer=request.user,
                time_reviewed=now,
            )
        return count

    @admin.action(description="Approve selected incidents")
    def approve_selected(self, request, queryset):
        count = self._bulk_set_status(
            request, queryset, IncidentReport.StatusType.approved
        )
        self.message_user(request, f"{count} incident(s) approved.")

    @admin.action(description="Reject selected incidents")
    def reject_selected(self, request, queryset):
        count = self._bulk_set_status(
            request, queryset, IncidentReport.StatusType.rejected
        )
        self.message_user(request, f"{count} incident(s) rejected.")

    # --- per-row Review page ----------------------------------------------

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path(
                "<path:object_id>/review/",
                self.admin_site.admin_view(self.review_view),
                name="dash_incidentreport_review",
            ),
        ]
        # Custom URL must precede the catch-all change view.
        return custom + urls

    def review_view(self, request, object_id):
        if not self.has_change_permission(request):
            from django.core.exceptions import PermissionDenied
            raise PermissionDenied

        # Scope lookup to the pending queue -> 404 for missing/non-pending.
        obj = get_object_or_404(self.get_queryset(request), pk=object_id)
        ModelForm = self.get_form(request, obj, change=True)

        if request.method == "POST":
            form = ModelForm(request.POST, request.FILES, instance=obj)
            if form.is_valid():
                incident = form.save(commit=False)
                now = timezone.now()
                if "approve" in request.POST:
                    incident.status = IncidentReport.StatusType.approved
                    incident.reviewer = request.user
                    incident.time_reviewed = now
                    msg = "Incident approved."
                elif "reject" in request.POST:
                    incident.status = IncidentReport.StatusType.rejected
                    incident.reviewer = request.user
                    incident.time_reviewed = now
                    msg = "Incident rejected."
                else:
                    # Plain save keeps the row pending and does not touch
                    # reviewer/time_reviewed.
                    incident.status = IncidentReport.StatusType.pending
                    msg = "Incident saved."
                with transaction.atomic():
                    incident.save()
                    form.save_m2m()
                self.message_user(request, msg)
                changelist = reverse("admin:dash_incidentreport_changelist")
                return redirect(changelist)
        else:
            form = ModelForm(instance=obj)

        admin_form = admin.helpers.AdminForm(
            form, list(self.get_fieldsets(request, obj)),
            self.get_prepopulated_fields(request, obj),
            self.get_readonly_fields(request, obj),
            model_admin=self,
        )

        context = {
            **self.admin_site.each_context(request),
            "title": "Review incident",
            "opts": self.model._meta,
            "original": obj,
            "adminform": admin_form,
            "form": form,
            "media": self.media + admin_form.form.media,
            "changelist_url": reverse("admin:dash_incidentreport_changelist"),
            "raw_items": sorted((obj.raw_data or {}).items()),
        }
        return TemplateResponse(
            request, "admin/dash/incidentreport/review.html", context
        )
