from django.conf import settings
from django.db import models

#A reporting of harm to animals by AI
class IncidentReport(models.Model):
    #Which pipeline source produced this record
    class SourceType(models.TextChoices):
        NHTSA = "nhtsa_incident_report", "NHTSA Incident Report"
        OPENALEX = "openalex_work", "OpenAlex Work"

    #LLM judgment confidence mapped to a plain int (percentage later)
    class ConfidenceLevel(models.IntegerChoices):
        UNKNOWN = 0, "Unknown"
        LOW = 1, "Low"
        MEDIUM = 2, "Medium"
        HIGH = 3, "High"

    #Maps the LLM's qualitative confidence to ConfidenceLevel ints; default 0/unknown
    CONFIDENCE_MAP = {"High": 3, "Medium": 2, "Low": 1}

    #--- Provenance (set at ingest; required) ---

    #Pipeline source that produced this record
    source = models.CharField(max_length=32, choices=SourceType.choices)

    #Natural key from the source (NHTSA Report ID / OpenAlex id); stable across reruns
    source_id = models.TextField()

    #Trimmed json_blob as produced by the pipeline
    raw_data = models.JSONField()

    #--- Core content (extracted later; optional) ---

    # Short descriptive title of the incident
    title = models.TextField(blank=True, default="")

    #Summary of the incident report
    description = models.TextField(blank=True, default="")

    #Where the source report can be found
    url = models.TextField(blank=True, default="")

    #--- Timestamps ---

    #When incident occurred (OpenAlex has none; NHTSA is month-granularity)
    time_occurred = models.DateTimeField(null=True)

    #When incident was reported by source
    time_reported = models.DateTimeField(null=True)

    #When incident was ingested by pipeline (required)
    time_ingested = models.DateTimeField()

    #When LLM passed judgment on incident
    time_judged = models.DateTimeField(null=True)

    #When human approved or rejected event for display
    time_reviewed = models.DateTimeField(null=True)

    #--- Animal details (extracted later; optional) ---

    AnimalType = models.TextChoices("AnimalType", "farmed wild companion other unknown")
    animal_type = models.CharField(
        max_length=9,
        choices=AnimalType.choices,
        default=AnimalType.unknown,
    )

    animal_species = models.TextField(blank=True, default="")

    #Number of animals impacted; -1 for unknown
    animal_count = models.IntegerField(default=-1)

    #Will standardize on this eventually
    harm_type = models.TextField(blank=True, default="")

    #Longer description of the harm
    harm_description = models.TextField(blank=True, default="")

    #Will standardize on this eventually
    HarmSeverity = models.TextChoices("HarmSeverity", "low medium high")

    #Ex. ChatGPT, Waymo, etc.
    ai_system = models.TextField(blank=True, default="")

    #Ex. Anthropic, OpenAI, etc.
    ai_system_manufacturer = models.TextField(blank=True, default="")

    city = models.TextField(blank=True, default="")

    country = models.TextField(blank=True, default="")

    #--- Judgment / review ---

    class StatusType(models.TextChoices):
        new = "new", "New"
        pending = "pending", "Pending"
        approved = "approved", "Approved"
        rejected = "rejected", "Rejected"

    status = models.CharField(
        max_length=8,
        choices=StatusType.choices,
        default=StatusType.new,
    )

    #Level of confidence LLM expreses in judgment
    confidence = models.IntegerField(default=0)

    #LLM rationale for the keep/skip judgment
    llm_reasoning = models.TextField(blank=True, default="")

    #Model that produced the judgment (from pipeline llm_meta)
    llm_model = models.TextField(blank=True, default="")

    #figure out how to map a reviwer to the auth_user table and then add reviewer here
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["source", "source_id"],
                name="uniq_source_sourceid",
            ),
        ]

    def __str__(self):
        return self.title
