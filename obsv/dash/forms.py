"""Public forms: user incident submission and newsletter subscription.

Both forms carry a honeypot field (``website``) rendered off-screen via CSS.
Real users never fill it; bots that auto-complete every field do. The view
layer drops any submission where it is non-empty.

``IncidentSubmitForm`` is a plain ``forms.Form`` (not a ModelForm) because
it is deliberately a partial set of fields. Some required fields are provenance
fields which can't be user generated and a valid user submission has different
requirements because it's more of a tip than an official report. Moderators are
responsible for translating tips into reports.
"""

from django import forms

from .models import IncidentReport


class HoneypotMixin(forms.Form):
    """Adds an off-screen ``website`` field used only to catch bots."""

    website = forms.CharField(
        required=False,
        label="Website",
        widget=forms.TextInput(
            attrs={
                "class": "hp-field",
                "autocomplete": "off",
                "tabindex": "-1",
            }
        ),
    )


class IncidentSubmitForm(HoneypotMixin, forms.Form):
    url = forms.URLField(
        required=False,
        label="Link to the incident",
        widget=forms.URLInput(attrs={"placeholder": "https://..."}),
    )
    description = forms.CharField(
        required=False,
        label="What happened?",
        help_text="If you're not linking a source above, please also include where you found this information.",
        widget=forms.Textarea(attrs={"rows": 4}),
    )
    submitter_email = forms.EmailField(
        required=False,
        label="Your email",
        help_text="In case we have follow-up questions. Optional.",
    )

    def clean(self):
        cleaned = super().clean()
        if not cleaned.get("url") and not cleaned.get("description"):
            raise forms.ValidationError(
                "Please provide at least a link to the incident or a "
                "description of it."
            )
        return cleaned


class SubscribeForm(HoneypotMixin, forms.Form):
    email = forms.EmailField(
        label="Email address",
        widget=forms.EmailInput(attrs={"placeholder": "you@example.com"}),
    )
