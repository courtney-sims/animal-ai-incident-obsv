"""Newsletter signup + Slack pointer."""

from django.shortcuts import render

from ..forms import SubscribeForm
from ..models import Subscriber


def subscribe(request):
    """Newsletter signup.

    Duplicate and new signups render the identical success message to avoid
    email enumeration; the unique constraint prevents duplicate rows.
    Honeypot-filled POSTs are dropped.
    """
    context: dict[str, object] = {
        "success": False,
    }
    if request.method == "POST":
        form = SubscribeForm(request.POST)
        if form.is_valid():
            if not form.cleaned_data.get("website"):
                Subscriber.objects.get_or_create(email=form.cleaned_data["email"])
            context["form"] = SubscribeForm()
            context["success"] = True
            return render(request, "dash/subscribe.html", context)
    else:
        form = SubscribeForm()
    context["form"] = form
    return render(request, "dash/subscribe.html", context)
