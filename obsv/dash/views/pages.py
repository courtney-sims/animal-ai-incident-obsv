"""Static / public content pages."""

from django.conf import settings
from django.shortcuts import render


def about(request):
    return render(request, "dash/about.html")


def contact(request):
    return render(
        request,
        "dash/contact.html",
        {"contact_email": settings.CONTACT_EMAIL},
    )
