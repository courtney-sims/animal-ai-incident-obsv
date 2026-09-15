from django.urls import path

from . import views

urlpatterns = [
    path("", views.incident_list, name="incident_list"),
    path("incident/<int:pk>/", views.incident_detail, name="incident_detail"),
    path("about/", views.about, name="about"),
    path("submit/", views.incident_submit, name="incident_submit"),
    path("subscribe/", views.subscribe, name="subscribe"),
    path("contact/", views.contact, name="contact"),
]
