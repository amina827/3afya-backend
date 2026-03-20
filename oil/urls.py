from django.urls import path
from .views import scan_entry

urlpatterns = [
    path("scan/<slug:bottle_id>/", scan_entry, name="scan-entry"),
]
