from django.urls import path
from oil.api.views import ImageUploadView, ScanResultView, TargetLevelView, FeedbackView

urlpatterns = [
    path("upload-bottle-image/", ImageUploadView.as_view(), name="upload-bottle-image"),
    path("result/<uuid:scan_id>/", ScanResultView.as_view(), name="scan-result"),
    path("target-level/", TargetLevelView.as_view(), name="target-level"),
    path("feedback/", FeedbackView.as_view(), name="feedback"),
]
