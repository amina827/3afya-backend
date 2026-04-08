from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include, re_path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


def healthz(_request):
    return JsonResponse({"status": "ok"})

schema_view = get_schema_view(
    openapi.Info(
        title="3afya API",
        default_version="v1",
        description="API documentation for the 3afya backend",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    # path("healthz/", healthz, name="healthz"),
    path("admin/", admin.site.urls),
    path("", include("oil.urls")),
    path("api/", include("oil.api.urls")),
    re_path(r"^swagger(?P<format>\.json|\.yaml)$", schema_view.without_ui(cache_timeout=0), name="schema-json"),
    path("swagger/", schema_view.with_ui("swagger", cache_timeout=0), name="schema-swagger-ui"),
    path("redoc/", schema_view.with_ui("redoc", cache_timeout=0), name="schema-redoc"),
]

# Always serve /media/ — Django's default static() helper only registers
# this in DEBUG mode. WhiteNoise handles /static/ but not /media/, so we
# wire up Django's serve view directly. Fine for this scale; for higher
# volume move uploads to object storage (S3/R2/Railway Volume).
from django.views.static import serve as media_serve  # noqa: E402
urlpatterns += [
    re_path(r"^media/(?P<path>.*)$", media_serve, {"document_root": settings.MEDIA_ROOT}),
]
