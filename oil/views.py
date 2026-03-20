from django.shortcuts import get_object_or_404
from django.urls import reverse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from .models import BottleSpecification, ScanSession


@swagger_auto_schema(
    method="post",
    operation_id="create_scan_session",
    operation_description="Initialize a new scan session for a specified bottle.",
    responses={
        201: openapi.Response(
            description="Scan session initialized successfully",
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "scan_id": openapi.Schema(type=openapi.TYPE_STRING, format="uuid", description="The UUID of the created scan session"),
                    "bottle_id": openapi.Schema(type=openapi.TYPE_STRING, description="The ID of the bottle"),
                    "upload_url": openapi.Schema(type=openapi.TYPE_STRING, format="uri", description="URL to upload the bottle image"),
                },
            ),
        ),
        404: openapi.Response(description="Bottle not found"),
    },
)
@api_view(["POST"])
def scan_entry(request, bottle_id):
    bottle = get_object_or_404(BottleSpecification, bottle_id=bottle_id)
    scan = ScanSession.objects.create(bottle=bottle)
    upload_url = request.build_absolute_uri(reverse("upload-bottle-image"))
    return Response(
        {
            "scan_id": str(scan.id),
            "bottle_id": bottle.bottle_id,
            "upload_url": upload_url,
        },
        status=status.HTTP_201_CREATED,
    )
