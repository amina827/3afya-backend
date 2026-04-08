from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from PIL import Image
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from django.db.models import Count

from oil.models import BottleSpecification, ScanSession, ScanImage, CupTarget, TrainingImage
from oil.api.serializers import (
    ImageUploadSerializer,
    ScanResultSerializer,
    TargetLevelSerializer,
    TargetResponseSerializer,
    FeedbackSerializer,
    TrainingImageUploadSerializer,
    TrainingImageResponseSerializer,
    TrainingStatsSerializer,
)
from oil.services.image_processing import render_target_overlay, ProcessingError
from oil.tasks import process_scan_image


class ImageUploadView(APIView):
    @swagger_auto_schema(
        operation_id="upload_image",
        operation_description="Upload a bottle image to start a scan session. If scan_id is not provided, a new session is created.",
        request_body=ImageUploadSerializer,
        responses={
            201: openapi.Response(
                description="Image uploaded and scan process initiated",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'scan_id': openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_UUID, description="The UUID of the created or existing scan session."),
                        'status': openapi.Schema(type=openapi.TYPE_STRING, description="The current status of the scan session."),
                    }
                )
            ),
            400: "Bad Request - Invalid input, image size, or image format.",
        }
    )
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        bottle = get_object_or_404(BottleSpecification, bottle_id=serializer.validated_data["bottle_id"])
        image = serializer.validated_data["image"]

        # Validate file size
        max_size = getattr(settings, "FILE_UPLOAD_MAX_MEMORY_SIZE", 10 * 1024 * 1024)
        if image.size > max_size:
            return Response({"error": "Image exceeds size limit"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate image content
        try:
            Image.open(image).verify()
        except Exception:
            return Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)

        scan_id = serializer.validated_data.get("scan_id")
        if scan_id:
            scan = get_object_or_404(ScanSession, id=scan_id)
            if scan.bottle_id != bottle.id:
                return Response({"error": "Bottle mismatch for scan"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            scan = ScanSession.objects.create(
                bottle=bottle,
                device_metadata=serializer.validated_data.get("device_metadata", {}),
                lighting_conditions=serializer.validated_data.get("lighting_conditions", ""),
                camera_type=serializer.validated_data.get("camera_type", ""),
                environment=serializer.validated_data.get("environment", ""),
            )

        scan_image = ScanImage.objects.create(scan=scan, original_image=image)

        process_scan_image.delay(str(scan.id))
        scan.refresh_from_db()

        return Response({"scan_id": str(scan.id), "status": scan.status}, status=status.HTTP_201_CREATED)


class ScanResultView(APIView):
    @swagger_auto_schema(
        operation_id="get_scan_result",
        operation_description="Retrieve the status and result of a scan session. The 'result' field is only present if an image has been uploaded for the scan.",
        responses={
            200: openapi.Response(
                description="Scan status and optional result.",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'status': openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description="Current status of the scan (e.g., pending, processing, done, failed)."
                        ),
                        'result': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                "scan": openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_UUID),
                                "oil_ratio": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "remaining_volume_liters": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "consumed_volume_liters": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "remaining_cups": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "consumed_cups": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "consumed_cups_range": openapi.Schema(
                                    type=openapi.TYPE_ARRAY,
                                    items=openapi.Schema(type=openapi.TYPE_NUMBER),
                                ),
                                "remaining_liters_estimate": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "processed_image_url": openapi.Schema(type=openapi.TYPE_STRING),
                                "confidence_score": openapi.Schema(type=openapi.TYPE_NUMBER),
                                "processing_time_ms": openapi.Schema(type=openapi.TYPE_INTEGER),
                            },
                        ),
                    },
                    required=['status']
                )
            ),
            404: "Scan session not found.",
        }
    )
    def get(self, request, scan_id):
        scan = get_object_or_404(ScanSession, id=scan_id)
        if not hasattr(scan, "image"):
            return Response({"status": scan.status}, status=status.HTTP_200_OK)
        serializer = ScanResultSerializer(scan.image, context={"request": request})

        response = {
            "status": scan.status,
            "result": serializer.data,
        }
        return Response(response, status=status.HTTP_200_OK)


class TargetLevelView(APIView):
    @swagger_auto_schema(
        operation_id="create_target_level",
        operation_description="Generate an image overlay showing a target consumption level.",
        request_body=TargetLevelSerializer,
        responses={
            201: TargetResponseSerializer,
            400: "Bad Request - Processing error.",
            404: "Scan session or image not found.",
        }
    )
    def post(self, request):
        serializer = TargetLevelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        scan = get_object_or_404(ScanSession, id=serializer.validated_data["scan_id"])
        if not hasattr(scan, "image"):
            return Response({"error": "Scan image not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            target_image_path = render_target_overlay(
                scan.image.original_image.path,
                scan.bottle,
                serializer.validated_data["target_cups"],
            )
        except ProcessingError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        target = CupTarget.objects.create(
            scan=scan,
            target_cups=serializer.validated_data["target_cups"],
            target_image=target_image_path,
        )

        response = TargetResponseSerializer(target, context={"request": request}).data
        return Response(response, status=status.HTTP_201_CREATED)


class FeedbackView(APIView):
    @swagger_auto_schema(
        operation_id="submit_feedback",
        operation_description="Submit accuracy feedback for a completed scan.",
        request_body=FeedbackSerializer,
        responses={
            201: FeedbackSerializer,
            400: "Invalid data provided.",
        }
    )
    def post(self, request):
        serializer = FeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        feedback = serializer.save()
        return Response(FeedbackSerializer(feedback).data, status=status.HTTP_201_CREATED)


class TrainingImageUploadView(APIView):
    @swagger_auto_schema(
        operation_id="upload_training_image",
        operation_description="Upload a real-world bottle image for training the local detection engine. "
                              "Include metadata about lighting, environment, and the actual oil percentage.",
        request_body=TrainingImageUploadSerializer,
        responses={
            201: TrainingImageResponseSerializer,
            400: "Bad Request.",
            404: "Bottle not found.",
        }
    )
    def post(self, request):
        serializer = TrainingImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        bottle = get_object_or_404(BottleSpecification, bottle_id=serializer.validated_data["bottle_id"])
        image = serializer.validated_data["image"]

        # Validate file size
        max_size = getattr(settings, "FILE_UPLOAD_MAX_MEMORY_SIZE", 10 * 1024 * 1024)
        if image.size > max_size:
            return Response({"error": "Image exceeds size limit"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            Image.open(image).verify()
        except Exception:
            return Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)

        training_img = TrainingImage.objects.create(
            bottle=bottle,
            image=image,
            actual_oil_percentage=serializer.validated_data["actual_oil_percentage"],
            lighting=serializer.validated_data.get("lighting", "daylight"),
            environment=serializer.validated_data.get("environment", "kitchen"),
            camera_info=serializer.validated_data.get("camera_info", ""),
            notes=serializer.validated_data.get("notes", ""),
            uploaded_by=serializer.validated_data.get("uploaded_by", ""),
        )

        response = TrainingImageResponseSerializer(training_img, context={"request": request}).data
        return Response(response, status=status.HTTP_201_CREATED)


class TrainingStatsView(APIView):
    @swagger_auto_schema(
        operation_id="training_stats",
        operation_description="Get statistics about collected training images.",
        responses={200: TrainingStatsSerializer}
    )
    def get(self, request):
        qs = TrainingImage.objects.all()
        total = qs.count()
        verified = qs.filter(is_verified=True).count()

        by_lighting = dict(qs.values_list("lighting").annotate(c=Count("id")).values_list("lighting", "c"))
        by_env = dict(qs.values_list("environment").annotate(c=Count("id")).values_list("environment", "c"))
        by_bottle = list(
            qs.values("bottle__bottle_id", "bottle__bottle_name")
            .annotate(count=Count("id"))
            .order_by("-count")
        )

        return Response({
            "total_images": total,
            "verified_images": verified,
            "by_lighting": by_lighting,
            "by_environment": by_env,
            "by_bottle": by_bottle,
        })
