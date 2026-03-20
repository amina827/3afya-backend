from rest_framework import serializers
from oil.models import (
    BottleSpecification,
    ScanSession,
    ScanImage,
    CupTarget,
    AccuracyFeedback,
)


class ScanSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScanSession
        fields = ["id", "bottle", "status", "device_metadata", "created_at"]
        read_only_fields = ["id", "status", "created_at"]


class ImageUploadSerializer(serializers.Serializer):
    bottle_id = serializers.SlugField()
    scan_id = serializers.UUIDField(required=False)
    image = serializers.ImageField()
    device_metadata = serializers.JSONField(required=False)
    lighting_conditions = serializers.CharField(required=False, allow_blank=True)
    camera_type = serializers.CharField(required=False, allow_blank=True)
    environment = serializers.CharField(required=False, allow_blank=True)


class ScanResultSerializer(serializers.ModelSerializer):
    processed_image_url = serializers.SerializerMethodField()
    consumed_cups_range = serializers.SerializerMethodField()
    remaining_liters_estimate = serializers.SerializerMethodField()

    class Meta:
        model = ScanImage
        fields = [
            "scan",
            "oil_ratio",
            "remaining_volume_liters",
            "consumed_volume_liters",
            "remaining_cups",
            "consumed_cups",
            "consumed_cups_range",
            "remaining_liters_estimate",
            "processed_image_url",
            "confidence_score",
            "processing_time_ms",
        ]

    def get_processed_image_url(self, obj):
        request = self.context.get("request")
        if not obj.processed_image:
            return None
        if request:
            return request.build_absolute_uri(obj.processed_image.url)
        return obj.processed_image.url

    def get_consumed_cups_range(self, obj):
        if obj.consumed_cups is None:
            return None
        confidence = obj.confidence_score or 0.0
        margin = max(0.2, (1 - confidence) * 1.0)
        low = max(0.0, obj.consumed_cups - margin)
        high = obj.consumed_cups + margin
        return [round(low, 2), round(high, 2)]

    def get_remaining_liters_estimate(self, obj):
        if obj.remaining_volume_liters is None:
            return None
        return round(obj.remaining_volume_liters, 3)


class TargetLevelSerializer(serializers.Serializer):
    scan_id = serializers.UUIDField()
    target_cups = serializers.FloatField()


class TargetResponseSerializer(serializers.ModelSerializer):
    target_image_url = serializers.SerializerMethodField()

    class Meta:
        model = CupTarget
        fields = ["scan", "target_cups", "target_image_url"]

    def get_target_image_url(self, obj):
        request = self.context.get("request")
        if not obj.target_image:
            return None
        if request:
            return request.build_absolute_uri(obj.target_image.url)
        return obj.target_image.url


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccuracyFeedback
        fields = ["scan", "actual_cups", "notes", "created_at"]
        read_only_fields = ["created_at"]
