from rest_framework import serializers
from oil.models import (
    BottleSpecification,
    ScanSession,
    ScanImage,
    CupTarget,
    AccuracyFeedback,
    TrainingImage,
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
    original_image_url = serializers.SerializerMethodField()
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
            "original_image_url",
            "bottle_bbox",
            "confidence_score",
            "processing_time_ms",
        ]

    def get_original_image_url(self, obj):
        request = self.context.get("request")
        if not obj.original_image:
            return None
        if request:
            return request.build_absolute_uri(obj.original_image.url)
        return obj.original_image.url

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


class TrainingImageUploadSerializer(serializers.Serializer):
    bottle_id = serializers.SlugField()
    image = serializers.ImageField()
    actual_oil_percentage = serializers.FloatField(min_value=0, max_value=100)
    lighting = serializers.ChoiceField(
        choices=["daylight", "fluorescent", "dim", "direct_sun", "mixed"],
        default="daylight",
    )
    environment = serializers.ChoiceField(
        choices=["kitchen", "store", "outdoor", "office", "other"],
        default="kitchen",
    )
    camera_info = serializers.CharField(required=False, allow_blank=True, default="")
    notes = serializers.CharField(required=False, allow_blank=True, default="")
    uploaded_by = serializers.CharField(required=False, allow_blank=True, default="")


class TrainingImageResponseSerializer(serializers.ModelSerializer):
    bottle_id = serializers.CharField(source="bottle.bottle_id")
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = TrainingImage
        fields = [
            "id", "bottle_id", "image_url", "actual_oil_percentage",
            "lighting", "environment", "camera_info", "notes",
            "uploaded_by", "is_verified", "created_at",
        ]

    def get_image_url(self, obj):
        request = self.context.get("request")
        if not obj.image:
            return None
        if request:
            return request.build_absolute_uri(obj.image.url)
        return obj.image.url


class TrainingStatsSerializer(serializers.Serializer):
    total_images = serializers.IntegerField()
    verified_images = serializers.IntegerField()
    by_lighting = serializers.DictField()
    by_environment = serializers.DictField()
    by_bottle = serializers.ListField()
