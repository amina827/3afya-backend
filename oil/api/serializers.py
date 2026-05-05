from rest_framework import serializers
from oil.models import (
    BottleSpecification,
    ScanSession,
    ScanImage,
    CupTarget,
    AccuracyFeedback,
    TrainingImage,
    Label,
    QRCode,
    VerificationLog,
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
    processed_image_url        = serializers.SerializerMethodField()
    original_image_url         = serializers.SerializerMethodField()
    consumed_cups_range        = serializers.SerializerMethodField()
    remaining_liters_estimate  = serializers.SerializerMethodField()
    oil_percentage             = serializers.SerializerMethodField()
    oil_line_position_from_top = serializers.SerializerMethodField()
    # Fixed: rounded via method to avoid floating-point noise
    consumed_volume_liters     = serializers.SerializerMethodField()
    consumed_cups              = serializers.SerializerMethodField()

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
            "oil_percentage",
            "oil_line_position_from_top",
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

    def get_consumed_volume_liters(self, obj):
        if obj.consumed_volume_liters is None:
            return None
        return round(float(obj.consumed_volume_liters), 3)

    def get_consumed_cups(self, obj):
        if obj.consumed_cups is None:
            return None
        return round(float(obj.consumed_cups), 2)

    def get_consumed_cups_range(self, obj):
        if obj.consumed_cups is None:
            return None
        confidence = obj.confidence_score or 0.0
        margin = max(0.2, (1 - confidence) * 1.0)
        low  = max(0.0, round(float(obj.consumed_cups) - margin, 2))
        high = round(float(obj.consumed_cups) + margin, 2)
        return [low, high]

    def get_remaining_liters_estimate(self, obj):
        if obj.remaining_volume_liters is None:
            return None
        return round(float(obj.remaining_volume_liters), 3)

    def get_oil_percentage(self, obj):
        if obj.oil_ratio is None:
            return None
        return round(float(obj.oil_ratio) * 100.0, 2)

    def get_oil_line_position_from_top(self, obj):
        """Normalized line position in bottle space (0.0=top, 1.0=bottom)."""
        if obj.oil_ratio is None:
            return None
        return round(1.0 - float(obj.oil_ratio), 4)


class TargetLevelSerializer(serializers.Serializer):
    scan_id = serializers.UUIDField()
    target_cups = serializers.FloatField(
        required=False,
        help_text="Target cups (e.g. 0.5, 1, 1.5). Either this OR target_volume_ml is required.",
    )
    target_volume_ml = serializers.FloatField(
        required=False, min_value=0,
        help_text="Target volume in ml. Either this OR target_cups is required.",
    )

    def validate(self, attrs):
        cups = attrs.get("target_cups")
        ml   = attrs.get("target_volume_ml")
        if cups is None and ml is None:
            raise serializers.ValidationError(
                "Provide either target_cups or target_volume_ml."
            )
        if cups is not None and ml is not None:
            raise serializers.ValidationError(
                "Provide only one of target_cups or target_volume_ml, not both."
            )
        return attrs


class TargetResponseSerializer(serializers.ModelSerializer):
    target_image_url            = serializers.SerializerMethodField()
    target_volume_ml            = serializers.SerializerMethodField()
    level_position_percent      = serializers.SerializerMethodField()
    cup_ml                      = serializers.SerializerMethodField()
    target_volume_liters        = serializers.SerializerMethodField()
    target_ratio                = serializers.SerializerMethodField()
    target_percentage           = serializers.SerializerMethodField()
    target_line_position_from_top = serializers.SerializerMethodField()

    class Meta:
        model = CupTarget
        fields = [
            "scan",
            "target_cups",
            "target_volume_ml",
            "level_position_percent",
            "cup_ml",
            "target_volume_liters",
            "target_ratio",
            "target_percentage",
            "target_line_position_from_top",
            "target_image_url",
        ]

    def _cup_ml(self, obj):
        return float(obj.scan.bottle.cup_conversion_ratio) * 1000.0

    def get_target_image_url(self, obj):
        request = self.context.get("request")
        if not obj.target_image:
            return None
        if request:
            return request.build_absolute_uri(obj.target_image.url)
        return obj.target_image.url

    def get_cup_ml(self, obj):
        return round(self._cup_ml(obj), 2)

    def get_target_volume_ml(self, obj):
        return round(obj.target_cups * self._cup_ml(obj), 2)

    def get_level_position_percent(self, obj):
        """Oil amount percentage where 100 means full bottle."""
        ratio = self.get_target_ratio(obj)
        return round(ratio * 100.0, 2)

    def get_target_volume_liters(self, obj):
        bottle = obj.scan.bottle
        liters = float(obj.target_cups) * float(bottle.cup_conversion_ratio)
        return round(max(0.0, liters), 3)

    def get_target_ratio(self, obj):
        bottle = obj.scan.bottle
        total  = float(bottle.total_volume_liters)
        if total <= 0:
            return 0.0
        liters = float(obj.target_cups) * float(bottle.cup_conversion_ratio)
        ratio  = max(0.0, min(1.0, liters / total))
        return round(ratio, 4)

    def get_target_percentage(self, obj):
        ratio = self.get_target_ratio(obj)
        return round(ratio * 100.0, 2)

    def get_target_line_position_from_top(self, obj):
        """Normalized line position in bottle space (0.0=top, 1.0=bottom)."""
        ratio = self.get_target_ratio(obj)
        return round(1.0 - ratio, 4)


class SliderStepSerializer(serializers.Serializer):
    index            = serializers.IntegerField()
    cups             = serializers.FloatField()
    volume_ml        = serializers.FloatField()
    position_percent = serializers.FloatField()
    label            = serializers.CharField()


class SliderConfigSerializer(serializers.Serializer):
    bottle_id      = serializers.CharField()
    bottle_name    = serializers.CharField()
    total_volume_ml = serializers.FloatField()
    cup_ml         = serializers.FloatField()
    step_ml        = serializers.FloatField()
    max_cups       = serializers.FloatField()
    steps          = SliderStepSerializer(many=True)


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccuracyFeedback
        fields = ["scan", "actual_cups", "notes", "created_at"]
        read_only_fields = ["created_at"]


class TrainingImageUploadSerializer(serializers.Serializer):
    bottle_id             = serializers.SlugField()
    image                 = serializers.ImageField()
    actual_oil_percentage = serializers.FloatField(min_value=0, max_value=100)
    lighting              = serializers.ChoiceField(
        choices=["daylight", "fluorescent", "dim", "direct_sun", "mixed"],
        default="daylight",
    )
    environment = serializers.ChoiceField(
        choices=["kitchen", "store", "outdoor", "office", "other"],
        default="kitchen",
    )
    camera_info = serializers.CharField(required=False, allow_blank=True, default="")
    notes       = serializers.CharField(required=False, allow_blank=True, default="")
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
    total_images    = serializers.IntegerField()
    verified_images = serializers.IntegerField()
    by_lighting     = serializers.DictField()
    by_environment  = serializers.DictField()
    by_bottle       = serializers.ListField()


# =====================================================================
# QR Code & Label Verification
# =====================================================================

class LabelSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = Label
        fields = [
            "id", "name", "name_en", "product_type", "volume_ml",
            "barcode", "image_url", "description", "is_active",
        ]

    def get_image_url(self, obj):
        if not obj.image:
            return None
        request = self.context.get("request")
        if request:
            return request.build_absolute_uri(obj.image.url)
        return obj.image.url


class QRCodeSerializer(serializers.ModelSerializer):
    label = LabelSerializer(read_only=True)

    class Meta:
        model = QRCode
        fields = [
            "id", "code", "label", "batch_number",
            "production_date", "expiry_date", "factory_code", "is_active",
        ]


class VerifyQRSerializer(serializers.Serializer):
    """Input for QR vs Label verification."""
    qr_data             = serializers.CharField(help_text="The scanned QR code content")
    scanned_label_name  = serializers.CharField(
        required=False, allow_blank=True,
        help_text="What label the user sees on the bottle (optional)",
    )
    bottle_image = serializers.ImageField(
        required=False,
        help_text="Photo of the bottle for verification (optional)",
    )


class VerifyResultSerializer(serializers.Serializer):
    """Output of QR vs Label verification."""
    result         = serializers.ChoiceField(choices=VerificationLog.RESULT_CHOICES)
    message        = serializers.CharField()
    qr_code        = QRCodeSerializer(allow_null=True)
    expected_label = LabelSerializer(allow_null=True)