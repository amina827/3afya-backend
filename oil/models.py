import uuid
from django.db import models


class BottleSpecification(models.Model):
    SHAPE_CYLINDER = "cylinder"
    SHAPE_RECT = "rectangular"
    SHAPE_CURVED = "curved"
    SHAPE_CHOICES = [
        (SHAPE_CYLINDER, "Cylinder"),
        (SHAPE_RECT, "Rectangular"),
        (SHAPE_CURVED, "Curved"),
    ]

    bottle_name = models.CharField(max_length=120)
    bottle_id = models.SlugField(max_length=64, unique=True)
    total_volume_liters = models.DecimalField(max_digits=6, decimal_places=3)
    bottle_height_reference = models.PositiveIntegerField(
        help_text="Reference bottle height in pixels for calibration."
    )
    height_to_volume_ratio = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        help_text="Liters per pixel of height based on calibration.",
    )
    cup_conversion_ratio = models.DecimalField(
        max_digits=6,
        decimal_places=3,
        default=0.250,
        help_text="Liters per cup. Default: 0.250",
    )
    shape_type = models.CharField(max_length=32, choices=SHAPE_CHOICES, default=SHAPE_CYLINDER)
    calibration_points = models.JSONField(
        default=list,
        help_text="List of calibration points. Example: [{pixel: 1200, liters: 5.0}].",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.bottle_name} ({self.bottle_id})"


class BottleReferenceImage(models.Model):
    bottle = models.ForeignKey(BottleSpecification, on_delete=models.CASCADE, related_name="reference_images")
    image = models.ImageField(upload_to="reference_bottles/")
    note = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ScanSession(models.Model):
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_DONE, "Done"),
        (STATUS_FAILED, "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bottle = models.ForeignKey(BottleSpecification, on_delete=models.PROTECT, related_name="scan_sessions")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    device_metadata = models.JSONField(default=dict, blank=True)
    lighting_conditions = models.CharField(max_length=120, blank=True)
    camera_type = models.CharField(max_length=120, blank=True)
    environment = models.CharField(max_length=120, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class ScanImage(models.Model):
    scan = models.OneToOneField(ScanSession, on_delete=models.CASCADE, related_name="image")
    original_image = models.ImageField(upload_to="scans/original/")
    processed_image = models.ImageField(upload_to="scans/processed/", blank=True)
    oil_height_pixels = models.FloatField(null=True, blank=True)
    bottle_height_pixels = models.FloatField(null=True, blank=True)
    oil_ratio = models.FloatField(null=True, blank=True)
    remaining_volume_liters = models.FloatField(null=True, blank=True)
    consumed_volume_liters = models.FloatField(null=True, blank=True)
    remaining_cups = models.FloatField(null=True, blank=True)
    consumed_cups = models.FloatField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    processing_time_ms = models.PositiveIntegerField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class CupTarget(models.Model):
    scan = models.ForeignKey(ScanSession, on_delete=models.CASCADE, related_name="cup_targets")
    target_cups = models.FloatField()
    target_image = models.ImageField(upload_to="scans/targets/", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class AccuracyFeedback(models.Model):
    scan = models.ForeignKey(ScanSession, on_delete=models.CASCADE, related_name="feedback")
    actual_cups = models.FloatField()
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
