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
    bottle_bbox = models.JSONField(
        null=True,
        blank=True,
        help_text="Bottle bounding box: {x, y, w, h, image_w, image_h}",
    )
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


# =====================================================================
# QR Code & Label verification
# =====================================================================

class Label(models.Model):
    """A product label representing a specific Afia oil product."""
    PRODUCT_TYPES = [
        ("corn", "Corn Oil / زيت ذرة"),
        ("sunflower", "Sunflower Oil / زيت عباد الشمس"),
        ("olive", "Olive Oil / زيت زيتون"),
        ("blend", "Blended Oil / زيت مخلوط"),
        ("canola", "Canola Oil / زيت كانولا"),
    ]

    name = models.CharField(max_length=200, help_text="Product name (e.g. 'عافية زيت ذرة نقي 1.5 لتر')")
    name_en = models.CharField(max_length=200, blank=True, help_text="English product name")
    product_type = models.CharField(max_length=32, choices=PRODUCT_TYPES, default="corn")
    volume_ml = models.PositiveIntegerField(help_text="Volume in ml (e.g. 1500)")
    bottle = models.ForeignKey(
        BottleSpecification, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="labels",
        help_text="Linked bottle specification for oil level detection",
    )
    barcode = models.CharField(max_length=64, blank=True, help_text="EAN/UPC barcode number")
    image = models.ImageField(upload_to="labels/", blank=True, help_text="Reference photo of the label")
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.volume_ml}ml)"


class QRCode(models.Model):
    """A QR code printed on an Afia bottle, linked to a specific label/product."""
    code = models.CharField(
        max_length=500, unique=True,
        help_text="The full QR code content (URL or text scanned from the bottle)",
    )
    label = models.ForeignKey(
        Label, on_delete=models.CASCADE, related_name="qr_codes",
        help_text="The label/product this QR code belongs to",
    )
    batch_number = models.CharField(max_length=64, blank=True, help_text="Production batch number")
    production_date = models.DateField(null=True, blank=True)
    expiry_date = models.DateField(null=True, blank=True)
    factory_code = models.CharField(max_length=64, blank=True, help_text="Factory/plant code")
    is_active = models.BooleanField(default=True, help_text="Whether this QR code is still valid")
    scan_count = models.PositiveIntegerField(default=0, help_text="Number of times this QR was scanned")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "QR Code"
        verbose_name_plural = "QR Codes"
        ordering = ["-created_at"]

    def __str__(self):
        short = self.code[:50] + "..." if len(self.code) > 50 else self.code
        return f"QR:{short} → {self.label.name}"


class VerificationLog(models.Model):
    """Log of QR vs Label verification attempts."""
    RESULT_MATCH = "match"
    RESULT_MISMATCH = "mismatch"
    RESULT_QR_NOT_FOUND = "qr_not_found"
    RESULT_CHOICES = [
        (RESULT_MATCH, "Match ✓"),
        (RESULT_MISMATCH, "Mismatch ✗"),
        (RESULT_QR_NOT_FOUND, "QR Not Found"),
    ]

    qr_data = models.CharField(max_length=500, help_text="QR code content that was scanned")
    qr_code = models.ForeignKey(
        QRCode, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="verifications",
    )
    expected_label = models.ForeignKey(
        Label, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="verifications",
        help_text="The label the QR code should match",
    )
    scanned_label_name = models.CharField(
        max_length=200, blank=True,
        help_text="What label the user reported/detected on the bottle",
    )
    result = models.CharField(max_length=16, choices=RESULT_CHOICES)
    bottle_image = models.ImageField(upload_to="verifications/", blank=True)
    device_info = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.result} @ {self.created_at:%Y-%m-%d %H:%M}"


class TrainingImage(models.Model):
    """Real-world images collected to improve the local programmatic engine."""
    LIGHTING_CHOICES = [
        ("daylight", "Natural Daylight"),
        ("fluorescent", "Fluorescent/Indoor"),
        ("dim", "Dim/Low Light"),
        ("direct_sun", "Direct Sunlight"),
        ("mixed", "Mixed Lighting"),
    ]
    ENV_CHOICES = [
        ("kitchen", "Kitchen"),
        ("store", "Store/Shelf"),
        ("outdoor", "Outdoor"),
        ("office", "Office"),
        ("other", "Other"),
    ]

    bottle = models.ForeignKey(BottleSpecification, on_delete=models.CASCADE, related_name="training_images")
    image = models.ImageField(upload_to="training/")
    actual_oil_percentage = models.FloatField(help_text="Actual oil level 0-100 as reported by the tester")
    lighting = models.CharField(max_length=32, choices=LIGHTING_CHOICES, default="daylight")
    environment = models.CharField(max_length=32, choices=ENV_CHOICES, default="kitchen")
    camera_info = models.CharField(max_length=255, blank=True)
    notes = models.TextField(blank=True)
    uploaded_by = models.CharField(max_length=120, blank=True, help_text="Tester name or ID")
    is_verified = models.BooleanField(default=False, help_text="Verified by admin for use in training")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Training {self.bottle.bottle_id} {self.actual_oil_percentage}% ({self.lighting}/{self.environment})"
