from django.contrib import admin
from .models import (
    BottleSpecification,
    BottleReferenceImage,
    ScanSession,
    ScanImage,
    CupTarget,
    AccuracyFeedback,
    TrainingImage,
    Label,
    QRCode,
    VerificationLog,
)


class BottleReferenceInline(admin.TabularInline):
    model = BottleReferenceImage
    extra = 1


@admin.register(BottleSpecification)
class BottleSpecificationAdmin(admin.ModelAdmin):
    list_display = (
        "bottle_name",
        "bottle_id",
        "total_volume_liters",
        "bottle_height_reference",
        "shape_type",
    )
    search_fields = ("bottle_name", "bottle_id")
    inlines = [BottleReferenceInline]


@admin.register(ScanSession)
class ScanSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "bottle", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("id", "bottle__bottle_id")


@admin.register(ScanImage)
class ScanImageAdmin(admin.ModelAdmin):
    list_display = (
        "scan",
        "oil_ratio",
        "remaining_volume_liters",
        "consumed_volume_liters",
        "confidence_score",
        "processing_time_ms",
    )


@admin.register(CupTarget)
class CupTargetAdmin(admin.ModelAdmin):
    list_display = ("scan", "target_cups", "created_at")


@admin.register(AccuracyFeedback)
class AccuracyFeedbackAdmin(admin.ModelAdmin):
    list_display = ("scan", "actual_cups", "created_at")


# =====================================================================
# QR Code & Label Admin
# =====================================================================

class QRCodeInline(admin.TabularInline):
    model = QRCode
    extra = 1
    fields = ("code", "batch_number", "production_date", "expiry_date", "factory_code", "is_active", "scan_count")
    readonly_fields = ("scan_count",)


@admin.register(Label)
class LabelAdmin(admin.ModelAdmin):
    list_display = ("name", "product_type", "volume_ml", "bottle", "barcode", "is_active", "qr_count")
    list_filter = ("product_type", "is_active", "volume_ml")
    search_fields = ("name", "name_en", "barcode")
    list_editable = ("is_active",)
    inlines = [QRCodeInline]

    @admin.display(description="QR Codes")
    def qr_count(self, obj):
        return obj.qr_codes.count()


@admin.register(QRCode)
class QRCodeAdmin(admin.ModelAdmin):
    list_display = ("short_code", "label", "batch_number", "production_date", "expiry_date", "is_active", "scan_count")
    list_filter = ("is_active", "label__product_type", "production_date")
    search_fields = ("code", "batch_number", "factory_code", "label__name")
    list_editable = ("is_active",)
    readonly_fields = ("scan_count",)
    raw_id_fields = ("label",)

    @admin.display(description="QR Code")
    def short_code(self, obj):
        return obj.code[:60] + "..." if len(obj.code) > 60 else obj.code


@admin.register(VerificationLog)
class VerificationLogAdmin(admin.ModelAdmin):
    list_display = ("result", "qr_data_short", "expected_label", "scanned_label_name", "created_at")
    list_filter = ("result", "created_at")
    search_fields = ("qr_data", "scanned_label_name")
    readonly_fields = ("qr_data", "qr_code", "expected_label", "result", "bottle_image", "device_info", "created_at")
    date_hierarchy = "created_at"

    @admin.display(description="QR Data")
    def qr_data_short(self, obj):
        return obj.qr_data[:40] + "..." if len(obj.qr_data) > 40 else obj.qr_data


@admin.register(TrainingImage)
class TrainingImageAdmin(admin.ModelAdmin):
    list_display = ("bottle", "actual_oil_percentage", "lighting", "environment", "uploaded_by", "is_verified", "created_at")
    list_filter = ("lighting", "environment", "is_verified", "bottle")
    search_fields = ("uploaded_by", "notes")
    list_editable = ("is_verified",)
    actions = ["mark_verified"]

    @admin.action(description="Mark selected images as verified")
    def mark_verified(self, request, queryset):
        queryset.update(is_verified=True)
