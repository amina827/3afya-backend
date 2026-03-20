from django.contrib import admin
from .models import (
    BottleSpecification,
    BottleReferenceImage,
    ScanSession,
    ScanImage,
    CupTarget,
    AccuracyFeedback,
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
