from django.contrib import admin
from .models import (
    BottleSpecification,
    BottleReferenceImage,
    ScanSession,
    ScanImage,
    CupTarget,
    AccuracyFeedback,
    TrainingImage,
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
