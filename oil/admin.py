from django.contrib import admin, messages
from django.utils.html import format_html
from .models import (
    BottleSpecification,
    BottleReferenceImage,
    OilReference,
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


@admin.register(OilReference)
class OilReferenceAdmin(admin.ModelAdmin):
    list_display = (
        "level_percentage",
        "bottle",
        "version",
        "is_active",
        "features_status",
        "thumbnail",
        "updated_at",
    )
    list_filter = ("is_active", "bottle", "level_percentage")
    search_fields = ("notes", "bottle__bottle_name")
    list_editable = ("is_active",)
    readonly_fields = (
        "version",
        "golden_amount",
        "normalized_cache_path",
        "extraction_error",
        "features_summary",
        "created_at",
        "updated_at",
    )
    actions = ["rebuild_features", "activate_refs", "deactivate_refs"]
    fieldsets = (
        (None, {
            "fields": ("bottle", "image", "level_percentage", "is_active", "notes"),
        }),
        ("Cached features", {
            "classes": ("collapse",),
            "fields": (
                "version",
                "golden_amount",
                "normalized_cache_path",
                "features_summary",
                "extraction_error",
            ),
        }),
        ("Timestamps", {
            "classes": ("collapse",),
            "fields": ("created_at", "updated_at"),
        }),
    )

    @admin.display(description="Features")
    def features_status(self, obj):
        if obj.extraction_error:
            return format_html('<span style="color:#c00;">ERROR</span>')
        if obj.has_features:
            return format_html('<span style="color:#080;">ready</span>')
        return format_html('<span style="color:#999;">pending</span>')

    @admin.display(description="Preview")
    def thumbnail(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="height:48px;border-radius:3px;" />',
                obj.image.url,
            )
        return "—"

    @admin.display(description="Feature summary")
    def features_summary(self, obj):
        if not obj.has_features:
            return "—"
        bp = obj.brightness_profile or []
        gp = obj.golden_profile or []
        hist = obj.histogram or []
        return format_html(
            "brightness_profile: {} rows<br>"
            "golden_profile: {} rows<br>"
            "histogram: {} bins<br>"
            "golden_amount: {:.4f}",
            len(bp), len(gp), len(hist), obj.golden_amount or 0.0,
        )

    @admin.action(description="Rebuild cached features")
    def rebuild_features(self, request, queryset):
        ok, failed = 0, 0
        for ref in queryset:
            try:
                ref.extract_features()
                if ref.has_features:
                    ok += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                self.message_user(
                    request, f"#{ref.pk}: {e}", level=messages.ERROR,
                )
        self.message_user(
            request,
            f"Rebuilt {ok} reference(s); {failed} failed.",
            level=messages.SUCCESS if failed == 0 else messages.WARNING,
        )

    @admin.action(description="Activate selected references")
    def activate_refs(self, request, queryset):
        updated = queryset.update(is_active=True)
        from oil.services.image_processing import invalidate_reference_cache
        invalidate_reference_cache()
        self.message_user(request, f"Activated {updated} reference(s).")

    @admin.action(description="Deactivate selected references")
    def deactivate_refs(self, request, queryset):
        updated = queryset.update(is_active=False)
        from oil.services.image_processing import invalidate_reference_cache
        invalidate_reference_cache()
        self.message_user(request, f"Deactivated {updated} reference(s).")


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
