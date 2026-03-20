from celery import shared_task
from django.db import transaction
from .models import ScanSession, ScanImage
from .services.image_processing import process_bottle_image, ProcessingError


@shared_task(bind=True)
def process_scan_image(self, scan_id: str):
    scan = ScanSession.objects.select_related("bottle").get(id=scan_id)
    scan.status = ScanSession.STATUS_PROCESSING
    scan.save(update_fields=["status", "updated_at"])

    try:
        scan_image = ScanImage.objects.get(scan=scan)
        result = process_bottle_image(scan_image.original_image.path, scan.bottle)
        with transaction.atomic():
            scan_image.processed_image.name = result["processed_path"]
            scan_image.oil_height_pixels = result["oil_height_pixels"]
            scan_image.bottle_height_pixels = result["bottle_height_pixels"]
            scan_image.oil_ratio = result["oil_ratio"]
            scan_image.remaining_volume_liters = result["remaining_volume_liters"]
            scan_image.consumed_volume_liters = result["consumed_volume_liters"]
            scan_image.remaining_cups = result["remaining_cups"]
            scan_image.consumed_cups = result["consumed_cups"]
            scan_image.confidence_score = result["confidence_score"]
            scan_image.processing_time_ms = result["processing_time_ms"]
            scan_image.save()

            scan.status = ScanSession.STATUS_DONE
            scan.save(update_fields=["status", "updated_at"])
    except (ProcessingError, FileNotFoundError, ScanImage.DoesNotExist):
        scan.status = ScanSession.STATUS_FAILED
        scan.save(update_fields=["status", "updated_at"])
        raise
