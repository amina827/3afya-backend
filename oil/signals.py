from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import OilReference


@receiver(post_delete, sender=OilReference)
def oil_reference_deleted(sender, instance, **kwargs):
    from oil.services.image_processing import invalidate_reference_cache
    invalidate_reference_cache()
