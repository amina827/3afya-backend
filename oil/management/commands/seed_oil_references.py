"""Seed OilReference rows from the bundled reference_data folder.

Run once after migrating to bootstrap admin-managed references. Safe to
re-run: rows are matched by (bottle, level_percentage) and updated in
place rather than duplicated.
"""
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand

from oil.models import BottleSpecification, OilReference
from oil.services.image_processing import REFERENCE_DIR, REFERENCE_LEVELS


class Command(BaseCommand):
    help = "Seed OilReference rows from the bundled reference_data folder."

    def add_arguments(self, parser):
        parser.add_argument(
            "--bottle-id",
            default="afia-1500",
            help="bottle_id of the BottleSpecification to attach references to.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-extract features even if the reference already exists.",
        )

    def handle(self, *args, **opts):
        bottle = BottleSpecification.objects.filter(bottle_id=opts["bottle_id"]).first()
        if not bottle:
            self.stdout.write(self.style.WARNING(
                f"No BottleSpecification with bottle_id={opts['bottle_id']} — "
                "references will be created without a bottle link."
            ))

        # Deactivate rows whose level isn't part of the current reference set
        # — keeps the classifier from mixing stale calibrations (e.g. the old
        # 8-level set) with a freshly seeded 17-level set.
        current_levels = {float(level) for _, level in REFERENCE_LEVELS}
        deactivated = OilReference.objects.filter(
            bottle=bottle, is_active=True,
        ).exclude(level_percentage__in=current_levels).update(is_active=False)
        if deactivated:
            self.stdout.write(self.style.WARNING(
                f"Deactivated {deactivated} stale reference row(s) "
                "with levels no longer in REFERENCE_LEVELS."
            ))

        created, updated, skipped = 0, 0, 0
        for filename, level in REFERENCE_LEVELS:
            src = Path(REFERENCE_DIR) / filename
            if not src.exists():
                self.stdout.write(self.style.WARNING(f"Missing: {src}"))
                continue

            existing = OilReference.objects.filter(
                bottle=bottle, level_percentage=float(level)
            ).first()

            if existing and not opts["force"]:
                skipped += 1
                self.stdout.write(f"  = {level}%  (already exists, skip)")
                continue

            if existing:
                with src.open("rb") as f:
                    existing.image.save(filename, File(f), save=False)
                existing.is_active = True
                existing.notes = f"Seeded from {filename}"
                existing.save()
                updated += 1
                self.stdout.write(self.style.SUCCESS(f"  ~ {level}%  (updated)"))
            else:
                ref = OilReference(
                    bottle=bottle,
                    level_percentage=float(level),
                    is_active=True,
                    notes=f"Seeded from {filename}",
                )
                with src.open("rb") as f:
                    ref.image.save(filename, File(f), save=False)
                ref.save()
                created += 1
                self.stdout.write(self.style.SUCCESS(f"  + {level}%  (created)"))

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. created={created} updated={updated} skipped={skipped}"
        ))
