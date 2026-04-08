from django.db import migrations


BOTTLES = [
    {
        "bottle_id": "test-oil-bottle",
        "bottle_name": "Test Oil Bottle",
        "total_volume_liters": "5.000",
        "bottle_height_reference": 500,
        "height_to_volume_ratio": "0.010000",
        "cup_conversion_ratio": "0.250",
        "shape_type": "cylinder",
        "calibration_points": [
            {"pixel": 0, "liters": 0},
            {"pixel": 500, "liters": 5.0},
        ],
    },
    {
        "bottle_id": "afia-5l",
        "bottle_name": "Afia Gold 5L",
        "total_volume_liters": "5.000",
        "bottle_height_reference": 1200,
        "height_to_volume_ratio": "0.004167",
        "cup_conversion_ratio": "0.250",
        "shape_type": "cylinder",
        "calibration_points": [],
    },
    {
        "bottle_id": "afia-2l",
        "bottle_name": "Afia Classic 2L",
        "total_volume_liters": "2.000",
        "bottle_height_reference": 800,
        "height_to_volume_ratio": "0.002500",
        "cup_conversion_ratio": "0.250",
        "shape_type": "cylinder",
        "calibration_points": [],
    },
    {
        "bottle_id": "afia-1500",
        "bottle_name": "Afia 1.5L",
        "total_volume_liters": "1.500",
        "bottle_height_reference": 800,
        "height_to_volume_ratio": "0.001875",
        "cup_conversion_ratio": "0.200",
        "shape_type": "cylinder",
        "calibration_points": [],
    },
]


def seed_bottles(apps, schema_editor):
    BottleSpecification = apps.get_model("oil", "BottleSpecification")
    for bottle in BOTTLES:
        BottleSpecification.objects.update_or_create(
            bottle_id=bottle["bottle_id"],
            defaults={
                "bottle_name": bottle["bottle_name"],
                "total_volume_liters": bottle["total_volume_liters"],
                "bottle_height_reference": bottle["bottle_height_reference"],
                "height_to_volume_ratio": bottle["height_to_volume_ratio"],
                "cup_conversion_ratio": bottle["cup_conversion_ratio"],
                "shape_type": bottle["shape_type"],
                "calibration_points": bottle["calibration_points"],
            },
        )


def unseed_bottles(apps, schema_editor):
    BottleSpecification = apps.get_model("oil", "BottleSpecification")
    BottleSpecification.objects.filter(
        bottle_id__in=[b["bottle_id"] for b in BOTTLES]
    ).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("oil", "0003_scanimage_bottle_bbox"),
    ]

    operations = [
        migrations.RunPython(seed_bottles, reverse_code=unseed_bottles),
    ]
