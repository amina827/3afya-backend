"""
Test V11 oil detection on all 16 reference images.
Run: python test_oil_detection.py
"""
import os
import sys
import cv2
import numpy as np

# Minimal Django setup so we can import image_processing
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

# We only need cv2/numpy — mock Django settings to avoid full Django init
class FakeSettings:
    MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")

sys.modules.setdefault("django", type(sys)("django"))
sys.modules.setdefault("django.conf", type(sys)("django.conf"))
sys.modules["django.conf"].settings = FakeSettings()

from oil.services.image_processing import (
    _detect_cap,
    _detect_oil_level,
    _fill_ratio_to_ml,
    _load_image,
    REFERENCE_DIR,
)

VOLUMES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
           1000, 1100, 1200, 1300, 1400, 1500]


def main():
    print("=" * 80)
    print(" V11 Algorithm Test on 16 Reference Images")
    print("=" * 80)
    print(f"{'Actual':>7} | {'Detected':>9} | {'Error':>7} | {'Ratio':>6} | {'Confidence':>10} | Note")
    print("-" * 80)

    high_conf_errors = []
    all_errors = []

    for vol in VOLUMES:
        img_path = str(REFERENCE_DIR / f"{vol}.jpeg")
        if not os.path.exists(img_path):
            print(f"{vol:>5}ml | MISSING: {img_path}")
            continue

        image = _load_image(img_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cap = _detect_cap(hsv, image.shape)
        if cap is None:
            print(f"{vol:>5}ml | CAP NOT FOUND")
            continue

        oil = _detect_oil_level(image, hsv, cap)
        fill_ratio = oil["fill_ratio"]
        detected_ml = _fill_ratio_to_ml(fill_ratio)
        error = detected_ml - vol
        conf = oil["confidence"]

        if conf == "high":
            high_conf_errors.append(abs(error))
        all_errors.append(abs(error))

        note = oil.get("confidence_note", "")[:50]
        print(
            f"{vol:>5}ml | {detected_ml:>7.0f}ml | {error:>+6.0f}ml | "
            f"{fill_ratio:>5.3f} | {conf:>10} | {note}"
        )

    print("-" * 80)
    if high_conf_errors:
        avg = sum(high_conf_errors) / len(high_conf_errors)
        mx = max(high_conf_errors)
        print(f"HIGH confidence: n={len(high_conf_errors)}, avg_err={avg:.1f}ml, max_err={mx:.0f}ml")
    if all_errors:
        avg = sum(all_errors) / len(all_errors)
        mx = max(all_errors)
        print(f"ALL images:      n={len(all_errors)}, avg_err={avg:.1f}ml, max_err={mx:.0f}ml")

    # Strict check: high-confidence images should be within +-100ml
    failed = 0
    for vol in [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]:
        img_path = str(REFERENCE_DIR / f"{vol}.jpeg")
        if not os.path.exists(img_path):
            continue
        image = _load_image(img_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cap = _detect_cap(hsv, image.shape)
        if cap is None:
            failed += 1
            continue
        oil = _detect_oil_level(image, hsv, cap)
        detected = _fill_ratio_to_ml(oil["fill_ratio"])
        if abs(detected - vol) > 100:
            print(f"FAIL: {vol}ml detected as {detected:.0f}ml (error > 100ml)")
            failed += 1

    if failed == 0:
        print("\nAll high-confidence tests PASSED (700-1500ml within +-100ml)")
    else:
        print(f"\n{failed} high-confidence test(s) FAILED")


if __name__ == "__main__":
    main()
