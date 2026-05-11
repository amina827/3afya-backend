"""Test V12 oil detection on all 16 reference images."""
import os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

class _S:
    MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")

sys.modules.setdefault("django", type(sys)("django"))
sys.modules.setdefault("django.conf", type(sys)("django.conf"))
sys.modules["django.conf"].settings = _S()

from oil.services.image_processing import (
    _detect_cap, _detect_oil_level, _fill_ratio_to_ml, _load_image, REFERENCE_DIR,
)
import cv2

VOLUMES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

def main():
    print("=" * 95)
    print(" V12 Algorithm Test — 3-Zone Detection")
    print("=" * 95)
    print(f"{'Actual':>7} | {'Detect':>7} | {'Error':>7} | {'Ratio':>6} | {'Conf':>6} | {'Zone':>12} | Note")
    print("-" * 95)

    errors = {"high": [], "medium": [], "low": [], "all": []}

    for vol in VOLUMES:
        path = str(REFERENCE_DIR / f"{vol}.jpeg")
        if not os.path.exists(path):
            print(f"{vol:>5}ml | MISSING")
            continue

        image = _load_image(path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cap = _detect_cap(hsv, image.shape)
        if cap is None:
            print(f"{vol:>5}ml | CAP NOT FOUND")
            continue

        oil = _detect_oil_level(image, hsv, cap)
        fr = oil["fill_ratio"]
        ml = _fill_ratio_to_ml(fr)
        err = ml - vol
        conf = oil["confidence"]
        zone = oil.get("detection_zone", "?")

        errors[conf].append(abs(err))
        errors["all"].append(abs(err))

        note = oil.get("confidence_note", "")[:40]
        print(f"{vol:>5}ml | {ml:>5.0f}ml | {err:>+6.0f}ml | {fr:>5.3f} | {conf:>6} | {zone:>12} | {note}")

    print("-" * 95)
    for k in ["high", "medium", "low", "all"]:
        e = errors[k]
        if e:
            print(f"  {k:>6}: n={len(e):>2}, avg_err={sum(e)/len(e):>6.1f}ml, max_err={max(e):>5.0f}ml")

    # Strict test: 1200-1500ml must be within ±100ml
    print()
    failed = 0
    for vol in [1200, 1300, 1400, 1500]:
        path = str(REFERENCE_DIR / f"{vol}.jpeg")
        image = _load_image(path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cap = _detect_cap(hsv, image.shape)
        oil = _detect_oil_level(image, hsv, cap)
        ml = _fill_ratio_to_ml(oil["fill_ratio"])
        if abs(ml - vol) > 100:
            print(f"  FAIL: {vol}ml -> {ml:.0f}ml (error > 100ml)")
            failed += 1

    # Empty bottle MUST be 0ml
    path = str(REFERENCE_DIR / "0.jpeg")
    image = _load_image(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cap = _detect_cap(hsv, image.shape)
    oil = _detect_oil_level(image, hsv, cap)
    ml = _fill_ratio_to_ml(oil["fill_ratio"])
    if ml > 50:
        print(f"  FAIL: 0ml -> {ml:.0f}ml (empty bottle not detected)")
        failed += 1

    if failed == 0:
        print("  All critical tests PASSED")
    else:
        print(f"  {failed} test(s) FAILED")

if __name__ == "__main__":
    main()
