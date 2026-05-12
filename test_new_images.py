"""Test detection on the user's problematic images."""
import os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

class _S:
    MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")

sys.modules.setdefault("django", type(sys)("django"))
sys.modules.setdefault("django.conf", type(sys)("django.conf"))
sys.modules["django.conf"].settings = _S()

from oil.services.image_processing import (
    _detect_cap, _detect_oil_level, _fill_ratio_to_ml, _load_image,
    _find_label_zone, _compute_body_oil_excl_pct,
    CAP_TO_BODY_RATIO, CAP_WIDTH_FACTOR, OIL_HSV_LOWER, OIL_HSV_UPPER,
)
import cv2
import numpy as np

TEST_IMAGES = [
    (r"c:\Users\Dell\Downloads\1300.jpeg", 1300),
    (r"c:\Users\Dell\Downloads\100.jpeg", 100),
    (r"c:\Users\Dell\Downloads\200.jpeg", 200),
    (r"c:\Users\Dell\Downloads\300.jpeg", 300),
]

def analyze_image(path, expected_ml):
    image = _load_image(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]

    cap = _detect_cap(hsv, image.shape)
    if cap is None:
        print(f"  {expected_ml:>5}ml | CAP NOT FOUND | img={w}x{h}")
        return

    print(f"  {expected_ml:>5}ml | img={w}x{h} | cap: top={cap['top_y']}, bot={cap['bottom_y']}, cx={cap['center_x']}, w={cap['width']}, h={cap['height']}")

    cap_bot = cap["bottom_y"]
    cap_h = cap["height"]
    cap_cx = cap["center_x"]
    bot_y = min(h - 10, cap_bot + int(CAP_TO_BODY_RATIO * cap_h))
    bl = max(0, cap_cx - int(CAP_WIDTH_FACTOR * cap_h))
    br = min(w, cap_cx + int(CAP_WIDTH_FACTOR * cap_h))
    body_h = bot_y - cap_bot
    body_w = br - bl

    label_top, label_bot = _find_label_zone(hsv, cap_bot, bot_y, bl, br)
    lt_rel = label_top - cap_bot
    lb_rel = label_bot - cap_bot

    print(f"         | body: h={body_h}, w={body_w}, bot_y={bot_y}")
    print(f"         | label: top={label_top}(rel={lt_rel}), bot={label_bot}(rel={lb_rel})")
    print(f"         | label coverage: {lt_rel/body_h*100:.1f}% - {lb_rel/body_h*100:.1f}%")

    # Side strip analysis
    strip_w = max(10, int(body_w * 0.06))
    lx1, lx2 = bl + 3, bl + 3 + strip_w
    oil_mask = cv2.inRange(hsv, OIL_HSV_LOWER, OIL_HSV_UPPER)
    strip = oil_mask[cap_bot:bot_y, lx1:lx2]
    row_frac = np.sum(strip > 0, axis=1).astype(float) / max(strip_w, 1)
    ks = min(15, len(row_frac))
    smoothed = np.convolve(row_frac, np.ones(ks) / ks, mode="same")

    # Find runs at 25% and 55%
    for thresh_name, thresh_val in [("25%", 0.25), ("55%", 0.55)]:
        run_start = None
        run_len = 0
        runs = []
        for i in range(len(smoothed)):
            if smoothed[i] >= thresh_val:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= 20:
                    runs.append((run_start, run_start + run_len - 1, run_len))
                run_start = None
                run_len = 0
        if run_len >= 20:
            runs.append((run_start, run_start + run_len - 1, run_len))

        run_str = ", ".join([f"[{s}-{e}]({l}px)" for s, e, l in runs[:4]])
        print(f"         | strip@{thresh_name}: {run_str if run_str else 'NONE'}")

    # Body oil excl
    oil_excl = _compute_body_oil_excl_pct(hsv, cap_bot, bot_y, bl, br)
    print(f"         | oil_excl_pct: {oil_excl:.2f}%")

    # Run detection
    oil_result = _detect_oil_level(image, hsv, cap)
    fr = oil_result["fill_ratio"]
    ml = _fill_ratio_to_ml(fr)
    err = ml - expected_ml
    zone = oil_result.get("detection_zone", "?")
    conf = oil_result["confidence"]
    note = oil_result.get("confidence_note", "")[:50]

    print(f"         | RESULT: {ml:.0f}ml (expected {expected_ml}ml, error {err:+.0f}ml)")
    print(f"         | ratio={fr:.3f}, conf={conf}, zone={zone}")
    print(f"         | note: {note}")
    print()

def main():
    print("=" * 100)
    print(" Testing User's Problematic Images")
    print("=" * 100)
    for path, expected in TEST_IMAGES:
        if os.path.exists(path):
            analyze_image(path, expected)
        else:
            print(f"  MISSING: {path}")

if __name__ == "__main__":
    main()
