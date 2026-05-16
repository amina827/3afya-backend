"""Compare cap-ratio bottle bottom vs label-anchored visual bottom for each
reference image. Confirms the empty-bottle (0ml) fix without affecting
detection.
"""
import os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class _S:
    MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")

sys.modules.setdefault("django", type(sys)("django"))
sys.modules.setdefault("django.conf", type(sys)("django.conf"))
sys.modules["django.conf"].settings = _S()

import cv2
from oil.services.image_processing import (
    _detect_cap, _detect_oil_level, _load_image, REFERENCE_DIR, CAP_TO_BODY_RATIO,
)

VOLUMES = [0, 100, 500, 1000, 1500]

print(f"{'Vol':>5} | {'img_h':>5} | {'cap_h':>5} | {'cap_bot':>7} | {'cap_ratio_bot':>13} | {'visual_bot':>10} | {'diff':>5} | refined?")
print("-" * 90)

for vol in VOLUMES:
    path = str(REFERENCE_DIR / f"{vol}.jpeg")
    image = _load_image(path)
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cap = _detect_cap(hsv, image.shape)
    if cap is None:
        print(f"{vol:>5}ml | cap not found")
        continue

    cap_h = cap["height"]
    cap_bot = cap["bottom_y"]
    cap_ratio_bot = min(h - 10, cap_bot + int(CAP_TO_BODY_RATIO * cap_h))

    oil = _detect_oil_level(image, hsv, cap)
    bounds = oil["bottle_bounds"]
    visual_bot = bounds[3]

    diff = cap_ratio_bot - visual_bot
    refined = "YES" if visual_bot != cap_ratio_bot else "no"
    print(f"{vol:>5} | {h:>5} | {cap_h:>5} | {cap_bot:>7} | {cap_ratio_bot:>13} | {visual_bot:>10} | {diff:>5} | {refined}")
