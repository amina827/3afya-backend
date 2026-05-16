import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class _S:
    MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
sys.modules.setdefault("django", type(sys)("django"))
sys.modules.setdefault("django.conf", type(sys)("django.conf"))
sys.modules["django.conf"].settings = _S()

import cv2
from oil.services.image_processing import (
    _detect_cap, _find_label_zone, _load_image, REFERENCE_DIR, CAP_TO_BODY_RATIO, CAP_WIDTH_FACTOR,
)

print(f"{'Vol':>5} | {'cap_bot':>7} | {'cap_h':>5} | {'cap_ratio_bot':>13} | {'label_top':>9} | {'label_bot':>9} | {'label_h':>7} | {'body_label_pct':>14}")
print("-" * 110)

for vol in [0, 100, 500, 1000, 1500]:
    path = str(REFERENCE_DIR / f"{vol}.jpeg")
    image = _load_image(path)
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cap = _detect_cap(hsv, image.shape)

    cap_h = cap["height"]
    cap_bot = cap["bottom_y"]
    cap_cx = cap["center_x"]
    cap_ratio_bot = min(h - 10, cap_bot + int(CAP_TO_BODY_RATIO * cap_h))
    bl = max(0, cap_cx - int(CAP_WIDTH_FACTOR * cap_h))
    br = min(w, cap_cx + int(CAP_WIDTH_FACTOR * cap_h))

    lt, lb = _find_label_zone(hsv, cap_bot, cap_ratio_bot, bl, br)
    body_h = cap_ratio_bot - cap_bot
    label_h = lb - lt
    pct = label_h / body_h * 100 if body_h else 0
    print(f"{vol:>5} | {cap_bot:>7} | {cap_h:>5} | {cap_ratio_bot:>13} | {lt:>9} | {lb:>9} | {label_h:>7} | {pct:>13.1f}%")
