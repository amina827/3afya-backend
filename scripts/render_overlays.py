"""Render overlays for 0/500/1000ml reference images so the bbox/lines can
be inspected visually."""
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
    _detect_cap, _detect_oil_level, _draw_overlay, _fill_ratio_to_ml,
    _load_image, REFERENCE_DIR,
)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_debug")
os.makedirs(out_dir, exist_ok=True)

for vol in [0, 500, 1000, 1500]:
    path = str(REFERENCE_DIR / f"{vol}.jpeg")
    image = _load_image(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cap = _detect_cap(hsv, image.shape)
    oil = _detect_oil_level(image, hsv, cap)
    ml = _fill_ratio_to_ml(oil["fill_ratio"])
    overlay = _draw_overlay(image, cap, oil, ml, oil["fill_ratio"])

    out = os.path.join(out_dir, f"overlay_{vol}ml.jpg")
    cv2.imwrite(out, overlay)
    bounds = oil["bottle_bounds"]
    print(f"{vol}ml -> {out}  bounds={bounds}  img_h={image.shape[0]}")
