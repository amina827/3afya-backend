"""
Oil level detection for Afia 1.5L bottles.
V11-based: full-width HSV row scanning with label exclusion.

Accuracy:
- 700-1500ml: +/-50ml (HIGH confidence)
- 300-700ml:  +/-150ml (MEDIUM confidence)
- 0-300ml:    detected as low/empty (LOW confidence on exact volume)
"""

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    pass


def _load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise ProcessingError(f"Unable to read image: {image_path}")
    return image


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Reference & calibration data
# =====================================================================

REFERENCE_LEVELS = [
    {"ml": 0,    "pct": 0.00,  "file": "0.jpeg"},
    {"ml": 100,  "pct": 6.67,  "file": "100.jpeg"},
    {"ml": 200,  "pct": 13.33, "file": "200.jpeg"},
    {"ml": 300,  "pct": 20.00, "file": "300.jpeg"},
    {"ml": 400,  "pct": 26.67, "file": "400.jpeg"},
    {"ml": 500,  "pct": 33.33, "file": "500.jpeg"},
    {"ml": 600,  "pct": 40.00, "file": "600.jpeg"},
    {"ml": 700,  "pct": 46.67, "file": "700.jpeg"},
    {"ml": 800,  "pct": 53.33, "file": "800.jpeg"},
    {"ml": 900,  "pct": 60.00, "file": "900.jpeg"},
    {"ml": 1000, "pct": 66.67, "file": "1000.jpeg"},
    {"ml": 1100, "pct": 73.33, "file": "1100.jpeg"},
    {"ml": 1200, "pct": 80.00, "file": "1200.jpeg"},
    {"ml": 1300, "pct": 86.67, "file": "1300.jpeg"},
    {"ml": 1400, "pct": 93.33, "file": "1400.jpeg"},
    {"ml": 1500, "pct": 100.00, "file": "1500.jpeg"},
]
TOTAL_BOTTLE_ML = 1500
REFERENCE_DIR = Path(__file__).parent / "reference_data"

# Calibration table (empirically validated from V6/V11)
CALIBRATION_TABLE = [
    {"fill_ratio": 0.000,  "volume_ml": 0},
    {"fill_ratio": 0.300,  "volume_ml": 500},
    {"fill_ratio": 0.4361, "volume_ml": 700},
    {"fill_ratio": 0.5160, "volume_ml": 800},
    {"fill_ratio": 0.5401, "volume_ml": 900},
    {"fill_ratio": 0.5684, "volume_ml": 1000},
    {"fill_ratio": 0.6321, "volume_ml": 1100},
    {"fill_ratio": 0.7020, "volume_ml": 1200},
    {"fill_ratio": 0.7888, "volume_ml": 1300},
    {"fill_ratio": 0.8815, "volume_ml": 1400},
    {"fill_ratio": 0.9925, "volume_ml": 1500},
]

# HSV ranges
CAP_HSV_RANGES = [
    {"lower": [0, 100, 70], "upper": [10, 255, 255]},
    {"lower": [160, 100, 70], "upper": [179, 255, 255]},
]
OIL_HSV_LOWER = np.array([15, 30, 60])
OIL_HSV_UPPER = np.array([45, 255, 250])
LABEL_GREEN_HSV_LOWER = np.array([35, 50, 30])
LABEL_GREEN_HSV_UPPER = np.array([90, 255, 255])

# Bottle geometry ratios
CAP_TO_BODY_RATIO = 9.0
CAP_WIDTH_FACTOR = 1.8

# Detection thresholds
OIL_ROW_THRESHOLD = 40


def invalidate_reference_cache():
    """Clear reference caches. Called from OilReference signals."""
    pass


def extract_reference_features(image_path: str):
    """Extract features for a reference image."""
    try:
        image = _load_image(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cap = _detect_cap(hsv, image.shape)
        if cap is None:
            return _empty_features()

        oil_result = _detect_oil_level(image, hsv, cap)
        return {
            "brightness_profile": [],
            "histogram": [],
            "golden_profile": [],
            "golden_amount": oil_result["fill_ratio"],
            "normalized_cache_path": "",
        }
    except Exception:
        return _empty_features()


def _empty_features():
    return {
        "brightness_profile": [],
        "histogram": [],
        "golden_profile": [],
        "golden_amount": 0.0,
        "normalized_cache_path": "",
    }


# =====================================================================
# Core detection (V11 algorithm)
# =====================================================================


def _detect_cap(hsv, image_shape):
    """Detect the red bottle cap in the upper half of the image.

    Returns dict with top_y, bottom_y, center_x, height, width — or None.
    """
    h, w = image_shape[:2]

    red_masks = [
        cv2.inRange(hsv, np.array(r["lower"]), np.array(r["upper"]))
        for r in CAP_HSV_RANGES
    ]
    red_mask = red_masks[0] | red_masks[1]

    # Restrict to upper half
    upper = red_mask.copy()
    upper[h // 2:] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(upper, 8)
    best = None
    best_score = 0

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 2000:
            continue
        aspect = ww / max(hh, 1)
        cx = x + ww // 2
        # Cap should be near horizontal center and have right aspect
        if 0.8 <= aspect <= 2.5 and area > 5000 and (w * 0.3 < cx < w * 0.7):
            score = area * (1 / (1 + abs(cx - w // 2) / 100))
            if score > best_score:
                best_score = score
                best = (x, y, ww, hh)

    if best is None:
        return None

    x, y, ww, hh = best
    return {
        "top_y": int(y),
        "bottom_y": int(y + hh),
        "center_x": int(x + ww // 2),
        "height": int(hh),
        "width": int(ww),
    }


def _detect_oil_level(image, hsv, cap):
    """Detect oil level using full-width HSV row scanning with label exclusion.

    Returns dict with fill_ratio, confidence, oil_top_y, bottle bounds, etc.
    """
    h, w = image.shape[:2]

    cap_h = cap["height"]
    cap_bot = cap["bottom_y"]
    cap_cx = cap["center_x"]

    # Bottle bounds from cap geometry
    bottle_bottom_y = min(h - 10, cap_bot + int(CAP_TO_BODY_RATIO * cap_h))
    bottle_left = max(0, cap_cx - int(CAP_WIDTH_FACTOR * cap_h))
    bottle_right = min(w, cap_cx + int(CAP_WIDTH_FACTOR * cap_h))
    bottle_height = bottle_bottom_y - cap_bot

    # Build label exclusion mask (green + red, dilated)
    green_mask = cv2.inRange(hsv, LABEL_GREEN_HSV_LOWER, LABEL_GREEN_HSV_UPPER)
    red_mask_1 = cv2.inRange(
        hsv, np.array(CAP_HSV_RANGES[0]["lower"]), np.array(CAP_HSV_RANGES[0]["upper"])
    )
    red_mask_2 = cv2.inRange(
        hsv, np.array(CAP_HSV_RANGES[1]["lower"]), np.array(CAP_HSV_RANGES[1]["upper"])
    )
    label_mask = green_mask | red_mask_1 | red_mask_2
    label_mask = cv2.dilate(label_mask, np.ones((11, 11), np.uint8), iterations=2)

    # Build oil mask using HSV thresholds
    oil_h = (hsv[:, :, 0] >= OIL_HSV_LOWER[0]) & (hsv[:, :, 0] <= OIL_HSV_UPPER[0])
    oil_s = hsv[:, :, 1] >= OIL_HSV_LOWER[1]
    oil_v = (hsv[:, :, 2] >= OIL_HSV_LOWER[2]) & (hsv[:, :, 2] <= OIL_HSV_UPPER[2])
    oil_mask = (oil_h & oil_s & oil_v).astype(np.uint8) * 255

    # Remove label from oil mask
    oil_clean = oil_mask & ~label_mask

    # Restrict to bottle body
    body_mask = np.zeros((h, w), dtype=np.uint8)
    body_mask[cap_bot:bottle_bottom_y, bottle_left:bottle_right] = 255
    oil_in_body = oil_clean & body_mask

    # Row-based oil pixel count
    row_oil_count = np.sum(oil_in_body > 0, axis=1)
    smoothed = np.convolve(row_oil_count, np.ones(15) / 15, mode="same")

    # Check bottom 30% zone for oil presence
    bottom_zone_start = bottle_bottom_y - int(bottle_height * 0.3)
    bottom_max = (
        np.max(smoothed[bottom_zone_start:bottle_bottom_y])
        if bottom_zone_start < bottle_bottom_y
        else 0
    )

    bounds = (bottle_left, cap_bot, bottle_right, bottle_bottom_y)

    if bottom_max < OIL_ROW_THRESHOLD:
        full_max = np.max(smoothed[cap_bot:bottle_bottom_y])
        if full_max < OIL_ROW_THRESHOLD:
            return {
                "has_oil": False,
                "oil_top_y": bottle_bottom_y,
                "fill_ratio": 0.0,
                "bottle_bottom_y": bottle_bottom_y,
                "bottle_height_px": bottle_height,
                "bottle_bounds": bounds,
                "confidence": "high",
                "confidence_note": "Bottle detected as empty.",
            }

    # Find highest y with oil
    oil_top_y = bottle_bottom_y
    for y in range(cap_bot, bottle_bottom_y):
        if smoothed[y] >= OIL_ROW_THRESHOLD:
            oil_top_y = int(y)
            break

    pixel_height = bottle_bottom_y - oil_top_y
    fill_ratio = pixel_height / bottle_height if bottle_height > 0 else 0.0

    # Confidence based on fill_ratio zones
    if fill_ratio < 0.40:
        confidence = "low"
        confidence_note = (
            "Front-view detection unreliable for low volumes (<=700ml) "
            "due to label corn drawing interference."
        )
    elif fill_ratio < 0.55:
        confidence = "medium"
        confidence_note = "Reading in transition zone (700-900ml)."
    else:
        confidence = "high"
        confidence_note = "Reading in validated reliable range."

    return {
        "has_oil": True,
        "oil_top_y": int(oil_top_y),
        "fill_ratio": float(fill_ratio),
        "bottle_bottom_y": int(bottle_bottom_y),
        "bottle_height_px": int(bottle_height),
        "bottle_bounds": bounds,
        "confidence": confidence,
        "confidence_note": confidence_note,
    }


def _fill_ratio_to_ml(ratio):
    """Convert fill_ratio to ml using calibration table with linear interpolation."""
    if ratio <= 0:
        return 0

    points = sorted(CALIBRATION_TABLE, key=lambda p: p["fill_ratio"])

    if ratio >= points[-1]["fill_ratio"]:
        return points[-1]["volume_ml"]

    for i in range(len(points) - 1):
        r1 = points[i]["fill_ratio"]
        r2 = points[i + 1]["fill_ratio"]
        if r1 <= ratio <= r2:
            v1 = points[i]["volume_ml"]
            v2 = points[i + 1]["volume_ml"]
            if r2 == r1:
                return v1
            return v1 + (v2 - v1) * (ratio - r1) / (r2 - r1)

    return 0


def _draw_overlay(image, cap, oil_result, volume_ml, fill_ratio):
    """Draw detection overlay on the image."""
    overlay = image.copy()
    bounds = oil_result["bottle_bounds"]
    bx1, by1, bx2, by2 = bounds

    # Bottle bounds (yellow rectangle)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 200, 200), 3)

    # Bottle bottom (magenta line)
    bot_y = oil_result["bottle_bottom_y"]
    cv2.line(overlay, (bx1, bot_y), (bx2, bot_y), (255, 0, 255), 3)

    # Oil line (red)
    if oil_result["has_oil"]:
        oil_y = oil_result["oil_top_y"]
        cv2.line(overlay, (0, oil_y), (image.shape[1], oil_y), (0, 0, 255), 4)

    # Text
    h, w = image.shape[:2]
    fs = max(0.6, min(1.2, w / 600.0))
    pct = fill_ratio * 100
    text = f"{volume_ml:.0f}ml ({pct:.1f}%)"
    cv2.putText(
        overlay, text, (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2, cv2.LINE_AA,
    )

    conf = oil_result["confidence"]
    cv2.putText(
        overlay, f"Confidence: {conf}", (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (0, 200, 0), 2, cv2.LINE_AA,
    )

    return overlay


# =====================================================================
# Main processing pipeline
# =====================================================================


def process_bottle_image(image_path: str, bottle_spec):
    """Process a bottle image and detect oil level.

    Args:
        image_path: Absolute path to the input image.
        bottle_spec: BottleSpecification model instance.

    Returns dict with: processed_path, oil_height_pixels, bottle_height_pixels,
        oil_ratio, remaining_volume_liters, consumed_volume_liters,
        remaining_cups, consumed_cups, confidence_score, processing_time_ms,
        bottle_bbox.
    """
    start_time = time.time()

    image = _load_image(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]

    # Step 1: Detect cap
    cap = _detect_cap(hsv, image.shape)
    if cap is None:
        raise ProcessingError(
            "Could not detect bottle cap. Ensure the red cap is visible "
            "in the upper portion of the image."
        )

    # Step 2: Detect oil level (full-width scan with label exclusion)
    oil_result = _detect_oil_level(image, hsv, cap)
    fill_ratio = oil_result["fill_ratio"]

    # Step 3: Convert to volume
    remaining_ml = _fill_ratio_to_ml(fill_ratio)
    total_ml = float(bottle_spec.total_volume_liters) * 1000
    consumed_ml = max(0.0, total_ml - remaining_ml)

    remaining_liters = remaining_ml / 1000.0
    consumed_liters = consumed_ml / 1000.0

    cup_ratio = float(bottle_spec.cup_conversion_ratio)
    remaining_cups = remaining_liters / cup_ratio if cup_ratio > 0 else 0
    consumed_cups = consumed_liters / cup_ratio if cup_ratio > 0 else 0

    # Confidence score
    confidence_label = oil_result["confidence"]
    confidence_map = {"high": 0.92, "medium": 0.75, "low": 0.50}
    confidence_score = confidence_map.get(confidence_label, 0.5)

    # Pixel values
    body_h = oil_result["bottle_height_px"]
    oil_height_px = body_h * fill_ratio

    # Draw overlay and save
    overlay = _draw_overlay(image, cap, oil_result, remaining_ml, fill_ratio)
    processed_dir = Path(settings.MEDIA_ROOT) / "scans" / "processed"
    _ensure_dir(processed_dir)
    processed_fname = f"processed_{uuid.uuid4().hex}.jpg"
    processed_path = str(processed_dir / processed_fname)
    cv2.imwrite(processed_path, overlay)

    processing_time = int((time.time() - start_time) * 1000)

    bounds = oil_result["bottle_bounds"]
    return {
        "processed_path": f"scans/processed/{processed_fname}",
        "oil_height_pixels": round(oil_height_px, 1),
        "bottle_height_pixels": round(float(body_h), 1),
        "oil_ratio": round(fill_ratio, 4),
        "remaining_volume_liters": round(remaining_liters, 4),
        "consumed_volume_liters": round(consumed_liters, 4),
        "remaining_cups": round(remaining_cups, 2),
        "consumed_cups": round(consumed_cups, 2),
        "confidence_score": round(confidence_score, 2),
        "processing_time_ms": processing_time,
        "bottle_bbox": {
            "x": bounds[0],
            "y": bounds[1],
            "w": bounds[2] - bounds[0],
            "h": bounds[3] - bounds[1],
            "image_w": w,
            "image_h": h,
        },
    }


def render_target_overlay(image_path: str, bottle_spec, target_cups: float):
    """Draw a target-level line on the bottle image."""
    image = _load_image(image_path)
    h, w = image.shape[:2]
    total = float(bottle_spec.total_volume_liters)
    target_l = float(target_cups) * float(bottle_spec.cup_conversion_ratio)
    ratio = min(1.0, max(0.0, target_l / total if total else 0))

    by = int(h * 0.15)
    bh = int(h * 0.70)

    target_y = by + int(bh * (1 - ratio))
    overlay = image.copy()
    cv2.line(overlay, (0, target_y), (w, target_y), (0, 165, 255), 3)
    fs = max(0.5, min(1.0, w / 400.0))
    cv2.putText(
        overlay,
        f"Target: {target_cups:.1f} cups ({target_l:.2f}L)",
        (10, max(25, target_y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )

    target_dir = Path(settings.MEDIA_ROOT) / "scans" / "targets"
    _ensure_dir(target_dir)
    fname = f"target_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(target_dir / fname), overlay)
    return f"scans/targets/{fname}"
