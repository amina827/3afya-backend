"""
Oil level detection for Afia 1.5L bottles.
V13: Dual-threshold validated detection.

Detection strategy:
- 1200-1500ml: Above-label clear zone scan (HIGH confidence)
- 800-1100ml:  Side-strip scan with stable dual-threshold (MEDIUM confidence)
- 0-700ml:     Side-strip with F55 reference matching for unstable detections
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

# Unified calibration (from side-strip 600-1100 + above-label 1200-1500)
CALIBRATION_TABLE = [
    {"fill_ratio": 0.000, "volume_ml": 0},
    {"fill_ratio": 0.253, "volume_ml": 200},
    {"fill_ratio": 0.367, "volume_ml": 500},
    {"fill_ratio": 0.380, "volume_ml": 600},
    {"fill_ratio": 0.456, "volume_ml": 700},
    {"fill_ratio": 0.519, "volume_ml": 800},
    {"fill_ratio": 0.571, "volume_ml": 900},
    {"fill_ratio": 0.574, "volume_ml": 1000},
    {"fill_ratio": 0.631, "volume_ml": 1100},
    {"fill_ratio": 0.705, "volume_ml": 1200},
    {"fill_ratio": 0.790, "volume_ml": 1300},
    {"fill_ratio": 0.926, "volume_ml": 1400},
    {"fill_ratio": 0.993, "volume_ml": 1500},
]

# HSV ranges
CAP_HSV_RANGES = [
    {"lower": [0, 100, 70], "upper": [10, 255, 255]},
    {"lower": [160, 100, 70], "upper": [179, 255, 255]},
]
OIL_HSV_LOWER = np.array([15, 30, 60])
OIL_HSV_UPPER = np.array([45, 255, 250])
LABEL_GREEN_LOWER = np.array([35, 50, 30])
LABEL_GREEN_UPPER = np.array([90, 255, 255])

# Bottle geometry
CAP_TO_BODY_RATIO = 9.0
CAP_WIDTH_FACTOR = 1.8


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
# Cap detection
# =====================================================================


def _detect_cap(hsv, image_shape):
    """Detect the red bottle cap in the upper half of the image.

    Returns dict with top_y, bottom_y, center_x, height, width — or None.
    """
    h, w = image_shape[:2]

    r1 = cv2.inRange(hsv, np.array(CAP_HSV_RANGES[0]["lower"]),
                     np.array(CAP_HSV_RANGES[0]["upper"]))
    r2 = cv2.inRange(hsv, np.array(CAP_HSV_RANGES[1]["lower"]),
                     np.array(CAP_HSV_RANGES[1]["upper"]))
    red_mask = r1 | r2

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


# =====================================================================
# Label zone detection
# =====================================================================


def _find_label_zone(hsv, cap_bot, bot_y, bl, br):
    """Find the label region using contour detection on combined label colors.

    The Afia label contains green + red + white panels. Detecting only green
    misses labels where green is partially shadowed/occluded. Build a combined
    strong-color mask (green + red), close gaps with morphology, then take the
    largest wide connected component as the label.

    Returns (label_top_y, label_bottom_y).
    """
    body_h = bot_y - cap_bot
    body_w = br - bl

    green = cv2.inRange(hsv, LABEL_GREEN_LOWER, LABEL_GREEN_UPPER)
    red1 = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 80, 60]), np.array([179, 255, 255]))
    label_strong = green | red1 | red2

    body_mask = np.zeros_like(label_strong)
    body_mask[cap_bot:bot_y, bl:br] = label_strong[cap_bot:bot_y, bl:br]

    # Close gaps to merge green/red regions of the same label
    kernel = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    best_area = 0
    best_y = None
    best_h = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # Real labels span most of bottle width
        if w < body_w * 0.30 or area < 800:
            continue
        if area > best_area:
            best_area = area
            best_y = y
            best_h = h

    if best_y is not None:
        return int(best_y), int(best_y + best_h)

    # Fallback 1: green-only detection (original behavior)
    green_body = green.copy()
    green_body[:cap_bot] = 0
    green_body[bot_y:] = 0
    green_body[:, :bl] = 0
    green_body[:, br:] = 0
    green_rows = np.sum(green_body > 0, axis=1)
    green_ys = np.where(green_rows > 10)[0]
    if len(green_ys) > 0:
        return int(green_ys[0]), int(green_ys[-1])

    # Fallback 2: assume label starts at 36% and ends at 98% of body
    return cap_bot + int(body_h * 0.36), cap_bot + int(body_h * 0.98)


# =====================================================================
# Zone 1: Above-label oil detection (MOST RELIABLE)
# =====================================================================


def _scan_above_label(hsv, cap_bot, label_top, bl, br, body_h):
    """Scan the clear zone above the label for oil.

    Returns (oil_top_y, fill_ratio) or (None, 0.0) if no oil found.
    Requires at least MIN_CONSEC consecutive rows above threshold to avoid
    false positives from background colors through clear plastic.
    """
    if label_top <= cap_bot + 20:
        return None, 0.0

    oil_mask = cv2.inRange(hsv, OIL_HSV_LOWER, OIL_HSV_UPPER)

    # Restrict to above-label zone
    zone = oil_mask.copy()
    zone[:cap_bot] = 0
    zone[label_top:] = 0
    zone[:, :bl] = 0
    zone[:, br:] = 0

    row_counts = np.sum(zone > 0, axis=1)
    ks = min(15, label_top - cap_bot)
    if ks < 3:
        return None, 0.0
    smoothed = np.convolve(row_counts, np.ones(ks) / ks, mode="same")

    # Proportional threshold: 15% of bottle width (was 8%, too sensitive)
    bottle_w = br - bl
    threshold = max(30, int(bottle_w * 0.15))

    # Require MIN_CONSEC consecutive rows above threshold
    # Real oil creates a continuous band; reflections/background are scattered
    MIN_CONSEC = 20
    run_start = None
    run_len = 0
    for y in range(cap_bot, label_top):
        if smoothed[y] >= threshold:
            if run_start is None:
                run_start = y
            run_len += 1
        else:
            if run_len >= MIN_CONSEC:
                bot_y = cap_bot + body_h
                fill_ratio = (bot_y - run_start) / body_h
                return int(run_start), float(fill_ratio)
            run_start = None
            run_len = 0
    # Check final run
    if run_len >= MIN_CONSEC and run_start is not None:
        bot_y = cap_bot + body_h
        fill_ratio = (bot_y - run_start) / body_h
        return int(run_start), float(fill_ratio)

    return None, 0.0


# =====================================================================
# Zone 2: Side-strip oil detection with dual-threshold validation
# =====================================================================

# Reference F55 values (fill_ratio at 55% threshold) from calibration images.
# Used for nearest-reference matching when detection is unstable.
_REF_F55 = [
    (0.04, 400),
    (0.14, 600),
    (0.16, 500),   # Also 700ml — disambiguated by F25
    (0.18, 100),
    (0.20, 200),
]


def _find_first_sustained_run(smoothed, threshold, min_run=30):
    """Find first run of min_run consecutive rows above threshold.

    Returns (start_index, run_length) or (None, 0).
    """
    run_start = None
    run_len = 0
    for i in range(len(smoothed)):
        if smoothed[i] >= threshold:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_len >= min_run:
                return run_start, run_len
            run_start = None
            run_len = 0
    if run_len >= min_run:
        return run_start, run_len
    return None, 0


def _match_f55_reference(f55, f25):
    """Find the closest reference by F55 fill ratio.

    For 500/700ml ambiguity (both F55~0.16), use F25 as tiebreaker.
    """
    best_vol = 0
    best_dist = float("inf")
    for ref_f55, ref_vol in _REF_F55:
        d = abs(f55 - ref_f55)
        if d < best_dist:
            best_dist = d
            best_vol = ref_vol

    # Disambiguate 500 vs 700: both have F55~0.16
    # Use tight range (±0.01) to avoid catching 100ml(0.18) or 600ml(0.14)
    if abs(f55 - 0.16) < 0.012:
        best_vol = 500 if f25 < 0.42 else 700

    return best_vol


def _scan_side_strips(hsv, cap_bot, bot_y, bl, br, body_h):
    """Scan left side strip with dual-threshold validation.

    Uses 25% threshold for primary detection and 55% threshold to validate.
    If 25% and 55% results disagree (|F25-F55| > 0.10), the 25% detection
    is a label artifact — uses F55 nearest-reference matching instead.

    Returns (oil_top_y, fill_ratio, detection_info) or (None, 0.0, {}).
    """
    body_w = br - bl
    strip_w = max(10, int(body_w * 0.06))
    lx1, lx2 = bl + 3, bl + 3 + strip_w

    oil_mask = cv2.inRange(hsv, OIL_HSV_LOWER, OIL_HSV_UPPER)
    strip = oil_mask[cap_bot:bot_y, lx1:lx2]
    if strip.size == 0:
        return None, 0.0, {}

    row_frac = np.sum(strip > 0, axis=1).astype(float) / max(strip_w, 1)
    ks = min(15, len(row_frac))
    if ks < 3:
        return None, 0.0, {}
    smoothed = np.convolve(row_frac, np.ones(ks) / ks, mode="same")

    # Scan at 25% threshold (sensitive, catches both oil and artifacts)
    s25, len25 = _find_first_sustained_run(smoothed, 0.25)

    if s25 is None:
        return None, 0.0, {}

    # Density check for 25% run
    remaining = smoothed[s25:]
    filled = np.sum(remaining >= 0.25)
    if filled < 30 or filled / len(remaining) < 0.25:
        return None, 0.0, {}

    f25 = (body_h - s25) / body_h

    # Scan at 55% threshold (strict, filters out thin label artifacts)
    s55, len55 = _find_first_sustained_run(smoothed, 0.55)

    if s55 is None:
        # 25% run doesn't survive at 55% — very likely artifact
        # Check body oil excluding label as last resort
        return None, 0.0, {"f25": f25, "rejected": True}

    f55 = (body_h - s55) / body_h
    stability = abs(f25 - f55)

    if stability < 0.10:
        # STABLE: 25% and 55% agree — real oil signal
        oil_top_y = cap_bot + s25
        return int(oil_top_y), float(f25), {"method": "stable", "f25": f25, "f55": f55}

    # UNSTABLE: 25% was label artifact, 55% shows where real oil is
    # Use nearest-reference matching on F55 value
    matched_vol = _match_f55_reference(f55, f25)
    # Convert matched volume back to fill_ratio via calibration table
    matched_ratio = _volume_to_fill_ratio(matched_vol)
    oil_top_y = int(cap_bot + body_h * (1 - matched_ratio))
    return oil_top_y, float(matched_ratio), {
        "method": "f55_matched",
        "f25": f25,
        "f55": f55,
        "matched_vol": matched_vol,
    }


def _volume_to_fill_ratio(volume_ml):
    """Reverse lookup: volume_ml -> fill_ratio from calibration table."""
    points = sorted(CALIBRATION_TABLE, key=lambda p: p["volume_ml"])
    if volume_ml <= 0:
        return 0.0
    if volume_ml >= points[-1]["volume_ml"]:
        return points[-1]["fill_ratio"]
    for i in range(len(points) - 1):
        v1 = points[i]["volume_ml"]
        v2 = points[i + 1]["volume_ml"]
        if v1 <= volume_ml <= v2:
            r1 = points[i]["fill_ratio"]
            r2 = points[i + 1]["fill_ratio"]
            if v2 == v1:
                return r1
            return r1 + (r2 - r1) * (volume_ml - v1) / (v2 - v1)
    return 0.0


# =====================================================================
# Meniscus edge detection (oil surface)
# =====================================================================


def _find_horizontal_edges(image, top_y, bot_y, bl, br):
    """Find prominent horizontal lines (oil surface candidates) in a region.

    Uses Canny + HoughLinesP, filtered to near-horizontal long lines.
    Returns list of (absolute_y, score) sorted descending by score.
    """
    if bot_y <= top_y or br <= bl:
        return []

    roi = image[top_y:bot_y, bl:br]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    body_w = br - bl
    min_line_len = max(20, int(body_w * 0.45))
    max_gap = max(4, int(body_w * 0.12))
    hough_thresh = max(20, int(body_w * 0.25))

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []

    horiz = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < min_line_len * 0.8:
            continue
        angle_deg = np.degrees(np.arctan2(dy, max(dx, 1)))
        if angle_deg > 6:
            continue
        y_avg = (y1 + y2) / 2
        score = float(dx) * max(0.1, 1 - angle_deg / 6)
        horiz.append((int(top_y + y_avg), score))

    horiz.sort(key=lambda t: -t[1])
    return horiz


# =====================================================================
# Main detection orchestrator
# =====================================================================


def _compute_body_oil_excl_pct(hsv, cap_bot, bot_y, bl, br):
    """Compute oil pixel percentage in body excluding label colors.

    Returns float (0-100). Anomalously low values (<2%) indicate the label
    covers nearly all visible area, making side strip detection unreliable.
    """
    body_hsv = hsv[cap_bot:bot_y, bl:br]
    total_px = body_hsv.shape[0] * body_hsv.shape[1]
    if total_px == 0:
        return 0.0

    oil_mask = cv2.inRange(body_hsv, OIL_HSV_LOWER, OIL_HSV_UPPER)

    # Build label exclusion mask (green + red + white label areas).
    # Wider HSV ranges than before to catch shaded label edges; more dilation
    # iterations to cover the label's drop-shadow on the bottle.
    green = cv2.inRange(body_hsv, LABEL_GREEN_LOWER, LABEL_GREEN_UPPER)
    red1 = cv2.inRange(body_hsv, np.array([0, 80, 60]), np.array([12, 255, 255]))
    red2 = cv2.inRange(body_hsv, np.array([160, 80, 60]), np.array([179, 255, 255]))
    white = cv2.inRange(body_hsv, np.array([0, 0, 170]), np.array([179, 50, 255]))
    label_mask = green | red1 | red2 | white
    kernel = np.ones((7, 7), np.uint8)
    label_dilated = cv2.dilate(label_mask, kernel, iterations=3)

    oil_excl = oil_mask.copy()
    oil_excl[label_dilated > 0] = 0
    return float(np.sum(oil_excl > 0)) / total_px * 100


def _detect_oil_level(image, hsv, cap):
    """Dual-threshold validated oil level detection.

    Zone 1: Above label (clear zone) — HIGH confidence for 1200-1500ml
    Zone 2: Side strips with 25%/55% threshold validation
    Zone 3: Empty detection — if no oil in any zone

    Returns dict with fill_ratio, confidence, oil_top_y, etc.
    """
    h, w = image.shape[:2]

    cap_h = cap["height"]
    cap_bot = cap["bottom_y"]
    cap_cx = cap["center_x"]

    # Bottle bounds
    bottle_bottom_y = min(h - 10, cap_bot + int(CAP_TO_BODY_RATIO * cap_h))
    bottle_left = max(0, cap_cx - int(CAP_WIDTH_FACTOR * cap_h))
    bottle_right = min(w, cap_cx + int(CAP_WIDTH_FACTOR * cap_h))
    bottle_height = bottle_bottom_y - cap_bot

    bounds = (bottle_left, cap_bot, bottle_right, bottle_bottom_y)

    if bottle_height <= 0:
        return _empty_result(bottle_bottom_y, bottle_height, bounds)

    # Step 1: Find label zone
    label_top, label_bot = _find_label_zone(
        hsv, cap_bot, bottle_bottom_y, bottle_left, bottle_right
    )

    # Step 2: Scan ABOVE label (most reliable)
    above_oil_top, above_ratio = _scan_above_label(
        hsv, cap_bot, label_top, bottle_left, bottle_right, bottle_height
    )

    if above_oil_top is not None and above_ratio > 0.60:
        # Cross-validate: for real 1200-1500ml, side strips MUST also show
        # substantial stable oil. Background colors (granite/wood/wall) showing
        # through clear plastic can trigger false above-label on empty bottles.
        strip_chk, strip_r, strip_nfo = _scan_side_strips(
            hsv, cap_bot, bottle_bottom_y, bottle_left, bottle_right, bottle_height
        )
        # For real high-fill bottles, side strip WILL detect oil (raw f25 > 0.40).
        # For empty bottles with background false positives, side strip returns None
        # (density check fails) or has very low f25.
        raw_f25 = strip_nfo.get("f25", 0.0)
        strip_confirmed = (strip_chk is not None and raw_f25 > 0.40)

        if strip_confirmed:
            # Cross-validate with meniscus edge: a real oil surface produces
            # a sharp horizontal Canny edge within ±15px of the HSV-detected
            # top. If no such edge exists, the "oil" is likely a soft color
            # gradient from background through clear plastic — downgrade to
            # medium confidence rather than reporting "high".
            edges_near = _find_horizontal_edges(
                image,
                max(cap_bot, above_oil_top - 15),
                min(label_top, above_oil_top + 15),
                bottle_left, bottle_right,
            )
            body_w = bottle_right - bottle_left
            has_meniscus = any(e[1] >= body_w * 0.35 for e in edges_near)

            return {
                "has_oil": True,
                "oil_top_y": above_oil_top,
                "fill_ratio": above_ratio,
                "bottle_bottom_y": bottle_bottom_y,
                "bottle_height_px": bottle_height,
                "bottle_bounds": bounds,
                "confidence": "high" if has_meniscus else "medium",
                "confidence_note": (
                    "Oil visible above label — meniscus edge confirmed."
                    if has_meniscus else
                    "Oil visible above label — soft edge (no sharp meniscus)."
                ),
                "detection_zone": "above_label",
            }
        else:
            # Above-label found "oil" but side strips disagree — false positive
            # from background. Fall through to side strip detection.
            logger.info(
                "Above-label rejected: side strip not confirmed "
                "(strip_r=%.3f, method=%s)",
                strip_r, strip_nfo.get("method", "?"),
            )

    # Step 3: Side strip with dual-threshold validation
    strip_oil_top, strip_ratio, strip_info = _scan_side_strips(
        hsv, cap_bot, bottle_bottom_y, bottle_left, bottle_right, bottle_height
    )

    if strip_oil_top is not None and strip_ratio > 0.01:
        method = strip_info.get("method", "")

        # Body oil exclusion check — catches anomalous label coverage
        oil_excl = _compute_body_oil_excl_pct(
            hsv, cap_bot, bottle_bottom_y, bottle_left, bottle_right
        )
        if oil_excl < 2.0 and strip_ratio > 0.30:
            # Label covers nearly everything; side strip is unreliable.
            # Estimate conservatively as low-fill.
            low_ratio = _volume_to_fill_ratio(250)
            oil_y = int(cap_bot + bottle_height * (1 - low_ratio))
            return {
                "has_oil": True,
                "oil_top_y": oil_y,
                "fill_ratio": low_ratio,
                "bottle_bottom_y": bottle_bottom_y,
                "bottle_height_px": bottle_height,
                "bottle_bounds": bounds,
                "confidence": "low",
                "confidence_note": "Label covers body — estimated low fill.",
                "detection_zone": "body_oil_estimate",
            }

        if method == "f55_matched":
            conf = "medium"
            note = f"Validated via dual-threshold (matched {strip_info.get('matched_vol', '?')}ml ref)."
        elif method == "stable":
            conf = "medium" if strip_ratio > 0.30 else "low"
            note = "Stable side-strip detection."
        else:
            conf = "low"
            note = "Side-strip detection — low confidence."

        return {
            "has_oil": True,
            "oil_top_y": strip_oil_top,
            "fill_ratio": strip_ratio,
            "bottle_bottom_y": bottle_bottom_y,
            "bottle_height_px": bottle_height,
            "bottle_bounds": bounds,
            "confidence": conf,
            "confidence_note": note,
            "detection_zone": "side_strip",
        }

    # Step 4: No oil detected → empty bottle
    return _empty_result(bottle_bottom_y, bottle_height, bounds)


def _empty_result(bottle_bottom_y, bottle_height, bounds):
    return {
        "has_oil": False,
        "oil_top_y": bottle_bottom_y,
        "fill_ratio": 0.0,
        "bottle_bottom_y": bottle_bottom_y,
        "bottle_height_px": bottle_height,
        "bottle_bounds": bounds,
        "confidence": "high",
        "confidence_note": "No oil detected — bottle appears empty.",
        "detection_zone": "empty",
    }


# =====================================================================
# Volume conversion
# =====================================================================


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


# =====================================================================
# Overlay drawing
# =====================================================================


def _draw_overlay(image, cap, oil_result, volume_ml, fill_ratio):
    """Draw detection overlay on the image.

    The yellow bottle bbox and magenta bottom line are intentionally not
    drawn — the cap-ratio estimate (cap_bot + 9 * cap_h) overshoots on
    photos where the camera distance differs from the reference set, which
    on empty-bottle scans put the line well below the visible bottle. The
    bbox geometry is still returned in the API payload (`bottle_bbox`,
    `bottle_height_pixels`) so the slider/target feature can draw its own
    overlay if it wants to.
    """
    overlay = image.copy()
    bounds = oil_result["bottle_bounds"]
    bx1, by1, bx2, by2 = bounds

    if oil_result["has_oil"]:
        oil_y = oil_result["oil_top_y"]
        oil_y = max(by1, min(oil_y, by2))
        cv2.line(overlay, (bx1, oil_y), (bx2, oil_y), (0, 0, 255), 4)

    h, w = image.shape[:2]
    fs = max(0.6, min(1.2, w / 600.0))
    pct = fill_ratio * 100
    text = f"{volume_ml:.0f}ml ({pct:.1f}%)"
    cv2.putText(overlay, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2, cv2.LINE_AA)

    conf = oil_result["confidence"]
    zone = oil_result.get("detection_zone", "")
    cv2.putText(overlay, f"{conf} [{zone}]", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (0, 200, 0), 2, cv2.LINE_AA)

    return overlay


# =====================================================================
# Main processing pipeline
# =====================================================================


def process_bottle_image(image_path: str, bottle_spec):
    """Process a bottle image and detect oil level."""
    start_time = time.time()

    image = _load_image(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]

    cap = _detect_cap(hsv, image.shape)
    if cap is None:
        raise ProcessingError(
            "Could not detect bottle cap. Ensure the red cap is visible "
            "in the upper portion of the image."
        )

    oil_result = _detect_oil_level(image, hsv, cap)
    fill_ratio = oil_result["fill_ratio"]

    remaining_ml = _fill_ratio_to_ml(fill_ratio)
    total_ml = float(bottle_spec.total_volume_liters) * 1000
    consumed_ml = max(0.0, total_ml - remaining_ml)

    remaining_liters = remaining_ml / 1000.0
    consumed_liters = consumed_ml / 1000.0

    cup_ratio = float(bottle_spec.cup_conversion_ratio)
    remaining_cups = remaining_liters / cup_ratio if cup_ratio > 0 else 0
    consumed_cups = consumed_liters / cup_ratio if cup_ratio > 0 else 0

    confidence_label = oil_result["confidence"]
    confidence_map = {"high": 0.92, "medium": 0.75, "low": 0.50}
    confidence_score = confidence_map.get(confidence_label, 0.5)

    body_h = oil_result["bottle_height_px"]
    oil_height_px = body_h * fill_ratio

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
        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), 2, cv2.LINE_AA,
    )

    target_dir = Path(settings.MEDIA_ROOT) / "scans" / "targets"
    _ensure_dir(target_dir)
    fname = f"target_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(target_dir / fname), overlay)
    return f"scans/targets/{fname}"
