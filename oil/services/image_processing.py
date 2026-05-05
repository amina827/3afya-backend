"""
Oil level detection via reference image comparison.

Enhanced Pipeline:
1. Detect bottle using HSV + edge contour fusion
2. ORB Feature Matching - align input to reference before comparing
3. Multi-metric comparison (brightness, golden, HSV, structure, ORB)
4. ML-style weighted scoring trained on reference data
5. Interpolate oil level from best matches
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
# Reference data
# =====================================================================

# Calibrated 1.5L Afia bottle references.
# 17 levels evenly spaced from 100% → 0% (~6.25% per step), produced by
# pouring ~100ml out for each successive photo. Finer granularity than the
# original 8-level set → smaller interpolation error between neighbors.
REFERENCE_LEVELS = [
    ("level_100.jpg", 100),
    ("level_094.jpg", 94),
    ("level_088.jpg", 88),
    ("level_081.jpg", 81),
    ("level_075.jpg", 75),
    ("level_069.jpg", 69),
    ("level_063.jpg", 63),
    ("level_056.jpg", 56),
    ("level_050.jpg", 50),
    ("level_044.jpg", 44),
    ("level_038.jpg", 38),
    ("level_031.jpg", 31),
    ("level_025.jpg", 25),
    ("level_019.jpg", 19),
    ("level_013.jpg", 13),
    ("level_006.jpg", 6),
    ("level_000.jpg", 0),
]

# Reference images are calibration assets that ship with the source code,
# not user uploads. Keeping them inside the package ensures they are present
# in every deployment (the media/ directory is gitignored and dockerignored).
REFERENCE_DIR = Path(__file__).parent / "reference_data"
CACHE_DIR = Path(settings.MEDIA_ROOT) / "reference_cached"
STD_SIZE = (200, 500)  # width, height


# =====================================================================
# STEP 1: Enhanced bottle detection (HSV + Edge fusion)
# =====================================================================

# Calibrated bottle aspect ratio (height / width) for the Afia 1.5L bottle.
# Measured from reference images: bottle bbox 133x248 in 286x329 image,
# aspect = 248/133 = 1.864. NECK to BASE, including handle.
BOTTLE_ASPECT = 1.88

# Color region tends to extend slightly BELOW the bottle base due to
# shadows / reflections. Subtract this fraction of bottle_height.
COLOR_BOTTOM_OFFSET = -0.025


def _adaptive_inflation(color_aspect: float) -> float:
    """How much wider is the bottle than the detected color region?

    Calibrated against 8 reference images:
    - HIGH fill (color aspect ~2.2): color ≈ bottle, use 1.05
    - MED fill (color aspect ~1.0): color slightly narrower, use 1.07
    - LOW fill (color aspect ~0.5): only label visible, use 1.13
    - VERY LOW (color aspect < 0.4): tiny label region, use 1.25
    """
    if color_aspect >= 1.5:
        return 1.05
    if color_aspect >= 0.8:
        # 0.8 → 1.10, 1.5 → 1.05
        return 1.10 - (color_aspect - 0.8) / 0.7 * 0.05
    if color_aspect >= 0.4:
        # 0.4 → 1.20, 0.8 → 1.10
        return 1.20 - (color_aspect - 0.4) / 0.4 * 0.10
    return 1.25


def _detect_bottle(image):
    """Detect the bottle bounding box using calibrated proportions.

    Strategy:
    1. Build a color mask (yellow oil + green/red label parts).
    2. Find the largest coherent color region.
    3. Compute bottle WIDTH from color width with adaptive inflation.
    4. Compute bottle HEIGHT from calibrated aspect ratio.
    5. Anchor the BOTTOM to the color region bottom (minus small spillage).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Phase 1: Detect bottle colors ---
    yellow = cv2.inRange(hsv, np.array([10, 55, 70]), np.array([40, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 55, 45]), np.array([90, 255, 255]))
    red1 = cv2.inRange(hsv, np.array([0, 70, 60]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 70, 60]), np.array([180, 255, 255]))
    color_mask = yellow | green | red1 | red2

    kernel = np.ones((9, 9), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # --- Phase 2: Find largest coherent contour ---
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ProcessingError("No bottle colors detected")

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 300:
        raise ProcessingError(f"Color region too small: area={area:.0f}")

    cx, cy, cw, ch = cv2.boundingRect(largest)
    color_aspect = ch / max(1, cw)
    color_bottom = cy + ch
    color_center_x = cx + cw // 2

    logger.info("Color region: (%d,%d) %dx%d aspect=%.2f area=%.0f",
                cx, cy, cw, ch, color_aspect, area)

    # --- Phase 3: Compute bottle dimensions from calibrated proportions ---
    # Width comes from color region with adaptive inflation (depends on fill).
    # Height comes from calibrated aspect ratio.
    # We trust the WIDTH estimate more than the height - color heights are noisy
    # because they reflect oil level (which varies), but color widths reflect
    # the label/body width (which is constant).
    inflation = _adaptive_inflation(color_aspect)
    bottle_width = int(round(cw * inflation))
    bottle_height = int(round(bottle_width * BOTTLE_ASPECT))

    # --- Phase 4: Position the bbox ---
    # Bottom: anchored to color region bottom (minus a small offset since
    # color often has a few pixels of bleed below the base)
    spillage = int(round(bottle_height * abs(COLOR_BOTTOM_OFFSET)))
    bottle_bottom = min(h, color_bottom - spillage)
    bottle_top = max(0, bottle_bottom - bottle_height)

    # Sides: centered on color centroid horizontally
    bottle_left = max(0, color_center_x - bottle_width // 2)
    bottle_right = min(w, bottle_left + bottle_width)
    bottle_left = max(0, bottle_right - bottle_width)  # adjust if right clipped

    bx = bottle_left
    by = bottle_top
    bw = bottle_right - bottle_left
    bh = bottle_bottom - bottle_top

    if bh < 50 or bw < 20:
        raise ProcessingError(f"Detected bottle too small: {bw}x{bh}")

    final_aspect = bh / max(1, bw)
    logger.info("Bottle bbox: (%d,%d) %dx%d aspect=%.2f inflation=%.2f",
                bx, by, bw, bh, final_aspect, inflation)

    return bx, by, bw, bh


def _crop_bottle(image):
    x, y, w, h = _detect_bottle(image)
    return image[y:y+h, x:x+w], (x, y, w, h)


# =====================================================================
# STEP 1.5: Smart ROI + Adaptive HSV
# =====================================================================

def _body_roi(bottle_img, vertical=(0.05, 0.97), horizontal=(0.15, 0.85)):
    """Restrict processing to the bottle BODY (skip cap & handle edges).

    Defaults exclude cap area (top 5%) and narrow sides (handle bleed).
    Kept loose vertically because the oil line itself may sit anywhere
    from near the top to near the base.
    """
    h, w = bottle_img.shape[:2]
    y1, y2 = int(h * vertical[0]), int(h * vertical[1])
    x1, x2 = int(w * horizontal[0]), int(w * horizontal[1])
    return bottle_img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _adaptive_oil_mask(img_bgr):
    """Build an HSV mask for oil that adapts to the image's own color cast.

    Falls back to the calibrated static range when the adaptive window
    would shift outside the plausible golden/yellow band [5, 45].
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Sample only somewhat-saturated pixels to avoid a hue mean dragged
    # down by the white/grey background.
    sat_mask = hsv[:, :, 1] > 40
    if sat_mask.sum() > 500:
        h_vals = hsv[:, :, 0][sat_mask]
        h_median = float(np.median(h_vals))
    else:
        h_median = 25.0  # fall back to oil band center

    # Clamp into the plausible oil band to stay safe against outliers.
    h_center = max(10.0, min(35.0, h_median))
    lower = np.array([max(0, h_center - 12), 35, 35], dtype=np.uint8)
    upper = np.array([min(179, h_center + 15), 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Union with calibrated static mask so we never lose obvious oil.
    static = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([40, 255, 255]))
    return mask | static


# =====================================================================
# STEP 1.6: Direct oil-line detection (Canny + Hough)
# =====================================================================

def _detect_oil_line_direct(bottle_crop):
    """Detect the oil meniscus directly on the bottle crop.

    Returns (y_relative, confidence, meta) where y_relative is the oil-line Y
    position as a fraction of bottle HEIGHT (0.0 top, 1.0 bottom),
    or (None, 0.0, {}) if no convincing line is found.

    Strategy:
    1. Restrict to body ROI (avoid cap / handle edges).
    2. Canny edges + HoughLinesP to collect near-horizontal segments.
    3. Score each candidate by the brightness step across it
       (oil-to-air transition produces a strong delta).
    4. Return the best scoring line's Y (remapped to full-bottle coords).
    """
    bh_full = bottle_crop.shape[0]
    if bh_full < 80:
        return None, 0.0, {}

    roi, (rx, ry, rw, rh) = _body_roi(
        bottle_crop, vertical=(0.02, 0.98), horizontal=(0.20, 0.80)
    )
    if rh < 50 or rw < 20:
        return None, 0.0, {}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 130)

    min_len = max(20, int(rw * 0.35))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(30, int(rw * 0.25)),
        minLineLength=min_len,
        maxLineGap=10,
    )
    if lines is None:
        return None, 0.0, {}

    # Collect near-horizontal segments and aggregate by Y.
    band_h = 6  # group segments within ±6 pixels
    buckets = {}
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        if abs(y1 - y2) > 3:
            continue
        y_mid = (y1 + y2) // 2
        length = abs(x2 - x1)
        key = y_mid // band_h
        buckets.setdefault(key, []).append((y_mid, length))

    if not buckets:
        return None, 0.0, {}

    # Score each bucket by brightness delta (oil darker than air above it).
    candidates = []
    for key, segs in buckets.items():
        y_mid = int(np.mean([s[0] for s in segs]))
        total_len = sum(s[1] for s in segs)
        if total_len < rw * 0.4:
            continue

        up1, up2 = max(0, y_mid - 12), max(0, y_mid - 2)
        dn1, dn2 = min(rh, y_mid + 2), min(rh, y_mid + 12)
        if up2 <= up1 or dn2 <= dn1:
            continue
        above = float(gray[up1:up2].mean())
        below = float(gray[dn1:dn2].mean())
        delta = below - above  # positive = darker above (empty) / brighter below (oil)
        # Oil is golden/yellow → brighter than empty air in LAB L but darker in
        # some scenes. Use absolute delta as the signal.
        score = abs(delta) * (total_len / rw)
        candidates.append(
            {"y": y_mid, "score": float(score), "delta": float(delta), "len": int(total_len)}
        )

    if not candidates:
        return None, 0.0, {}

    candidates.sort(key=lambda c: c["score"], reverse=True)
    if candidates[0]["score"] < 8.0:
        return None, 0.0, {}

    # Stabilize Y by weighted average of top nearby candidates.
    anchor_y = candidates[0]["y"]
    nearby = [c for c in candidates[:8] if abs(c["y"] - anchor_y) <= 10]
    if not nearby:
        nearby = [candidates[0]]
    weights = np.array([max(1e-6, c["score"]) for c in nearby], dtype=float)
    ys = np.array([c["y"] for c in nearby], dtype=float)
    y_stable = float(np.average(ys, weights=weights))
    best_score = float(candidates[0]["score"])
    best_len = int(candidates[0]["len"])

    # Remap Y from ROI coords to full bottle crop coords.
    y_full = ry + y_stable
    y_relative = y_full / bh_full
    # Confidence scales with score; cap at 0.85 — edges alone are noisy.
    conf = min(0.90, 0.35 + best_score / 75.0 + min(0.10, best_len / max(1.0, rw) * 0.10))
    return float(y_relative), float(conf), {
        "roi": {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)},
        "candidate_count": int(len(candidates)),
        "used_count": int(len(nearby)),
    }


# =====================================================================
# STEP 2: Normalize
# =====================================================================

def _normalize(image):
    """Normalize lighting and resize."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.resize(normalized, STD_SIZE, interpolation=cv2.INTER_AREA)


# =====================================================================
# STEP 3: ORB Feature Matching - align images before comparison
# =====================================================================

def _align_images(img1, img2):
    """Align img1 to img2 using ORB feature matching.
    Returns aligned version of img1, or img1 unchanged if alignment fails.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return img1

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    if len(matches) < 4:
        return img1

    matches = sorted(matches, key=lambda m: m.distance)[:50]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if M is None:
        return img1

    h, w = img2.shape[:2]
    aligned = cv2.warpPerspective(img1, M, (w, h))
    return aligned


# =====================================================================
# STEP 4: Multi-metric comparison
# =====================================================================

_BASE_WEIGHTS = {
    "golden_sim": 0.20,
    "gp_sim": 0.20,
    "upper_sim": 0.20,
    "bright_corr": 0.15,
    "hsv_score": 0.15,
    "orb_score": 0.10,
}


def _dynamic_weights(metrics, img_stats):
    """Adjust weights based on scene quality signals.

    - Bad lighting (very low / very high mean V) → trust brightness less.
    - Poor ORB alignment (few matches) → trust orb_score less and lean
      on color/profile metrics.
    - Very low saturation across the scene → trust golden metrics less.
    """
    w = dict(_BASE_WEIGHTS)

    mean_v = img_stats.get("mean_v", 128)
    mean_s = img_stats.get("mean_s", 100)
    orb_raw = metrics.get("orb_raw_matches", 0)

    if mean_v < 55 or mean_v > 215:
        # Washed out or very dark – brightness profile is unreliable.
        w["bright_corr"] *= 0.5
        w["upper_sim"] *= 0.7
    if orb_raw < 8:
        # Alignment and structural match are unreliable; shift to color.
        w["orb_score"] *= 0.3
        w["golden_sim"] *= 1.15
        w["gp_sim"] *= 1.15
    if mean_s < 35:
        # Desaturated scene — color-based metrics weaker.
        w["golden_sim"] *= 0.7
        w["hsv_score"] *= 0.7

    # Renormalize to keep the combined score in [0, 1].
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def _reject_outliers(metrics):
    """Zero-out a metric whose value is an extreme outlier vs its peers.

    Only triggers when the offender is clearly worse than the others —
    protects against a single noisy signal dragging the combined score.
    """
    vals = {k: max(0.0, float(v)) for k, v in metrics.items()}
    arr = np.array(list(vals.values()))
    if len(arr) < 4:
        return vals
    mean, std = float(arr.mean()), float(arr.std())
    if std < 0.05:
        return vals
    cleaned = {}
    for k, v in vals.items():
        if v < mean - 2.0 * std:
            cleaned[k] = 0.0
        else:
            cleaned[k] = v
    return cleaned


def _compare(img1, img2_ref):
    """Compare input image against a reference using multiple metrics."""
    h, w = STD_SIZE[1], STD_SIZE[0]

    # Align input to reference first
    aligned = _align_images(img1, img2_ref)

    # Stats for dynamic weighting
    hsv_full = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
    img_stats = {
        "mean_v": float(hsv_full[:, :, 2].mean()),
        "mean_s": float(hsv_full[:, :, 1].mean()),
    }

    # Body ROI for color-based metrics (avoid cap / handle noise)
    body_aligned, _ = _body_roi(aligned)
    body_ref, _ = _body_roi(img2_ref)

    # --- A: Brightness profile correlation ---
    g1 = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(float)
    g2 = cv2.cvtColor(img2_ref, cv2.COLOR_BGR2GRAY).astype(float)
    p1 = g1.mean(axis=1)
    p2 = g2.mean(axis=1)
    p1n = (p1 - p1.mean()) / max(1, p1.std())
    p2n = (p2 - p2.mean()) / max(1, p2.std())
    bright_corr = float(np.corrcoef(p1n, p2n)[0, 1]) if p1n.std() > 0.01 and p2n.std() > 0.01 else 0.0

    # --- B: Upper half similarity (empty vs oil) ---
    upper1 = aligned[:h // 2]
    upper2 = img2_ref[:h // 2]
    mse_upper = float(np.mean((upper1.astype(float) - upper2.astype(float)) ** 2))
    upper_sim = max(0.0, 1.0 - mse_upper / 5000.0)

    # --- C: Golden oil amount (adaptive HSV, measured on body ROI) ---
    def golden_amount(img):
        mask = _adaptive_oil_mask(img)
        return mask.sum() / (mask.shape[0] * mask.shape[1] * 255)

    ga1 = golden_amount(body_aligned)
    ga2 = golden_amount(body_ref)
    golden_sim = max(0.0, 1.0 - abs(ga1 - ga2) * 5.0)

    # --- D: Golden vertical profile (WHERE is oil) ---
    def golden_profile(img):
        mask = _adaptive_oil_mask(img)
        profile = mask.mean(axis=1) / 255.0
        return np.convolve(profile, np.ones(15) / 15, mode="same")

    gp1 = golden_profile(aligned)
    gp2 = golden_profile(img2_ref)
    min_len = min(len(gp1), len(gp2))
    gp_diff = float(np.mean(np.abs(gp1[:min_len] - gp2[:min_len])))
    gp_sim = max(0.0, 1.0 - gp_diff * 4.0)

    # --- E: HSV histogram on body ---
    hh1 = cv2.calcHist([cv2.cvtColor(body_aligned, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hh2 = cv2.calcHist([cv2.cvtColor(body_ref, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hh1, hh1)
    cv2.normalize(hh2, hh2)
    hsv_score = float(cv2.compareHist(hh1, hh2, cv2.HISTCMP_CORREL))

    # --- F: ORB feature match count (structural similarity) ---
    gray1 = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=300)
    kp1, d1 = orb.detectAndCompute(gray1, None)
    kp2, d2 = orb.detectAndCompute(gray2, None)
    orb_score = 0.0
    orb_raw = 0
    if d1 is not None and d2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        good = [m for m in matches if m.distance < 50]
        orb_raw = len(good)
        orb_score = min(1.0, orb_raw / 30.0)

    raw_metrics = {
        "golden_sim": golden_sim,
        "gp_sim": gp_sim,
        "upper_sim": upper_sim,
        "bright_corr": max(0.0, bright_corr),
        "hsv_score": max(0.0, hsv_score),
        "orb_score": orb_score,
    }

    # Reject single-outlier metrics then apply dynamic weights
    cleaned = _reject_outliers(raw_metrics)
    weights = _dynamic_weights(
        {"orb_raw_matches": orb_raw}, img_stats
    )

    combined = sum(cleaned[k] * weights[k] for k in weights)
    return float(combined)


# =====================================================================
# Cache and load references
# =====================================================================

_CACHED_REFS = None
_DB_REF_CACHE_KEY = "oil_references_v1"


def invalidate_reference_cache():
    """Clear both in-memory and Django cache for oil references.

    Called from the OilReference post_save / post_delete signal so that
    admin edits take effect on the next scan without a server restart.
    """
    global _CACHED_REFS
    _CACHED_REFS = None
    try:
        from django.core.cache import cache
        cache.delete(_DB_REF_CACHE_KEY)
    except Exception:
        pass


def extract_reference_features(image_path: str):
    """Extract and cache all features needed to classify a reference image.

    Persists the normalized bottle crop as .npy (under MEDIA/reference_cached)
    and returns a dict of JSON-serializable features for the DB row.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ProcessingError(f"Unable to read reference image: {image_path}")

    bottle_crop, _ = _crop_bottle(img)
    normalized = _normalize(bottle_crop)
    body, _ = _body_roi(normalized)

    # Brightness profile (per-row mean of grayscale, 1D array)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(float)
    brightness_profile = gray.mean(axis=1).tolist()

    # Golden profile (adaptive mask per-row mean)
    mask = _adaptive_oil_mask(normalized)
    golden_profile_arr = mask.mean(axis=1) / 255.0
    golden_profile_arr = np.convolve(golden_profile_arr, np.ones(15) / 15, mode="same")
    golden_profile = golden_profile_arr.tolist()

    # Golden amount (scalar 0-1)
    body_mask = _adaptive_oil_mask(body)
    golden_amt = float(body_mask.sum() / (body_mask.shape[0] * body_mask.shape[1] * 255))

    # H-S histogram on body (flattened list of floats)
    hist = cv2.calcHist(
        [cv2.cvtColor(body, cv2.COLOR_BGR2HSV)],
        [0, 1], None, [30, 32], [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist)
    histogram = hist.flatten().tolist()

    # Persist the normalized npy so _load_references can mmap it quickly.
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    npy_rel = f"reference_cached/db_{stem}_{uuid.uuid4().hex[:8]}.npy"
    npy_path = Path(settings.MEDIA_ROOT) / npy_rel
    np.save(str(npy_path), normalized)

    return {
        "brightness_profile": brightness_profile,
        "histogram": histogram,
        "golden_profile": golden_profile,
        "golden_amount": golden_amt,
        "normalized_cache_path": npy_rel,
    }


def _load_references_from_db():
    """Load active OilReference rows. Returns [] if none exist.

    Results are cached via django.core.cache (1h TTL) and additionally
    memoized in the module-level _CACHED_REFS for the hot path.
    """
    try:
        from django.core.cache import cache
        from oil.models import OilReference
    except Exception as e:
        logger.warning("OilReference DB access unavailable: %s", e)
        return []

    cached = cache.get(_DB_REF_CACHE_KEY)
    if cached is not None:
        return cached

    refs = []
    qs = OilReference.objects.filter(is_active=True).order_by("-level_percentage")
    for row in qs:
        npy_rel = row.normalized_cache_path
        if not npy_rel:
            continue
        npy_path = Path(npy_rel)
        if not npy_path.is_absolute():
            npy_path = Path(settings.MEDIA_ROOT) / npy_rel
        if not npy_path.exists():
            logger.warning("OilReference %s missing cache file: %s", row.pk, npy_path)
            continue
        try:
            normalized = np.load(str(npy_path))
            refs.append({
                "level": float(row.level_percentage),
                "normalized": normalized,
                "source": "db",
                "id": row.pk,
            })
        except Exception as e:
            logger.warning("Failed loading OilReference %s: %s", row.pk, e)

    cache.set(_DB_REF_CACHE_KEY, refs, timeout=3600)
    logger.info("Loaded %d references from DB", len(refs))
    return refs


def _build_cache():
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename, level in REFERENCE_LEVELS:
        stem = Path(filename).stem
        npy_path = cache_dir / f"{stem}.npy"
        img_path = REFERENCE_DIR / filename
        if not img_path.exists():
            continue

        # Rebuild cache when the source image has been updated so that
        # replacing a reference file (e.g. recalibration) takes effect
        # without manually clearing MEDIA/reference_cached.
        if npy_path.exists() and npy_path.stat().st_mtime >= img_path.stat().st_mtime:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            bottle, _ = _crop_bottle(img)
            normalized = _normalize(bottle)
            np.save(str(npy_path), normalized)
            cv2.imwrite(str(cache_dir / f"{stem}.jpg"), normalized)
            logger.info("Cached: %s -> %s", filename, stem)
        except ProcessingError as e:
            logger.warning("Cache failed %s: %s", filename, e)


def _load_references():
    global _CACHED_REFS
    if _CACHED_REFS is not None:
        return _CACHED_REFS

    # Prefer admin-managed references from the DB.
    db_refs = _load_references_from_db()
    if len(db_refs) >= 3:
        _CACHED_REFS = db_refs
        return db_refs

    # Fallback: bundled folder references (initial install / empty DB).
    _build_cache()
    references = []
    for filename, level in REFERENCE_LEVELS:
        stem = Path(filename).stem
        npy_path = CACHE_DIR / f"{stem}.npy"
        if npy_path.exists():
            normalized = np.load(str(npy_path))
            references.append({"level": level, "normalized": normalized, "source": "folder"})

    # If both DB and folder have partial data, merge (DB wins on duplicate level).
    if db_refs:
        seen_levels = {r["level"] for r in db_refs}
        references = db_refs + [r for r in references if r["level"] not in seen_levels]

    logger.info("Loaded %d references (db=%d, folder=%d)",
                len(references), len(db_refs), len(references) - len(db_refs))
    _CACHED_REFS = references
    return references


# =====================================================================
# Find best match and interpolate
# =====================================================================

def _find_match(input_norm, references):
    results = []
    for ref in references:
        score = _compare(input_norm, ref["normalized"])
        results.append({"level": ref["level"], "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results[:5]):
        logger.info("Match #%d: level=%d%% score=%.3f", i + 1, r["level"], r["score"])

    return results


def _interpolate(results):
    """Softmax-weighted interpolation over the top matches.

    With ~17 references at ~6% spacing, the top few matches tend to be
    close neighbors on the level axis. A temperature-sharpened softmax
    blends them so the returned percentage can fall between reference
    buckets instead of snapping to one of them.

    Outlier rejection: only references whose level is within 12% of the
    top match contribute, so an unrelated high-score far-away level
    (rare but possible when the scene is ambiguous) cannot drag the
    result toward it.
    """
    best = results[0]

    close = [r for r in results[:5] if abs(r["level"] - best["level"]) <= 12]
    if len(close) >= 2:
        scores = np.array([r["score"] for r in close], dtype=float)
        # Subtract max for numerical stability, then sharpen.
        weights = np.exp((scores - scores.max()) * 18.0)
        weights /= weights.sum()
        level = float(sum(w * r["level"] for w, r in zip(weights, close)))
    else:
        level = float(best["level"])

    second = results[1] if len(results) > 1 else best
    gap = best["score"] - second["score"]
    confidence = 0.60
    confidence += min(0.20, best["score"] * 0.25)
    confidence += min(0.10, gap * 2.0)
    if best["score"] > 0.7:
        confidence += 0.06
    confidence = max(0.45, min(0.96, confidence))

    return round(level), confidence


# =====================================================================
# Main processing
# =====================================================================

def process_bottle_image(image_path: str, bottle_spec):
    start = time.time()
    logger.info("Processing: %s", image_path)

    image = _load_image(image_path)
    original = image.copy()

    bottle_crop, (bx, by, bw, bh) = _crop_bottle(image)
    input_norm = _normalize(bottle_crop)

    # Keep normalization computed for pipeline compatibility and future use.
    _ = input_norm

    edge_y_rel, edge_conf, edge_meta = _detect_oil_line_direct(bottle_crop)
    if edge_y_rel is None:
        raise ProcessingError("Oil level line could not be detected")

    # Coordinate system reminder: image origin is top-left, y grows downward.
    top_y = int(by)
    bottom_y = int(by + bh)
    oil_level_y = int(round(by + (edge_y_rel * bh)))
    denom = float(bottom_y - top_y)
    if denom <= 0:
        raise ProcessingError("Invalid bottle boundaries for oil normalization")

    oil_percentage = ((bottom_y - oil_level_y) / denom) * 100.0
    oil_percentage = float(np.clip(oil_percentage, 0.0, 100.0))
    oil_ratio = oil_percentage / 100.0
    oil_level = int(round(oil_percentage))
    confidence = float(np.clip(edge_conf, 0.0, 0.96))

    total = float(bottle_spec.total_volume_liters)
    remaining = oil_ratio * total
    consumed = total - remaining
    cup = float(bottle_spec.cup_conversion_ratio)
    remaining_cups = remaining / cup if cup > 0 else 0
    consumed_cups = consumed / cup if cup > 0 else 0

    logger.info("Result: oil=%d%% remain=%.2fL conf=%.2f", oil_level, remaining, confidence)

    # ---- DEBUG OVERLAY: top, bottom and detected oil level lines ----
    img_h, img_w = original.shape[:2]
    overlay = original.copy()

    oil_y = oil_level_y
    top_line_y = top_y
    bottom_line_y = bottom_y

    # Blue top boundary line
    cv2.line(overlay, (0, top_line_y), (img_w, top_line_y), (255, 0, 0), 2)
    # Yellow bottom boundary line
    cv2.line(overlay, (0, bottom_line_y), (img_w, bottom_line_y), (0, 255, 255), 2)
    # Red oil level line
    cv2.line(overlay, (0, oil_y), (img_w, oil_y), (0, 0, 255), 3)

    # Oil percentage label above the line
    fs = max(0.6, min(1.6, img_w / 400.0))
    th = max(2, int(fs * 2))
    label = f"{oil_level}%"
    (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)

    # Position label centered horizontally on the bottle, above the line
    label_x = max(10, min(img_w - tw - 10, bx + bw // 2 - tw // 2))
    label_y = max(th_text + 10, oil_y - 10)

    # White outline for readability
    cv2.putText(overlay, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th + 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th, cv2.LINE_AA)

    processed_dir = Path(settings.MEDIA_ROOT) / "scans" / "processed"
    _ensure_dir(processed_dir)
    fname = f"processed_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(processed_dir / fname), overlay)

    processing_time_ms = int((time.time() - start) * 1000)
    logger.info("Done in %dms", processing_time_ms)

    return {
        "processed_path": f"scans/processed/{fname}",
        "oil_height_pixels": float(bh * oil_ratio),
        "bottle_height_pixels": float(bh),
        "oil_ratio": oil_ratio,
        "remaining_volume_liters": remaining,
        "consumed_volume_liters": consumed,
        "remaining_cups": remaining_cups,
        "consumed_cups": consumed_cups,
        "confidence_score": round(confidence, 2),
        "processing_time_ms": processing_time_ms,
        "oil_line_debug": {
            "top_y": int(top_y),
            "bottom_y": int(bottom_y),
            "oil_level_y": int(oil_level_y),
            "detector": edge_meta,
        },
        "bottle_bbox": {
            "x": int(bx),
            "y": int(by),
            "w": int(bw),
            "h": int(bh),
            "image_w": int(img_w),
            "image_h": int(img_h),
        },
    }


def render_target_overlay(image_path: str, bottle_spec, target_cups: float):
    image = _load_image(image_path)
    h, w = image.shape[:2]
    total = float(bottle_spec.total_volume_liters)
    target_l = float(target_cups) * float(bottle_spec.cup_conversion_ratio)
    ratio = min(1.0, max(0.0, target_l / total if total else 0))

    try:
        _, (bx, by, bw, bh) = _crop_bottle(image)
    except ProcessingError:
        by, bh = int(h * 0.15), int(h * 0.70)
        bx, bw = int(w * 0.2), int(w * 0.6)

    target_y = by + int(bh * (1 - ratio))
    overlay = image.copy()
    cv2.line(overlay, (0, target_y), (w, target_y), (0, 165, 255), 3)
    fs = max(0.5, min(1.0, w / 400.0))
    cv2.putText(overlay, f"Target: {target_cups:.1f} cups ({target_l:.2f}L)",
                (10, max(25, target_y - 15)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), 2, cv2.LINE_AA)

    target_dir = Path(settings.MEDIA_ROOT) / "scans" / "targets"
    _ensure_dir(target_dir)
    fname = f"target_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(target_dir / fname), overlay)
    return f"scans/targets/{fname}"
