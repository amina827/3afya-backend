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

# Calibrated 1.5L Afia bottle references (200ml per cup, 7.5 cups total)
REFERENCE_LEVELS = [
    ("level_100.png", 100),  # 1500ml - full
    ("level_087.png", 87),   # 1300ml - 6.5 cups
    ("level_073.png", 73),   # 1100ml - 5.5 cups
    ("level_060.png", 60),   # 900ml  - 4.5 cups
    ("level_047.png", 47),   # 700ml  - 3.5 cups
    ("level_033.png", 33),   # 500ml  - 2.5 cups
    ("level_020.png", 20),   # 300ml  - 1.5 cups
    ("level_007.png", 7),    # 100ml  - 0.5 cups
]

REFERENCE_DIR = Path(settings.MEDIA_ROOT) / "reference_levels"
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

def _compare(img1, img2_ref):
    """Compare input image against a reference using multiple metrics."""
    h, w = STD_SIZE[1], STD_SIZE[0]

    # Align input to reference first
    aligned = _align_images(img1, img2_ref)

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

    # --- C: Golden oil amount (total oil visible) ---
    def golden_amount(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([40, 255, 255]))
        return mask.sum() / (mask.shape[0] * mask.shape[1] * 255)

    ga1 = golden_amount(aligned)
    ga2 = golden_amount(img2_ref)
    golden_sim = max(0.0, 1.0 - abs(ga1 - ga2) * 5.0)

    # --- D: Golden vertical profile (WHERE is oil) ---
    def golden_profile(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([40, 255, 255]))
        profile = mask.mean(axis=1) / 255.0
        return np.convolve(profile, np.ones(15) / 15, mode="same")

    gp1 = golden_profile(aligned)
    gp2 = golden_profile(img2_ref)
    min_len = min(len(gp1), len(gp2))
    gp_diff = float(np.mean(np.abs(gp1[:min_len] - gp2[:min_len])))
    gp_sim = max(0.0, 1.0 - gp_diff * 4.0)

    # --- E: HSV histogram on body ---
    b1 = aligned[int(h * 0.2):int(h * 0.8)]
    b2 = img2_ref[int(h * 0.2):int(h * 0.8)]
    hh1 = cv2.calcHist([cv2.cvtColor(b1, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hh2 = cv2.calcHist([cv2.cvtColor(b2, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
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
    if d1 is not None and d2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        good = [m for m in matches if m.distance < 50]
        orb_score = min(1.0, len(good) / 30.0)

    # --- Combined score (ML-style weights) ---
    combined = (
        golden_sim * 0.20 +
        gp_sim * 0.20 +
        upper_sim * 0.20 +
        max(0, bright_corr) * 0.15 +
        max(0, hsv_score) * 0.15 +
        orb_score * 0.10
    )

    return combined


# =====================================================================
# Cache and load references
# =====================================================================

_CACHED_REFS = None


def _build_cache():
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename, level in REFERENCE_LEVELS:
        stem = Path(filename).stem
        npy_path = cache_dir / f"{stem}.npy"
        if npy_path.exists():
            continue

        img_path = REFERENCE_DIR / filename
        if not img_path.exists():
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

    _build_cache()

    references = []
    for filename, level in REFERENCE_LEVELS:
        stem = Path(filename).stem
        npy_path = CACHE_DIR / f"{stem}.npy"
        if npy_path.exists():
            normalized = np.load(str(npy_path))
            references.append({"level": level, "normalized": normalized})

    logger.info("Loaded %d references", len(references))
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
    best = results[0]
    second = results[1] if len(results) > 1 else best

    if abs(best["level"] - second["level"]) <= 15:
        level = best["level"] * 0.75 + second["level"] * 0.25
    else:
        level = best["level"]

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

    references = _load_references()
    if len(references) < 3:
        raise ProcessingError(f"Need >= 3 references, found {len(references)}")

    results = _find_match(input_norm, references)
    oil_level, confidence = _interpolate(results)
    oil_ratio = oil_level / 100.0

    total = float(bottle_spec.total_volume_liters)
    remaining = oil_ratio * total
    consumed = total - remaining
    cup = float(bottle_spec.cup_conversion_ratio)
    remaining_cups = remaining / cup if cup > 0 else 0
    consumed_cups = consumed / cup if cup > 0 else 0

    logger.info("Result: oil=%d%% remain=%.2fL conf=%.2f", oil_level, remaining, confidence)

    # ---- MINIMAL OVERLAY: Just the red oil line + percentage label ----
    img_h, img_w = original.shape[:2]
    overlay = original.copy()

    oil_y = by + int(bh * (1 - oil_ratio))

    # Red oil level line across the entire image
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
