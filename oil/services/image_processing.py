"""
Oil level detection via reference image comparison.

Pipeline:
1. Detect bottle using HSV color detection (oil yellow + label green + cap red)
2. Crop bottle tightly, remove background
3. Normalize (resize + lighting equalization)
4. Compare against cached reference images at known oil levels
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

REFERENCE_LEVELS = [
    ("level_100.jpg", 100), ("level_095.jpg", 95), ("level_090.jpg", 90),
    ("level_086.jpg", 86), ("level_081.jpg", 81), ("level_076.jpg", 76),
    ("level_071.jpg", 71), ("level_067.jpg", 67), ("level_062.jpg", 62),
    ("level_057.jpg", 57), ("level_052.jpg", 52), ("level_048.jpg", 48),
    ("level_043.jpg", 43), ("level_038.jpg", 38), ("level_033.jpg", 33),
    ("level_029.jpg", 29), ("level_024.jpg", 24), ("level_019.jpg", 19),
    ("level_014.jpg", 14), ("level_010.jpg", 10),
]

REFERENCE_DIR = Path(settings.MEDIA_ROOT) / "reference_levels"
CACHE_DIR = Path(settings.MEDIA_ROOT) / "reference_cached"
STD_SIZE = (200, 500)  # width, height


# =====================================================================
# STEP 1: Detect and crop the bottle
# =====================================================================

def _detect_bottle(image):
    """Detect the Afia bottle using HSV color detection.

    The bottle has distinctive colors:
    - Golden/yellow oil (hue 10-40)
    - Green label (hue 35-85)
    - Red cap/handle (hue 0-10, 160-180)

    These colors together form the bottle region.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect bottle-specific colors
    oil = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([40, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 35, 25]), np.array([85, 255, 255]))
    red1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))

    bottle_mask = oil | green | red1 | red2

    # Aggressive morphology to connect bottle parts
    kernel = np.ones((15, 15), np.uint8)
    bottle_mask = cv2.morphologyEx(bottle_mask, cv2.MORPH_CLOSE, kernel, iterations=6)
    bottle_mask = cv2.morphologyEx(bottle_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours and pick the best bottle candidate
    contours, _ = cv2.findContours(bottle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ProcessingError("No bottle detected in image")

    # Filter: must be tall (aspect > 1.0) and significant size (> 3% of image)
    candidates = []
    img_area = h * w
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw == 0 or ch == 0:
            continue
        aspect = ch / cw
        area_pct = (cw * ch) / img_area
        center_x = (x + cw / 2) / w  # 0.5 = centered

        if aspect > 0.8 and area_pct > 0.03:
            # Score: prefer tall + centered + large
            score = aspect * 0.4 + (1 - abs(center_x - 0.5)) * 0.3 + area_pct * 0.3
            candidates.append((c, x, y, cw, ch, score, aspect, area_pct))

    if not candidates:
        # Fallback: merge all colored regions
        all_pts = np.vstack(contours)
        x, y, cw, ch = cv2.boundingRect(all_pts)
        logger.warning("No good bottle candidate, using merged bounding box")
    else:
        candidates.sort(key=lambda c: c[5], reverse=True)
        _, x, y, cw, ch, score, aspect, area_pct = candidates[0]
        logger.info("Bottle detected: (%d,%d) %dx%d aspect=%.2f area=%.1f%%",
                     x, y, cw, ch, aspect, area_pct * 100)

    # Expand bounding box slightly to include full bottle
    pad_x = int(cw * 0.08)
    pad_y = int(ch * 0.08)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    cw = min(w - x, cw + 2 * pad_x)
    ch = min(h - y, ch + 2 * pad_y)

    return x, y, cw, ch


def _crop_bottle(image):
    """Detect and crop the bottle from the image."""
    x, y, w, h = _detect_bottle(image)
    return image[y:y+h, x:x+w], (x, y, w, h)


# =====================================================================
# STEP 2: Normalize the cropped bottle
# =====================================================================

def _normalize(image):
    """Normalize lighting and resize to standard size."""
    # CLAHE for lighting
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Resize
    return cv2.resize(normalized, STD_SIZE, interpolation=cv2.INTER_AREA)


# =====================================================================
# STEP 3: Compare two normalized bottle images
# =====================================================================

def _compare(img1, img2):
    """Compare two normalized bottle images. Returns similarity 0-1."""
    h, w = STD_SIZE[1], STD_SIZE[0]

    # A: Row-by-row brightness profile (captures oil level position)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(float)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(float)
    p1 = g1.mean(axis=1)
    p2 = g2.mean(axis=1)
    # Normalize
    p1n = (p1 - p1.mean()) / max(1, p1.std())
    p2n = (p2 - p2.mean()) / max(1, p2.std())
    if p1n.std() > 0.01 and p2n.std() > 0.01:
        bright_corr = float(np.corrcoef(p1n, p2n)[0, 1])
    else:
        bright_corr = 0.0

    # B: Upper half MSE (most discriminating region)
    upper1 = img1[:h // 2]
    upper2 = img2[:h // 2]
    mse = float(np.mean((upper1.astype(float) - upper2.astype(float)) ** 2))
    upper_sim = max(0.0, 1.0 - mse / 6000.0)

    # C: HSV histogram on body (middle 60%)
    b1 = img1[int(h * 0.2):int(h * 0.8)]
    b2 = img2[int(h * 0.2):int(h * 0.8)]
    hsv1 = cv2.cvtColor(b1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(b2, cv2.COLOR_BGR2HSV)
    hh1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hh2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hh1, hh1)
    cv2.normalize(hh2, hh2)
    hsv_score = float(cv2.compareHist(hh1, hh2, cv2.HISTCMP_CORREL))

    # D: Full histogram
    hist_scores = []
    for ch in range(3):
        h1 = cv2.calcHist([img1], [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([img2], [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        hist_scores.append(float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)))
    hist_score = float(np.mean(hist_scores))

    combined = (
        upper_sim * 0.35 +
        max(0, bright_corr) * 0.30 +
        max(0, hsv_score) * 0.20 +
        max(0, hist_score) * 0.15
    )

    return combined


# =====================================================================
# STEP 4: Cache and load references
# =====================================================================

_CACHED_REFS = None


def _build_cache():
    """Build cached normalized bottle crops from reference images."""
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename, level in REFERENCE_LEVELS:
        npy_path = cache_dir / f"ref_{level:03d}.npy"
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
            cv2.imwrite(str(cache_dir / f"ref_{level:03d}.jpg"), normalized)
            logger.info("Cached reference: %s -> ref_%03d", filename, level)
        except ProcessingError as e:
            logger.warning("Failed to cache %s: %s", filename, e)


def _load_references():
    """Load cached reference images."""
    global _CACHED_REFS
    if _CACHED_REFS is not None:
        return _CACHED_REFS

    # Build cache if needed
    _build_cache()

    cache_dir = CACHE_DIR
    references = []

    for _, level in REFERENCE_LEVELS:
        npy_path = cache_dir / f"ref_{level:03d}.npy"
        if npy_path.exists():
            normalized = np.load(str(npy_path))
            references.append({"level": level, "normalized": normalized})

    logger.info("Loaded %d cached references", len(references))
    _CACHED_REFS = references
    return references


# =====================================================================
# STEP 5: Find best match and interpolate
# =====================================================================

def _find_match(input_norm, references):
    """Find best matching reference."""
    results = []
    for ref in references:
        score = _compare(input_norm, ref["normalized"])
        results.append({"level": ref["level"], "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results[:5]):
        logger.info("Match #%d: level=%d%% score=%.3f", i + 1, r["level"], r["score"])

    return results


def _interpolate(results):
    """Get oil level from top 2 matches."""
    best = results[0]
    second = results[1] if len(results) > 1 else best

    # Only blend with second if it's close to the best (within 15%)
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

    # Step 1-2: Detect, crop, normalize
    bottle_crop, (bx, by, bw, bh) = _crop_bottle(image)
    input_norm = _normalize(bottle_crop)

    # Step 3-4: Compare with references
    references = _load_references()
    if len(references) < 3:
        raise ProcessingError(f"Need >= 3 references, found {len(references)}")

    results = _find_match(input_norm, references)
    oil_level, confidence = _interpolate(results)
    oil_ratio = oil_level / 100.0

    # Volumes
    total = float(bottle_spec.total_volume_liters)
    remaining = oil_ratio * total
    consumed = total - remaining
    cup = float(bottle_spec.cup_conversion_ratio)
    remaining_cups = remaining / cup if cup > 0 else 0
    consumed_cups = consumed / cup if cup > 0 else 0

    logger.info("Result: oil=%d%% remain=%.2fL conf=%.2f", oil_level, remaining, confidence)

    # ---- OVERLAY ----
    img_h, img_w = original.shape[:2]
    ruler_w = max(70, int(img_w * 0.10))
    canvas = np.ones((img_h, img_w + ruler_w, 3), dtype=np.uint8) * 35
    canvas[:, ruler_w:] = original
    overlay = canvas

    # Draw bottle detection box
    cv2.rectangle(overlay, (ruler_w + bx, by), (ruler_w + bx + bw, by + bh), (0, 255, 0), 3)

    # Oil level line position (relative to bottle box)
    oil_y = by + int(bh * (1 - oil_ratio))

    # Ruler
    cv2.rectangle(overlay, (0, by), (ruler_w, by + bh), (30, 30, 30), -1)
    rf = max(0.35, min(0.55, ruler_w / 100.0))
    rt = max(1, int(rf * 2))

    cv2.putText(overlay, "FULL", (2, by - 5), cv2.FONT_HERSHEY_SIMPLEX, rf * 0.8, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(overlay, "EMPTY", (2, by + bh + int(15 * rf * 2)), cv2.FONT_HERSHEY_SIMPLEX, rf * 0.8, (0, 0, 200), 1, cv2.LINE_AA)

    for pct in range(0, 101, 5):
        y = by + bh - int(bh * pct / 100)
        if pct % 20 == 0:
            cv2.line(overlay, (2, y), (ruler_w - 2, y), (255, 255, 255), 2)
            cv2.putText(overlay, f"{pct}%", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, rf, (255, 255, 255), rt, cv2.LINE_AA)
        elif pct % 10 == 0:
            cv2.line(overlay, (12, y), (ruler_w - 2, y), (180, 180, 180), 1)

    # Green fill on ruler
    fill_ov = overlay.copy()
    cv2.rectangle(fill_ov, (0, oil_y), (ruler_w, by + bh), (0, 160, 0), -1)
    cv2.addWeighted(fill_ov, 0.25, overlay, 0.75, 0, overlay)

    # Oil marker
    cv2.putText(overlay, f"{oil_level}%", (4, oil_y + int(18 * rf)), cv2.FONT_HERSHEY_SIMPLEX, rf * 1.2, (0, 0, 255), rt + 1, cv2.LINE_AA)

    # Red line across
    cv2.line(overlay, (0, oil_y), (ruler_w + img_w, oil_y), (0, 0, 255), 3)

    # Green fill on image
    oil_fill = overlay.copy()
    cv2.rectangle(oil_fill, (ruler_w + bx, oil_y), (ruler_w + bx + bw, by + bh), (0, 180, 0), -1)
    cv2.addWeighted(oil_fill, 0.12, overlay, 0.88, 0, overlay)

    # Labels
    fs = max(0.5, min(1.5, img_w / 400.0))
    th = max(1, int(fs * 2))
    rx = ruler_w + 10

    cv2.putText(overlay, f"Oil: {oil_level}% ({remaining:.2f}L)", (rx, max(25, oil_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th, cv2.LINE_AA)

    badge_col = (0, 200, 0) if confidence >= 0.75 else (0, 180, 255) if confidence >= 0.55 else (0, 0, 255)
    badge_txt = "HIGH" if confidence >= 0.75 else "MED" if confidence >= 0.55 else "LOW"
    cv2.putText(overlay, f"Confidence: {confidence:.0%} ({badge_txt})", (rx, 30),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, badge_col, th, cv2.LINE_AA)
    cv2.putText(overlay, f"Remaining: {remaining_cups:.1f} cups ({remaining:.2f}L)", (rx, 30 + int(35 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (200, 200, 0), th, cv2.LINE_AA)

    # Save
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
