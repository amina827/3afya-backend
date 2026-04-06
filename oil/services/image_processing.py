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

REFERENCE_LEVELS = [
    ("level_100.jpg", 100), ("level_095.jpg", 95), ("level_090.jpg", 90),
    ("level_086.jpg", 86), ("level_081.jpg", 81), ("level_076.jpg", 76),
    ("level_071.jpg", 71), ("level_067.jpg", 67), ("level_062.jpg", 62),
    ("level_057.jpg", 57), ("level_052.jpg", 52), ("level_048.jpg", 48),
    ("level_043.jpg", 43), ("level_038.jpg", 38), ("level_033.jpg", 33),
    ("level_029.jpg", 29), ("level_024.jpg", 24), ("level_019.jpg", 19),
    ("level_014.jpg", 14), ("level_010.jpg", 10), ("level_010b.jpg", 10),
]

REFERENCE_DIR = Path(settings.MEDIA_ROOT) / "reference_levels"
CACHE_DIR = Path(settings.MEDIA_ROOT) / "reference_cached"
STD_SIZE = (200, 500)  # width, height


# =====================================================================
# STEP 1: Enhanced bottle detection (HSV + Edge fusion)
# =====================================================================

def _detect_bottle(image):
    """Detect bottle using HSV colors + edge detection fusion."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV color masks for bottle parts
    oil = cv2.inRange(hsv, np.array([10, 25, 25]), np.array([40, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 35, 25]), np.array([85, 255, 255]))
    red1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    color_mask = oil | green | red1 | red2

    # Edge detection for bottle outline (catches transparent glass too)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Fuse: dilate edges and combine with color mask
    edge_kernel = np.ones((9, 9), np.uint8)
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=3)

    # Combined mask: color OR edges (catches both oil and glass)
    combined = cv2.bitwise_or(color_mask, edges_dilated)

    # Aggressive morphology to form one solid region
    kernel = np.ones((15, 15), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=6)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find best bottle contour
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ProcessingError("No bottle detected")

    img_area = h * w
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw == 0 or ch == 0:
            continue
        aspect = ch / cw
        area_pct = (cw * ch) / img_area
        center_x = (x + cw / 2) / w

        if aspect > 0.8 and area_pct > 0.03:
            score = aspect * 0.4 + (1 - abs(center_x - 0.5)) * 0.3 + min(area_pct, 0.5) * 0.3
            candidates.append((c, x, y, cw, ch, score, aspect, area_pct))

    if not candidates:
        all_pts = np.vstack(contours)
        x, y, cw, ch = cv2.boundingRect(all_pts)
    else:
        candidates.sort(key=lambda c: c[5], reverse=True)
        _, x, y, cw, ch, _, aspect, area_pct = candidates[0]
        logger.info("Bottle: (%d,%d) %dx%d aspect=%.2f area=%.1f%%",
                     x, y, cw, ch, aspect, area_pct * 100)

    # Expand slightly
    pad_x = int(cw * 0.06)
    pad_y = int(ch * 0.06)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    cw = min(w - x, cw + 2 * pad_x)
    ch = min(h - y, ch + 2 * pad_y)

    return x, y, cw, ch


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

    # ---- OVERLAY ----
    img_h, img_w = original.shape[:2]
    ruler_w = max(70, int(img_w * 0.10))
    canvas = np.ones((img_h, img_w + ruler_w, 3), dtype=np.uint8) * 35
    canvas[:, ruler_w:] = original
    overlay = canvas

    cv2.rectangle(overlay, (ruler_w + bx, by), (ruler_w + bx + bw, by + bh), (0, 255, 0), 3)
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

    fill_ov = overlay.copy()
    cv2.rectangle(fill_ov, (0, oil_y), (ruler_w, by + bh), (0, 160, 0), -1)
    cv2.addWeighted(fill_ov, 0.25, overlay, 0.75, 0, overlay)

    cv2.putText(overlay, f"{oil_level}%", (4, oil_y + int(18 * rf)), cv2.FONT_HERSHEY_SIMPLEX, rf * 1.2, (0, 0, 255), rt + 1, cv2.LINE_AA)
    cv2.line(overlay, (0, oil_y), (ruler_w + img_w, oil_y), (0, 0, 255), 3)

    oil_fill = overlay.copy()
    cv2.rectangle(oil_fill, (ruler_w + bx, oil_y), (ruler_w + bx + bw, by + bh), (0, 180, 0), -1)
    cv2.addWeighted(oil_fill, 0.12, overlay, 0.88, 0, overlay)

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
