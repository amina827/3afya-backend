"""
Advanced oil level detection pipeline.

Pipeline:
    1. Load & preprocess (reflection removal via LAB equalization)
    2. Detect bottle contour (filtered by shape + center proximity)
    3. Perspective correction (straighten tilted bottles)
    4. Crop precise ROI (skip cap 15%, base 15%, sides 10%)
    5. HSV color segmentation (detect oil mask)
    6. Edge detection on ROI
    7. Fuse HSV mask + edges for robust detection
    8. Horizontal projection to find oil surface line
    9. Side-strip intensity analysis as validation/fallback
    10. Calculate fill %, volumes, cups
    11. Draw annotated overlay
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
# 1. PREPROCESSING - Remove reflections & normalize lighting
# =====================================================================

def _remove_reflections(image):
    """Remove glass reflections using LAB color space equalization."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for adaptive contrast (better than simple equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    logger.info("Reflections removed via LAB/CLAHE")
    return result


# =====================================================================
# 2. BOTTLE DETECTION - Find and isolate the bottle
# =====================================================================

def _find_bottle_contour(image):
    """Find the bottle contour with strict shape filtering."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 25, 100)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ProcessingError("No contours found")

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue

        aspect_ratio = h / float(w)
        area_ratio = area / img_area

        # Bottle must be: tall, not tiny, not the whole image
        if aspect_ratio > 1.2 and 0.05 < area_ratio < 0.85:
            center_x = x + w / 2
            center_dist = abs(center_x - img_w / 2) / img_w
            # Score: bigger + more centered = better
            score = area * (1 - center_dist * 0.5)
            candidates.append((c, score, area_ratio, aspect_ratio))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        logger.info("Bottle found: area=%.1f%%, aspect=%.2f", best[2] * 100, best[3])
        return best[0]

    logger.warning("No bottle-shaped contour, using largest")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


# =====================================================================
# 3. PERSPECTIVE CORRECTION - Straighten tilted bottles
# =====================================================================

def _correct_perspective(image, contour):
    """Straighten the bottle if it's tilted."""
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct if tilt is significant (>2 degrees) but not extreme
    if abs(angle) < 2.0 or abs(angle) > 30.0:
        return image, contour

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Re-detect contour in corrected image
    try:
        new_contour = _find_bottle_contour(corrected)
        logger.info("Perspective corrected: angle=%.1f°", angle)
        return corrected, new_contour
    except ProcessingError:
        return image, contour


# =====================================================================
# 4. ROI EXTRACTION - Precise crop of bottle body
# =====================================================================

def _get_bottle_roi(image, contour):
    """Crop the bottle body, excluding cap, base, and label edges."""
    img_h, img_w = image.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)

    # If contour is the whole image, apply margins
    if (w * h) / (img_w * img_h) > 0.85:
        margin_x = int(img_w * 0.12)
        margin_y = int(img_h * 0.10)
        x, y = margin_x, margin_y
        w = img_w - 2 * margin_x
        h = img_h - 2 * margin_y
        logger.warning("Contour too large, applying margins")

    # Trim: top 15% (cap/lid), bottom 15% (base/shadow), sides 10% (label edges)
    cap_trim = int(h * 0.15)
    base_trim = int(h * 0.15)
    side_trim = int(w * 0.10)

    roi_x = x + side_trim
    roi_y = y + cap_trim
    roi_w = w - 2 * side_trim
    roi_h = h - cap_trim - base_trim

    # Clamp to image bounds
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_w = min(roi_w, img_w - roi_x)
    roi_h = min(roi_h, img_h - roi_y)

    if roi_w < 10 or roi_h < 10:
        raise ProcessingError("ROI too small after cropping")

    return roi_x, roi_y, roi_w, roi_h


# =====================================================================
# 5. HSV OIL DETECTION - Color-based oil mask
# =====================================================================

# HSV ranges for common cooking oils
OIL_HSV_RANGES = [
    # Yellow oils (corn, sunflower, canola, Afia)
    (np.array([15, 30, 50]), np.array([45, 255, 255])),
    # Golden/amber oils (olive, sesame)
    (np.array([10, 20, 40]), np.array([25, 255, 220])),
    # Light/pale oils
    (np.array([20, 10, 80]), np.array([50, 180, 255])),
]


def _detect_oil_hsv(roi_bgr):
    """Detect oil pixels using HSV color segmentation."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in OIL_HSV_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    coverage = combined_mask.sum() / (combined_mask.shape[0] * combined_mask.shape[1] * 255)
    logger.info("HSV oil coverage: %.1f%%", coverage * 100)
    return combined_mask


# =====================================================================
# 6. EDGE DETECTION on ROI
# =====================================================================

def _detect_edges_roi(roi_bgr):
    """Get edge map within the ROI."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return edges


# =====================================================================
# 7. FUSION - Combine HSV + edges + intensity for robust detection
# =====================================================================

def _horizontal_projection(mask):
    """Calculate horizontal projection (sum of white pixels per row)."""
    return mask.sum(axis=1).astype(np.float64)


def _find_oil_level(roi_bgr):
    """Find oil level using the full advanced pipeline.

    Combines 3 signals:
    A) HSV horizontal projection (where is oil-colored area?)
    B) Side-strip intensity (dark = oil, bright = empty)
    C) Side-strip saturation (high sat = oil)

    The oil surface is where all signals agree on a transition.
    """
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        raise ProcessingError("ROI too small for analysis")

    # --- Signal A: HSV oil mask horizontal projection ---
    oil_mask = _detect_oil_hsv(roi_bgr)
    hsv_projection = _horizontal_projection(oil_mask)
    max_proj = hsv_projection.max()
    if max_proj > 0:
        hsv_projection = hsv_projection / max_proj
    else:
        hsv_projection = np.zeros(h)

    # --- Signal B: Side-strip intensity ---
    strip_w = max(8, int(w * 0.12))
    left_gray = cv2.cvtColor(roi_bgr[:, :strip_w], cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(roi_bgr[:, w - strip_w:], cv2.COLOR_BGR2GRAY)
    intensity = (left_gray.mean(axis=1) + right_gray.mean(axis=1)) / 2.0

    # --- Signal C: Side-strip saturation ---
    hsv_full = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat_channel = hsv_full[:, :, 1]
    left_sat = sat_channel[:, :strip_w].mean(axis=1)
    right_sat = sat_channel[:, w - strip_w:].mean(axis=1)
    saturation = (left_sat + right_sat) / 2.0

    # --- Smooth all signals ---
    ks = max(5, h // 12)
    if ks % 2 == 0:
        ks += 1
    kern = np.ones(ks) / ks

    smooth_hsv = np.convolve(hsv_projection, kern, mode="same")
    smooth_int = np.convolve(intensity, kern, mode="same")
    smooth_sat = np.convolve(saturation, kern, mode="same")

    # --- Normalize to 0-1 ---
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_hsv = normalize(smooth_hsv)
    norm_int = normalize(smooth_int)
    norm_sat = normalize(smooth_sat)

    # --- Combined score per row ---
    # High score = likely oil:
    #   - high HSV projection (oil color present)
    #   - low intensity (oil is darker than empty glass)
    #   - high saturation (oil is more saturated)
    oil_score = norm_hsv * 0.4 + (1 - norm_int) * 0.35 + norm_sat * 0.25

    # --- Find transition: search for biggest drop in oil_score going upward ---
    gradient = np.diff(oil_score)
    smooth_grad = np.convolve(gradient, kern, mode="same")

    # Search in the middle 90% to skip edge artifacts
    margin = int(h * 0.05)
    search = smooth_grad[margin:h - margin]
    if search.size == 0:
        return h // 2, 0.40, oil_mask

    # The oil surface = where oil_score drops most sharply (going from bottom to top)
    # = biggest negative gradient going top-to-bottom (air→oil = score goes up)
    # Actually: oil_score is HIGH in oil region, LOW in air region
    # So going top→bottom, we look for the biggest POSITIVE gradient (air→oil)
    max_idx = int(np.argmax(search))
    oil_line_y = margin + max_idx

    # --- Confidence calculation ---
    grad_strength = abs(search[max_idx])
    max_grad = abs(search).max() if abs(search).max() > 0 else 1
    relative = grad_strength / max_grad

    # Check agreement between above/below the line
    above_score = oil_score[:oil_line_y].mean() if oil_line_y > 0 else 0.5
    below_score = oil_score[oil_line_y:].mean() if oil_line_y < h else 0.5
    separation = below_score - above_score  # Should be positive (oil below > air above)

    confidence = 0.50
    confidence += relative * 0.25          # Strong gradient = more confident
    confidence += max(0, separation) * 0.20  # Good separation = more confident

    # HSV coverage bonus
    hsv_coverage = oil_mask.sum() / (h * w * 255) if h * w > 0 else 0
    if 0.1 < hsv_coverage < 0.9:
        confidence += 0.05  # Reasonable HSV coverage

    confidence = max(0.30, min(0.96, confidence))

    logger.info(
        "Oil level: y=%d/%d (fill=%.1f%%), above_score=%.2f, below_score=%.2f, "
        "separation=%.2f, grad=%.3f, confidence=%.2f",
        oil_line_y, h, (1 - oil_line_y / h) * 100,
        above_score, below_score, separation, grad_strength, confidence,
    )

    return oil_line_y, confidence, oil_mask


# =====================================================================
# MAIN PROCESSING
# =====================================================================

def process_bottle_image(image_path: str, bottle_spec):
    start = time.time()
    logger.info("Processing: %s (bottle: %s)", image_path, bottle_spec)

    image = _load_image(image_path)
    original = image.copy()

    # Step 1: Remove reflections
    image = _remove_reflections(image)

    # Step 2: Find bottle
    contour = _find_bottle_contour(image)

    # Step 3: Perspective correction
    image, contour = _correct_perspective(image, contour)

    # Step 4: Extract ROI
    roi_x, roi_y, roi_w, roi_h = _get_bottle_roi(image, contour)
    bottle_height_pixels = float(roi_h)
    roi_bgr = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Steps 5-8: Detect oil level (HSV + edges + intensity + projection)
    oil_line_y, confidence, oil_mask = _find_oil_level(roi_bgr)

    # Calculate measurements
    oil_height_pixels = float(roi_h - oil_line_y)
    oil_ratio = max(0.0, min(1.0, oil_height_pixels / bottle_height_pixels))

    total_volume = float(bottle_spec.total_volume_liters)
    remaining_volume_liters = oil_ratio * total_volume
    consumed_volume_liters = total_volume - remaining_volume_liters

    cup_ratio = float(bottle_spec.cup_conversion_ratio)
    remaining_cups = remaining_volume_liters / cup_ratio if cup_ratio > 0 else 0.0
    consumed_cups = consumed_volume_liters / cup_ratio if cup_ratio > 0 else 0.0

    logger.info(
        "Result: oil=%.1f%%, remaining=%.2fL (%.1f cups), consumed=%.2fL (%.1f cups)",
        oil_ratio * 100, remaining_volume_liters, remaining_cups,
        consumed_volume_liters, consumed_cups,
    )

    # ---- DRAW OVERLAY on ORIGINAL image ----
    overlay = original.copy()

    # Green contour around bottle
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)

    # Cyan ROI box
    cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)

    # RED oil level line
    oil_y_abs = roi_y + oil_line_y
    cv2.line(overlay, (roi_x - 15, oil_y_abs), (roi_x + roi_w + 15, oil_y_abs), (0, 0, 255), 3)

    # Semi-transparent oil region fill (green tint below oil line)
    oil_region_overlay = overlay.copy()
    cv2.rectangle(oil_region_overlay, (roi_x, oil_y_abs), (roi_x + roi_w, roi_y + roi_h), (0, 180, 0), -1)
    cv2.addWeighted(oil_region_overlay, 0.15, overlay, 0.85, 0, overlay)

    # Font scaling based on image size
    font_scale = max(0.5, min(1.5, roi_w / 200.0))
    thickness = max(1, int(font_scale * 2))

    # Oil level label
    label = f"Oil: {oil_ratio * 100:.0f}% ({remaining_volume_liters:.2f}L)"
    cv2.putText(
        overlay, label,
        (roi_x, max(25, oil_y_abs - 15)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA,
    )

    # Confidence badge
    if confidence >= 0.75:
        badge_color, badge_text = (0, 200, 0), "HIGH"
    elif confidence >= 0.55:
        badge_color, badge_text = (0, 180, 255), "MED"
    else:
        badge_color, badge_text = (0, 0, 255), "LOW"
    cv2.putText(
        overlay, f"Confidence: {confidence:.0%} ({badge_text})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, badge_color, thickness, cv2.LINE_AA,
    )

    # Remaining cups label
    cv2.putText(
        overlay, f"Remaining: {remaining_cups:.1f} cups",
        (10, 30 + int(35 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (200, 200, 0), thickness, cv2.LINE_AA,
    )

    # Liter scale on right side
    liter_step = float(bottle_spec.cup_conversion_ratio)
    if liter_step <= 0:
        liter_step = 0.25
    steps = int(total_volume / liter_step)
    marker_font = max(0.3, min(0.5, roi_w / 350.0))
    for i in range(steps + 1):
        liters = i * liter_step
        ratio = liters / total_volume if total_volume else 0
        marker_y = roi_y + int(roi_h * (1 - ratio))
        cv2.line(overlay, (roi_x + roi_w + 5, marker_y), (roi_x + roi_w + 20, marker_y), (255, 100, 0), 1)
        if i % 4 == 0:  # Label every 1L (every 4 cups at 0.25L/cup)
            cv2.putText(
                overlay, f"{liters:.1f}L",
                (roi_x + roi_w + 25, marker_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, marker_font, (255, 100, 0), 1, cv2.LINE_AA,
            )

    # Save
    processed_dir = Path(settings.MEDIA_ROOT) / "scans" / "processed"
    _ensure_dir(processed_dir)
    filename = f"processed_{uuid.uuid4().hex}.jpg"
    output_path = processed_dir / filename
    cv2.imwrite(str(output_path), overlay)

    processing_time_ms = int((time.time() - start) * 1000)
    logger.info("Done in %dms", processing_time_ms)

    return {
        "processed_path": f"scans/processed/{filename}",
        "oil_height_pixels": oil_height_pixels,
        "bottle_height_pixels": bottle_height_pixels,
        "oil_ratio": oil_ratio,
        "remaining_volume_liters": remaining_volume_liters,
        "consumed_volume_liters": consumed_volume_liters,
        "remaining_cups": remaining_cups,
        "consumed_cups": consumed_cups,
        "confidence_score": round(confidence, 2),
        "processing_time_ms": processing_time_ms,
    }


# =====================================================================
# TARGET OVERLAY
# =====================================================================

def render_target_overlay(image_path: str, bottle_spec, target_cups: float):
    image = _load_image(image_path)
    image = _remove_reflections(image)

    contour = _find_bottle_contour(image)
    image, contour = _correct_perspective(image, contour)
    roi_x, roi_y, roi_w, roi_h = _get_bottle_roi(image, contour)

    target_liters = float(target_cups) * float(bottle_spec.cup_conversion_ratio)
    total_volume = float(bottle_spec.total_volume_liters)
    target_ratio = target_liters / total_volume if total_volume else 0
    target_ratio = max(0.0, min(1.0, target_ratio))
    target_y = roi_y + int(roi_h * (1 - target_ratio))

    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
    cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)

    # Orange target line
    cv2.line(overlay, (roi_x - 15, target_y), (roi_x + roi_w + 15, target_y), (0, 165, 255), 3)

    # Semi-transparent target zone
    target_overlay = overlay.copy()
    cv2.rectangle(target_overlay, (roi_x, target_y), (roi_x + roi_w, roi_y + roi_h), (0, 165, 255), -1)
    cv2.addWeighted(target_overlay, 0.12, overlay, 0.88, 0, overlay)

    font_scale = max(0.5, min(1.0, roi_w / 200.0))
    thickness = max(1, int(font_scale * 2))
    cv2.putText(
        overlay, f"Target: {target_cups:.1f} cups ({target_liters:.2f}L)",
        (roi_x, max(25, target_y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA,
    )

    target_dir = Path(settings.MEDIA_ROOT) / "scans" / "targets"
    _ensure_dir(target_dir)
    filename = f"target_{uuid.uuid4().hex}.jpg"
    output_path = target_dir / filename
    cv2.imwrite(str(output_path), overlay)

    return f"scans/targets/{filename}"
