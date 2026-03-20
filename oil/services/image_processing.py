import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings


class ProcessingError(Exception):
    pass


def _load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise ProcessingError("Unable to read image")
    return image


def _find_bottle_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ProcessingError("No contours found")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def _find_oil_level_line(roi_gray):
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=10)
    if lines is not None:
        horizontals = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 3:
                horizontals.append((x1, y1, x2, y2))
        if horizontals:
            # Use the longest horizontal line as oil boundary
            horizontals.sort(key=lambda l: abs(l[2] - l[0]), reverse=True)
            return horizontals[0][1], 0.85

    # Fallback: detect strongest horizontal intensity change
    row_means = roi_gray.mean(axis=1)
    diffs = np.abs(np.diff(row_means))
    if diffs.size == 0:
        raise ProcessingError("Unable to detect oil level")
    y = int(np.argmax(diffs))
    return y, 0.65


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_bottle_image(image_path: str, bottle_spec):
    start = time.time()
    image = _load_image(image_path)
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contour = _find_bottle_contour(edges)
    x, y, w, h = cv2.boundingRect(contour)
    bottle_height_pixels = float(h)

    roi_gray = gray[y : y + h, x : x + w]
    oil_line_y, line_confidence = _find_oil_level_line(roi_gray)

    oil_height_pixels = float(h - oil_line_y)
    oil_ratio = max(0.0, min(1.0, oil_height_pixels / bottle_height_pixels))

    ratio_liters_per_pixel = float(bottle_spec.height_to_volume_ratio)
    remaining_volume_liters = oil_height_pixels * ratio_liters_per_pixel
    total_volume = float(bottle_spec.total_volume_liters)
    if remaining_volume_liters > total_volume:
        remaining_volume_liters = total_volume
    consumed_volume_liters = float(total_volume - remaining_volume_liters)
    remaining_cups = float(remaining_volume_liters / float(bottle_spec.cup_conversion_ratio))
    consumed_cups = float(consumed_volume_liters / float(bottle_spec.cup_conversion_ratio))

    # Overlay drawing
    overlay = original.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    oil_y_abs = y + int(oil_line_y)
    cv2.line(overlay, (x, oil_y_abs), (x + w, oil_y_abs), (0, 0, 255), 2)

    # Mark liter and cup intervals
    liter_step = float(bottle_spec.cup_conversion_ratio)
    if liter_step <= 0:
        liter_step = 0.25
    total_liters = float(bottle_spec.total_volume_liters)
    steps = int(total_liters / liter_step)
    for i in range(steps + 1):
        liters = i * liter_step
        ratio = liters / total_liters if total_liters else 0
        marker_y = y + int(h * (1 - ratio))
        cv2.line(overlay, (x + w + 5, marker_y), (x + w + 20, marker_y), (255, 0, 0), 1)
        cv2.putText(
            overlay,
            f"{liters:.1f}L",
            (x + w + 25, marker_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    processed_dir = Path(settings.MEDIA_ROOT) / "scans" / "processed"
    _ensure_dir(processed_dir)
    filename = f"processed_{uuid.uuid4().hex}.jpg"
    output_path = processed_dir / filename
    cv2.imwrite(str(output_path), overlay)

    processing_time_ms = int((time.time() - start) * 1000)

    result = {
        "processed_path": f"scans/processed/{filename}",
        "oil_height_pixels": oil_height_pixels,
        "bottle_height_pixels": bottle_height_pixels,
        "oil_ratio": oil_ratio,
        "remaining_volume_liters": remaining_volume_liters,
        "consumed_volume_liters": consumed_volume_liters,
        "remaining_cups": remaining_cups,
        "consumed_cups": consumed_cups,
        "confidence_score": round(line_confidence, 2),
        "processing_time_ms": processing_time_ms,
    }
    return result


def render_target_overlay(image_path: str, bottle_spec, target_cups: float):
    image = _load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contour = _find_bottle_contour(edges)
    x, y, w, h = cv2.boundingRect(contour)

    target_liters = float(target_cups) * float(bottle_spec.cup_conversion_ratio)
    target_ratio = target_liters / float(bottle_spec.total_volume_liters) if bottle_spec.total_volume_liters else 0
    target_ratio = max(0.0, min(1.0, target_ratio))

    target_y = y + int(h * (1 - target_ratio))
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    cv2.line(overlay, (x, target_y), (x + w, target_y), (255, 165, 0), 2)
    cv2.putText(
        overlay,
        f"Target {target_cups:.1f} cups",
        (x, max(20, target_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 165, 0),
        1,
        cv2.LINE_AA,
    )

    target_dir = Path(settings.MEDIA_ROOT) / "scans" / "targets"
    _ensure_dir(target_dir)
    filename = f"target_{uuid.uuid4().hex}.jpg"
    output_path = target_dir / filename
    cv2.imwrite(str(output_path), overlay)

    return f"scans/targets/{filename}"
