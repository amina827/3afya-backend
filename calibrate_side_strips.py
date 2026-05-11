#!/usr/bin/env python3
"""
Afia 1.5L Oil Level Detection - Side Strip Calibration Script
==============================================================
Analyzes 16 reference images (0ml to 1500ml in 100ml steps) to build
a calibration table mapping fill_ratio -> volume_ml.

Pipeline per image:
  1. Detect red cap via HSV thresholding
  2. Derive bottle body bounds from cap geometry
  3. Define thin side strips on left and right edges of the bottle
  4. Detect oil in each strip using HSV colour range
  5. Compute fill_ratio = (bottle_bottom - oil_top) / (bottle_bottom - cap_bottom)
  6. Cross-validate left vs right strip for confidence scoring

Outputs:
  - Console table with all measurements
  - calibration_table.json   (volume_ml <-> fill_ratio mapping)
  - calibration_debug/       (annotated images for visual verification)
"""

import os
import sys
import json
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
REFERENCE_DIR = r"C:\Users\Dell\Desktop\3afya backend\oil\services\reference_data"
CALIBRATION_JSON = r"C:\Users\Dell\Desktop\3afya backend\oil\services\calibration_table.json"
DEBUG_DIR = r"C:\Users\Dell\Desktop\3afya backend\calibration_debug"

VOLUMES = list(range(0, 1600, 100))  # 0, 100, ..., 1500

CAP_TO_BODY_RATIO = 9.0
CAP_WIDTH_FACTOR = 1.8
STRIP_EDGE_MARGIN = 5
STRIP_WIDTH_FRACTION = 0.05
SMOOTHING_KERNEL = 15
OIL_THRESHOLD_FRACTION = 0.4

# HSV ranges
OIL_HSV_LOWER = np.array([15, 30, 60])
OIL_HSV_UPPER = np.array([45, 255, 250])

# ──────────────────────────────────────────────────────────────────────
# Helper: clamp value to [lo, hi]
# ──────────────────────────────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(val, hi))


# ──────────────────────────────────────────────────────────────────────
# Step 1: Detect the red cap
# ──────────────────────────────────────────────────────────────────────
def detect_cap(image, hsv):
    """
    Find the red bottle cap in the upper half of the image.
    Returns (cap_center_x, cap_bottom_y, cap_height) or None.
    """
    h, w = image.shape[:2]
    upper_half = h // 2

    # Red colour in HSV wraps around 0/180
    red_mask_1 = cv2.inRange(hsv[:upper_half], np.array([0, 100, 70]), np.array([10, 255, 255]))
    red_mask_2 = cv2.inRange(hsv[:upper_half], np.array([160, 100, 70]), np.array([179, 255, 255]))
    red_mask = red_mask_1 | red_mask_2

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    best = None
    best_area = 0

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 5000:
            continue

        aspect = comp_w / max(comp_h, 1)
        if aspect < 0.8 or aspect > 2.5:
            continue

        if area > best_area:
            best_area = area
            best = i

    if best is None:
        return None

    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    comp_w = stats[best, cv2.CC_STAT_WIDTH]
    comp_h = stats[best, cv2.CC_STAT_HEIGHT]

    cap_center_x = x + comp_w // 2
    cap_bottom_y = y + comp_h
    cap_height = comp_h

    return cap_center_x, cap_bottom_y, cap_height


# ──────────────────────────────────────────────────────────────────────
# Step 2: Derive bottle bounds from cap geometry
# ──────────────────────────────────────────────────────────────────────
def compute_bottle_bounds(cap_center_x, cap_bottom_y, cap_height, img_h, img_w):
    bottle_bottom_y = cap_bottom_y + int(CAP_TO_BODY_RATIO * cap_height)
    bottle_left = cap_center_x - int(CAP_WIDTH_FACTOR * cap_height)
    bottle_right = cap_center_x + int(CAP_WIDTH_FACTOR * cap_height)

    # Clamp to image bounds
    bottle_bottom_y = clamp(bottle_bottom_y, 0, img_h - 1)
    bottle_left = clamp(bottle_left, 0, img_w - 1)
    bottle_right = clamp(bottle_right, 0, img_w - 1)

    return bottle_bottom_y, bottle_left, bottle_right


# ──────────────────────────────────────────────────────────────────────
# Step 3: Define side strips
# ──────────────────────────────────────────────────────────────────────
def compute_strips(bottle_left, bottle_right, img_w):
    strip_width = int((bottle_right - bottle_left) * STRIP_WIDTH_FRACTION)
    strip_width = max(strip_width, 3)  # ensure at least 3 pixels wide

    left_start = clamp(bottle_left + STRIP_EDGE_MARGIN, 0, img_w - 1)
    left_end = clamp(left_start + strip_width, 0, img_w - 1)

    right_end = clamp(bottle_right - STRIP_EDGE_MARGIN, 0, img_w - 1)
    right_start = clamp(right_end - strip_width, 0, img_w - 1)

    return (left_start, left_end), (right_start, right_end)


# ──────────────────────────────────────────────────────────────────────
# Step 4: Detect oil level in a single strip
# ──────────────────────────────────────────────────────────────────────
def detect_oil_in_strip(oil_mask, cap_bottom_y, bottle_bottom_y, strip_x_start, strip_x_end):
    """
    Scan the strip from top to bottom to find the topmost row with significant oil.
    Returns (oil_top_y_absolute, fill_ratio) or (None, None) if no oil found.
    """
    strip_oil = oil_mask[cap_bottom_y:bottle_bottom_y, strip_x_start:strip_x_end]
    if strip_oil.size == 0:
        return None, None

    strip_w = strip_x_end - strip_x_start
    row_oil_count = np.sum(strip_oil > 0, axis=1)

    # Smooth to remove noise
    kernel_size = min(SMOOTHING_KERNEL, len(row_oil_count))
    if kernel_size < 1:
        return None, None
    smoothed = np.convolve(row_oil_count, np.ones(kernel_size) / kernel_size, mode='same')

    threshold = strip_w * OIL_THRESHOLD_FRACTION

    # Find TOPMOST row where smoothed >= threshold
    oil_rows = np.where(smoothed >= threshold)[0]
    if len(oil_rows) == 0:
        return None, None

    oil_top_local = oil_rows[0]
    oil_top_y = cap_bottom_y + oil_top_local

    body_height = bottle_bottom_y - cap_bottom_y
    if body_height <= 0:
        return None, None

    fill_ratio = (bottle_bottom_y - oil_top_y) / body_height
    fill_ratio = clamp(fill_ratio, 0.0, 1.0)

    return oil_top_y, fill_ratio


# ──────────────────────────────────────────────────────────────────────
# Step 6: Double-strip validation
# ──────────────────────────────────────────────────────────────────────
def merge_strip_results(left_ratio, right_ratio):
    """
    Merge left and right strip fill ratios.
    Returns (final_fill_ratio, confidence).
    """
    if left_ratio is not None and right_ratio is not None:
        diff = abs(left_ratio - right_ratio)
        if diff <= 0.05:
            return (left_ratio + right_ratio) / 2.0, "high"
        elif diff <= 0.15:
            return max(left_ratio, right_ratio), "medium"
        else:
            return max(left_ratio, right_ratio), "low"
    elif left_ratio is not None:
        return left_ratio, "single-left"
    elif right_ratio is not None:
        return right_ratio, "single-right"
    else:
        return 0.0, "none"


# ──────────────────────────────────────────────────────────────────────
# Draw debug annotations
# ──────────────────────────────────────────────────────────────────────
def draw_debug(image, cap_center_x, cap_bottom_y, cap_height,
               bottle_bottom_y, bottle_left, bottle_right,
               left_strip, right_strip,
               left_oil_top_y, right_oil_top_y,
               final_ratio, confidence, known_ml):
    debug = image.copy()
    h, w = debug.shape[:2]

    # Cap bounding box (reconstructed)
    cap_top = max(0, cap_bottom_y - cap_height)
    cap_half_w = int(CAP_WIDTH_FACTOR * cap_height * 0.5)
    cv2.rectangle(debug,
                  (cap_center_x - cap_half_w, cap_top),
                  (cap_center_x + cap_half_w, cap_bottom_y),
                  (0, 0, 255), 2)

    # Bottle body rectangle
    cv2.rectangle(debug,
                  (bottle_left, cap_bottom_y),
                  (bottle_right, bottle_bottom_y),
                  (255, 0, 0), 2)

    # Left strip
    cv2.rectangle(debug,
                  (left_strip[0], cap_bottom_y),
                  (left_strip[1], bottle_bottom_y),
                  (0, 255, 255), 2)

    # Right strip
    cv2.rectangle(debug,
                  (right_strip[0], cap_bottom_y),
                  (right_strip[1], bottle_bottom_y),
                  (0, 255, 255), 2)

    # Oil level lines
    if left_oil_top_y is not None:
        cv2.line(debug,
                 (left_strip[0], left_oil_top_y),
                 (left_strip[1], left_oil_top_y),
                 (0, 255, 0), 3)

    if right_oil_top_y is not None:
        cv2.line(debug,
                 (right_strip[0], right_oil_top_y),
                 (right_strip[1], right_oil_top_y),
                 (0, 255, 0), 3)

    # Draw merged oil level line across bottle
    if left_oil_top_y is not None or right_oil_top_y is not None:
        oil_ys = [y for y in [left_oil_top_y, right_oil_top_y] if y is not None]
        avg_oil_y = int(np.mean(oil_ys))
        cv2.line(debug,
                 (bottle_left, avg_oil_y),
                 (bottle_right, avg_oil_y),
                 (0, 200, 0), 2)

    # Text annotations
    label = f"{known_ml}ml  ratio={final_ratio:.3f}  conf={confidence}"
    cv2.putText(debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Cap center crosshair
    cv2.drawMarker(debug, (cap_center_x, cap_bottom_y - cap_height // 2),
                   (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    return debug


# ──────────────────────────────────────────────────────────────────────
# Main calibration loop
# ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    results = []
    print()
    print("=" * 130)
    print(f"{'File':<14} {'ml':>5} {'cap_cx':>7} {'cap_by':>7} {'cap_h':>6} {'bot_by':>7} "
          f"{'L_ratio':>8} {'R_ratio':>8} {'final':>8} {'conf':<12}")
    print("-" * 130)

    for vol in VOLUMES:
        fname = f"{vol}.jpeg"
        fpath = os.path.join(REFERENCE_DIR, fname)

        if not os.path.isfile(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue

        image = cv2.imread(fpath)
        if image is None:
            print(f"  WARNING: Failed to read {fpath}, skipping")
            continue

        img_h, img_w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Step 1: Cap detection ---
        cap_result = detect_cap(image, hsv)
        if cap_result is None:
            print(f"{fname:<14} {vol:>5}   *** CAP DETECTION FAILED ***")
            results.append({
                "volume_ml": vol, "fill_ratio": 0.0,
                "cap_center_x": None, "cap_bottom_y": None, "cap_height": None,
                "bottle_bottom_y": None,
                "left_strip_ratio": None, "right_strip_ratio": None,
                "confidence": "cap_failed"
            })
            continue

        cap_center_x, cap_bottom_y, cap_height = cap_result

        # --- Step 2: Bottle bounds ---
        bottle_bottom_y, bottle_left, bottle_right = compute_bottle_bounds(
            cap_center_x, cap_bottom_y, cap_height, img_h, img_w
        )

        # --- Step 3: Side strips ---
        left_strip, right_strip = compute_strips(bottle_left, bottle_right, img_w)

        # --- Step 4: Oil detection ---
        oil_mask = cv2.inRange(hsv, OIL_HSV_LOWER, OIL_HSV_UPPER)

        # Light morphological cleanup on oil mask
        oil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, oil_kernel, iterations=1)

        left_oil_top_y, left_ratio = detect_oil_in_strip(
            oil_mask, cap_bottom_y, bottle_bottom_y, left_strip[0], left_strip[1]
        )
        right_oil_top_y, right_ratio = detect_oil_in_strip(
            oil_mask, cap_bottom_y, bottle_bottom_y, right_strip[0], right_strip[1]
        )

        # --- Step 5 & 6: Merge ---
        # Special case: 0ml should always be 0.0
        if vol == 0 and left_ratio is None and right_ratio is None:
            final_ratio = 0.0
            confidence = "empty"
        else:
            final_ratio, confidence = merge_strip_results(left_ratio, right_ratio)

        left_str = f"{left_ratio:.4f}" if left_ratio is not None else "N/A"
        right_str = f"{right_ratio:.4f}" if right_ratio is not None else "N/A"

        print(f"{fname:<14} {vol:>5} {cap_center_x:>7} {cap_bottom_y:>7} {cap_height:>6} "
              f"{bottle_bottom_y:>7} {left_str:>8} {right_str:>8} {final_ratio:>8.4f} {confidence:<12}")

        results.append({
            "volume_ml": vol,
            "fill_ratio": round(final_ratio, 4),
            "cap_center_x": cap_center_x,
            "cap_bottom_y": cap_bottom_y,
            "cap_height": cap_height,
            "bottle_bottom_y": bottle_bottom_y,
            "left_strip_ratio": round(left_ratio, 4) if left_ratio is not None else None,
            "right_strip_ratio": round(right_ratio, 4) if right_ratio is not None else None,
            "confidence": confidence
        })

        # --- Debug image ---
        debug_img = draw_debug(
            image, cap_center_x, cap_bottom_y, cap_height,
            bottle_bottom_y, bottle_left, bottle_right,
            left_strip, right_strip,
            left_oil_top_y, right_oil_top_y,
            final_ratio, confidence, vol
        )
        debug_path = os.path.join(DEBUG_DIR, f"debug_{vol}ml.jpg")
        cv2.imwrite(debug_path, debug_img)

    print("=" * 130)
    print()

    # ── Post-processing: ensure monotonicity ──
    # fill_ratio should be non-decreasing with volume
    # Fix any small inversions caused by measurement noise
    print("Post-processing: checking monotonicity...")
    sorted_results = sorted(results, key=lambda r: r["volume_ml"])

    raw_ratios = [(r["volume_ml"], r["fill_ratio"]) for r in sorted_results]
    print("  Raw ratios:", [(v, f"{r:.4f}") for v, r in raw_ratios])

    # Enforce non-decreasing: if a ratio is less than its predecessor, set it to predecessor
    for i in range(1, len(sorted_results)):
        if sorted_results[i]["fill_ratio"] < sorted_results[i - 1]["fill_ratio"]:
            old = sorted_results[i]["fill_ratio"]
            sorted_results[i]["fill_ratio"] = sorted_results[i - 1]["fill_ratio"]
            print(f"  Fixed inversion at {sorted_results[i]['volume_ml']}ml: "
                  f"{old:.4f} -> {sorted_results[i]['fill_ratio']:.4f}")

    # ── Build calibration table (only volume_ml and fill_ratio) ──
    calibration_table = [
        {"volume_ml": r["volume_ml"], "fill_ratio": r["fill_ratio"]}
        for r in sorted_results
    ]

    os.makedirs(os.path.dirname(CALIBRATION_JSON), exist_ok=True)
    with open(CALIBRATION_JSON, "w") as f:
        json.dump(calibration_table, f, indent=2)

    print(f"\nCalibration table saved to: {CALIBRATION_JSON}")
    print(f"Debug images saved to:      {DEBUG_DIR}")

    # ── Print final calibration table ──
    print("\n" + "=" * 50)
    print("FINAL CALIBRATION TABLE")
    print("=" * 50)
    print(f"{'Volume (ml)':>12}  {'Fill Ratio':>12}")
    print("-" * 28)
    for entry in calibration_table:
        print(f"{entry['volume_ml']:>12}  {entry['fill_ratio']:>12.4f}")
    print("=" * 50)

    # ── Print detailed results ──
    print("\nDetailed per-image results:")
    print(json.dumps(sorted_results, indent=2))

    return sorted_results


if __name__ == "__main__":
    main()
