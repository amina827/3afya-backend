"""
Calibration script for Afia 1.5L bottle oil-level reference images.

Important: the reference images in ``media/reference_levels/`` are NOT the
hand-annotated bottles originally described in the task brief. They are
stock photos of the Afia 1.5L bottle in which the empty portion of the
bottle has been alpha-blended toward white to simulate different fill
levels. There is no hand-drawn blue outline and no white sticker.

Given that reality, this script:

1. Loads all 8 reference images with OpenCV.
2. Derives a stable bottle bounding box from ``level_100.png`` using the
   yellow-oil HSV mask (the full bottle is entirely yellow at 100 %).
   That bbox is then reused for all 8 images (they share the exact same
   bottle position, since the files are derived from the same source
   photo).
3. Detects the oil level per image by comparing each image's Lab ``b*``
   (yellow-blue) profile against ``level_100``'s profile inside the
   bottle bounding box. Rows that have "lost" yellow intensity compared
   to the 100 % reference are classified as whitened / empty. The top of
   the remaining oil is the lowest whitened row + 1.
4. Maps the pixel-space oil level to a ratio / percentage / millilitres
   using a fixed oil column (oil_top_y .. oil_bottom_y) learned from the
   100 % and 7 % frames.
5. Writes ``media/calibration.json`` with the per-image results and a
   summary block, and prints a readable table.

Run::

    C:\\Users\\Dell\\Desktop\\3afya backend\\venv\\Scripts\\python.exe oil/services/calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Users\Dell\Desktop\3afya backend")
REF_DIR = PROJECT_ROOT / "media" / "reference_levels"
OUT_JSON = PROJECT_ROOT / "media" / "calibration.json"

BOTTLE_CAPACITY_ML = 1500

# filename -> (expected_pct, expected_ml)
REFERENCES = [
    ("level_100.png", 100.0,  1500),
    ("level_087.png",  86.67, 1300),
    ("level_073.png",  73.33, 1100),
    ("level_060.png",  60.0,   900),
    ("level_047.png",  46.67,  700),
    ("level_033.png",  33.33,  500),
    ("level_020.png",  20.0,   300),
    ("level_007.png",   6.67,  100),
]

# Column band inside the bottle used for row profiling. Chosen to sit
# inside the bottle body, away from the left label text.
PROFILE_X_MIN = 130
PROFILE_X_MAX = 200

# A row is considered "whitened" when its b* dropped by more than this
# many units compared to the level_100 reference.
WHITEN_DELTA_THRESH = 15.0


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_bottle_bbox(img_bgr: np.ndarray):
    """Return (x, y, w, h) bbox of the bottle in the given image.

    Uses the yellow HSV mask (oil + label + cap) plus a large morphological
    close to fuse the bottle body into a single blob. This works well on
    the 100 % image where the bottle is almost entirely yellow. For the
    partially-whitened frames the same bbox is reused by the caller.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 60, 80), (40, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None
    x, y, w, h = cv2.boundingRect(largest)

    # The yellow cap sits slightly above the bottle shoulder and can be
    # missed by the morph-close. Extend the top of the bbox up to the
    # topmost yellow pixel in the raw mask.
    ys = np.where(yellow.any(axis=1))[0]
    if len(ys):
        y_top = int(ys.min())
        if y_top < y:
            h += y - y_top
            y = y_top
    return int(x), int(y), int(w), int(h)


def row_b_profile(img_bgr: np.ndarray) -> np.ndarray:
    """Return a per-row average of Lab ``b*`` within the profile column band."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    b = lab[:, :, 2].astype(np.int32) - 128  # signed yellow-blue
    band = b[:, PROFILE_X_MIN:PROFILE_X_MAX]
    return band.mean(axis=1)


def detect_oil_level_y(
    img_bgr: np.ndarray,
    ref_profile: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> int | None:
    """Detect the oil surface Y coordinate for a partially-filled image.

    Compares the image's row-wise b* profile against the 100 % reference
    profile. Rows whose b* dropped by more than ``WHITEN_DELTA_THRESH``
    (i.e. the oil was repainted white) are flagged as "whitened". The
    oil surface is one row below the lowest whitened row inside the
    bottle bbox.
    """
    prof = row_b_profile(img_bgr)
    diff = ref_profile - prof  # positive where yellow was lost
    x, y, w, h = bbox
    # restrict to rows inside the bottle
    diff_in_bottle = diff.copy()
    diff_in_bottle[:y] = 0
    diff_in_bottle[y + h:] = 0
    whitened = diff_in_bottle > WHITEN_DELTA_THRESH
    idx = np.where(whitened)[0]
    if len(idx) == 0:
        return None
    return int(idx.max()) + 1  # row just below the lowest whitened row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not REF_DIR.is_dir():
        print(f"[ERROR] Reference directory not found: {REF_DIR}",
              file=sys.stderr)
        return 1

    # --- Load the 100 % reference first; it anchors the bottle bbox and
    # the row-profile baseline used by all other images.
    ref_path = REF_DIR / "level_100.png"
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        print(f"[ERROR] Cannot read reference image: {ref_path}",
              file=sys.stderr)
        return 1

    bbox = detect_bottle_bbox(ref_img)
    if bbox is None:
        print("[ERROR] Could not detect bottle bbox in level_100.png",
              file=sys.stderr)
        return 2
    bx, by, bw, bh = bbox
    ref_profile = row_b_profile(ref_img)

    # Oil column reference Y bounds. In the 100 % image the oil fills the
    # bottle from around the shoulder (~ bottle_top + cap height) down to
    # the bottom of the bottle. We use the bbox as the physical reference
    # but the oil-level detection also needs a logical "top of oil at 100 %"
    # and "bottom of oil at 0 %" to map pixels -> ratios. We learn these by
    # observing the highest whitened row across the partial-fill frames
    # (that row is where the oil surface sits when the bottle is full), and
    # the lowest (where it sits when nearly empty).
    partial_whitened_tops: list[int] = []
    partial_oil_levels: list[int] = []
    partial_results: dict[str, int] = {}
    for filename, _pct, _ml in REFERENCES:
        if filename == "level_100.png":
            continue
        p = REF_DIR / filename
        img = cv2.imread(str(p))
        if img is None:
            continue
        prof = row_b_profile(img)
        diff = ref_profile - prof
        diff_masked = diff.copy()
        diff_masked[: by] = 0
        diff_masked[by + bh :] = 0
        whitened = diff_masked > WHITEN_DELTA_THRESH
        idx = np.where(whitened)[0]
        if len(idx):
            partial_whitened_tops.append(int(idx.min()))
            oil_y = int(idx.max()) + 1
            partial_oil_levels.append(oil_y)
            partial_results[filename] = oil_y

    if not partial_oil_levels:
        print("[ERROR] Could not detect oil levels in any partial image.",
              file=sys.stderr)
        return 3

    # Top-of-oil Y when bottle is 100 % full: average of the whitened-start
    # rows (every partial frame starts whitening at the same row, the
    # shoulder of the bottle).
    oil_top_y = int(round(sum(partial_whitened_tops) / len(partial_whitened_tops)))

    # Bottom-of-oil Y when bottle is 0 % full: we use the lowest observed
    # oil surface (at 6.67 %) and extrapolate down to 0 %.
    # y_oil = oil_top_y + (1 - ratio) * (oil_bottom - oil_top)
    # Rearrange with ratio = expected_pct/100 for the 7 % frame to solve for
    # oil_bottom_y.
    seven_filename = "level_007.png"
    seven_ratio = 6.67 / 100.0
    if seven_filename in partial_results:
        y_seven = partial_results[seven_filename]
        oil_bottom_y = int(round(
            oil_top_y + (y_seven - oil_top_y) / (1.0 - seven_ratio)
        ))
    else:
        oil_bottom_y = by + bh

    oil_column_px = oil_bottom_y - oil_top_y
    if oil_column_px <= 0:
        print("[ERROR] Degenerate oil column.", file=sys.stderr)
        return 4

    # --- Build per-image rows using the learned oil column.
    bottles = []
    for filename, expected_pct, expected_ml in REFERENCES:
        path = REF_DIR / filename
        if not path.exists():
            print(f"[WARN] Missing reference image: {filename}", file=sys.stderr)
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read {filename}", file=sys.stderr)
            continue

        if filename == "level_100.png":
            oil_level_y = oil_top_y
        else:
            y = detect_oil_level_y(img, ref_profile, bbox)
            if y is None:
                print(f"[WARN] Oil level not detectable in {filename} - skipping",
                      file=sys.stderr)
                continue
            oil_level_y = y

        computed_ratio = (oil_bottom_y - oil_level_y) / oil_column_px
        computed_ratio = float(max(0.0, min(1.0, computed_ratio)))
        computed_pct = computed_ratio * 100.0
        computed_ml = int(round(computed_ratio * BOTTLE_CAPACITY_ML))
        error_pct = abs(computed_pct - expected_pct)

        bottles.append({
            "filename": filename,
            "expected_pct": round(expected_pct, 2),
            "expected_ml": int(expected_ml),
            "bottle_bbox": {"x": int(bx), "y": int(by),
                            "w": int(bw), "h": int(bh)},
            "oil_level_y": int(oil_level_y),
            "computed_ratio": round(computed_ratio, 4),
            "computed_pct": round(computed_pct, 2),
            "computed_ml": int(computed_ml),
            "error_pct": round(error_pct, 2),
        })

    if not bottles:
        print("[ERROR] No bottles processed successfully.", file=sys.stderr)
        return 5

    widths = [b["bottle_bbox"]["w"] for b in bottles]
    heights = [b["bottle_bbox"]["h"] for b in bottles]
    errors = [b["error_pct"] for b in bottles]

    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    avg_aspect = avg_h / avg_w if avg_w else 0.0

    summary = {
        "avg_bottle_aspect": round(avg_aspect, 4),
        "avg_bottle_width": round(avg_w, 2),
        "avg_bottle_height": round(avg_h, 2),
        "max_error_pct": round(max(errors), 2),
        "mean_error_pct": round(sum(errors) / len(errors), 2),
        "num_references": len(bottles),
        "bottle_capacity_ml": BOTTLE_CAPACITY_ML,
        "oil_top_y": int(oil_top_y),
        "oil_bottom_y": int(oil_bottom_y),
        "oil_column_px": int(oil_column_px),
        "profile_x_range": [PROFILE_X_MIN, PROFILE_X_MAX],
        "whiten_delta_threshold": WHITEN_DELTA_THRESH,
    }

    output = {"bottles": bottles, "summary": summary}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    # ------------------------------------------------------------------
    # Pretty print the results table
    # ------------------------------------------------------------------
    print("\nCalibration results")
    print("=" * 108)
    header = (
        f"{'File':<14} {'Exp%':>7} {'ExpML':>6} "
        f"{'BBox (x,y,w,h)':<22} {'OilY':>5} "
        f"{'Ratio':>7} {'Cmp%':>7} {'CmpML':>6} {'Err%':>6}"
    )
    print(header)
    print("-" * 108)
    for b in bottles:
        bb = b["bottle_bbox"]
        bbox_str = f"({bb['x']},{bb['y']},{bb['w']},{bb['h']})"
        print(
            f"{b['filename']:<14} "
            f"{b['expected_pct']:>7.2f} {b['expected_ml']:>6d} "
            f"{bbox_str:<22} {b['oil_level_y']:>5d} "
            f"{b['computed_ratio']:>7.4f} {b['computed_pct']:>7.2f} "
            f"{b['computed_ml']:>6d} {b['error_pct']:>6.2f}"
        )
    print("-" * 108)
    print(
        f"Summary: avg_w={summary['avg_bottle_width']:.2f}  "
        f"avg_h={summary['avg_bottle_height']:.2f}  "
        f"avg_aspect={summary['avg_bottle_aspect']:.4f}  "
        f"mean_err={summary['mean_error_pct']:.2f}%  "
        f"max_err={summary['max_error_pct']:.2f}%"
    )
    print(
        f"Oil column: top_y={summary['oil_top_y']}  "
        f"bottom_y={summary['oil_bottom_y']}  "
        f"height_px={summary['oil_column_px']}"
    )
    print(f"\nWrote calibration JSON -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
