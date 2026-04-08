"""
Standalone validation script for `_detect_bottle()`.

For each of the 8 calibrated references:
  * runs `_detect_bottle()` on the clean source image (media/reference_levels/),
  * extracts the ground-truth bbox from the hand-drawn BLUE outline on the
    corresponding annotated image (media/reference_levels_annotated/),
  * reports the Intersection-over-Union (IoU).

The task description points at `reference_levels/` but the blue outlines were
actually drawn on the copies under `reference_levels_annotated/`; both folders
contain the same underlying photo so the pixel coordinates line up 1:1.
"""

import os
import sys
from pathlib import Path

# --- Django bootstrap ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

import django  # noqa: E402

django.setup()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from oil.services.image_processing import _detect_bottle, ProcessingError  # noqa: E402


SOURCE_DIR = PROJECT_ROOT / "media" / "reference_levels"
ANNOTATED_DIR = PROJECT_ROOT / "media" / "reference_levels_annotated"

REFERENCE_FILES = [
    "level_007.png",
    "level_020.png",
    "level_033.png",
    "level_047.png",
    "level_060.png",
    "level_073.png",
    "level_087.png",
    "level_100.png",
]


def _blue_stroke_mask(annotated_img, source_img):
    """Return a binary mask of pixels that belong to the hand-drawn blue stroke.

    Uses absdiff(annotated, source) then keeps only the changes whose new
    colour is blue-dominant in BGR. PNGs are lossless so the diff is exact
    outside the drawn area.
    """
    if source_img.shape != annotated_img.shape:
        return None
    diff = cv2.absdiff(annotated_img, source_img)
    changed = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 20
    B, G, R = cv2.split(annotated_img)
    is_blue = (B.astype(int) - R.astype(int) > 30) & \
              (B.astype(int) - G.astype(int) > 10)
    mask = (changed & is_blue).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _rectangle_bbox_from_mask(mask):
    """Find the hand-drawn rectangle's bbox in a blue-stroke mask.

    The annotated images also have a blue frame/border drawn around the
    entire canvas, which shows up as one giant edge-touching component.
    We ignore components that touch more than one image edge, then merge
    the remaining stroke fragments into a single bounding rectangle.
    """
    img_h, img_w = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    inner_boxes = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < 50:
            continue
        # Skip the image-border frame: any component touching 2+ sides
        # of the canvas is considered the outer border.
        touches = 0
        if cx <= 1:
            touches += 1
        if cy <= 1:
            touches += 1
        if cx + cw >= img_w - 1:
            touches += 1
        if cy + ch >= img_h - 1:
            touches += 1
        if touches >= 2:
            continue
        # Also skip components whose bbox covers almost the whole image.
        if cw > img_w * 0.9 and ch > img_h * 0.9:
            continue
        inner_boxes.append((cx, cy, cw, ch))

    if not inner_boxes:
        return None

    # Fuse all remaining stroke fragments into one enclosing rectangle
    # — that's the hand-drawn bottle outline.
    x_min = min(b[0] for b in inner_boxes)
    y_min = min(b[1] for b in inner_boxes)
    x_max = max(b[0] + b[2] for b in inner_boxes)
    y_max = max(b[1] + b[3] for b in inner_boxes)
    return x_min, y_min, x_max - x_min, y_max - y_min


def ground_truth_bbox_from_blue(annotated_img, source_img=None):
    """Return (x, y, w, h) enclosing the hand-drawn blue outline.

    The drawn outline is an open rectangle so we take the extents of the
    stroke fragments. We also discard the outer blue image frame so only
    the inner bottle outline remains.
    """
    if source_img is None or source_img.shape != annotated_img.shape:
        hsv = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 120, 60]),
                           np.array([125, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        mask = _blue_stroke_mask(annotated_img, source_img)
        if mask is None:
            return None

    return _rectangle_bbox_from_mask(mask)


def ground_truth_bbox_from_all_refs(pairs):
    """Fuse all annotated/source pairs to find a single stable GT bbox.

    The 8 references show the same bottle at the same position so the
    drawn outline has the same coordinates in every image. OR-ing the
    per-image stroke masks makes the GT detection robust.
    """
    union = None
    for annotated_img, source_img in pairs:
        if source_img.shape != annotated_img.shape:
            continue
        m = _blue_stroke_mask(annotated_img, source_img)
        if m is None:
            continue
        union = m if union is None else cv2.bitwise_or(union, m)
    if union is None:
        return None
    return _rectangle_bbox_from_mask(union)


def iou(box_a, box_b):
    """Intersection over union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = aw * ah + bw * bh - inter_area
    return (inter_area / union) if union > 0 else 0.0


def fmt_box(box):
    if box is None:
        return "(None)"
    x, y, w, h = box
    return f"({x},{y},{w},{h})"


def main():
    # --- Load every source + annotated pair up front ---
    loaded = []
    for filename in REFERENCE_FILES:
        source_img = cv2.imread(str(SOURCE_DIR / filename))
        annotated_img = cv2.imread(str(ANNOTATED_DIR / filename))
        loaded.append((filename, source_img, annotated_img))

    # Fuse the drawn outline across all 8 annotated images to get a
    # single, stable ground-truth bbox (the bottle sits in the same
    # place in every reference).
    pairs = [
        (a, s) for (_, s, a) in loaded
        if s is not None and a is not None
    ]
    fused_gt = ground_truth_bbox_from_all_refs(pairs)

    header = (
        f"{'File':<16}"
        f"{'Detected (x,y,w,h)':<26}"
        f"{'GroundTruth (x,y,w,h)':<28}"
        f"{'IoU':<9}"
        f"Status"
    )
    print(header)
    print("-" * len(header))

    ious = []
    for filename, source_img, annotated_img in loaded:
        if source_img is None or annotated_img is None:
            print(f"{filename:<16}<could not read image>")
            continue

        # Per-image GT (falls back to fused GT if per-image fails).
        gt = ground_truth_bbox_from_blue(annotated_img, source_img)
        if gt is None:
            gt = fused_gt

        try:
            detected = _detect_bottle(source_img)
        except ProcessingError as e:
            det_str = f"ERROR: {e}"
            print(
                f"{filename:<16}"
                f"{det_str:<26}"
                f"{fmt_box(gt):<28}"
                f"{'-':<9}"
                f"FAIL"
            )
            ious.append(0.0)
            continue

        if gt is None:
            print(
                f"{filename:<16}"
                f"{fmt_box(detected):<26}"
                f"<no blue outline found>"
            )
            continue

        score = iou(detected, gt)
        ious.append(score)
        status = "PASS" if score > 0.8 else "FAIL"
        print(
            f"{filename:<16}"
            f"{fmt_box(detected):<26}"
            f"{fmt_box(gt):<28}"
            f"{score:<9.3f}"
            f"{status}"
        )

    print("-" * len(header))
    print(f"Fused GT (shared across all 8 refs): {fmt_box(fused_gt)}")
    if ious:
        mean_iou = sum(ious) / len(ious)
        passed = sum(1 for v in ious if v > 0.8)
        print(f"Mean IoU: {mean_iou:.3f}  |  Passed: {passed}/{len(ious)}")
    else:
        print("No IoU values computed.")


if __name__ == "__main__":
    main()
