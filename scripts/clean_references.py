"""
Clean hand-drawn annotations (blue bottle outline + white oil-level sticker)
from the 8 reference level images.

- Loads each PNG in media/reference_levels/
- Builds a conservative BLUE mask (HSV H=90-130, S>80, V>80) and dilates it
- Builds a WHITE mask (R>235 AND G>235 AND B>235)
- Inpaints both masks away with cv2.INPAINT_TELEA
- Backs up the original to media/reference_levels_annotated/
- Overwrites the cleaned image at the original path
- Prints a per-file summary and verifies each output is a valid PNG
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT_ROOT / "media" / "reference_levels"
BACKUP_DIR = PROJECT_ROOT / "media" / "reference_levels_annotated"

REFERENCE_FILES = [
    "level_100.png",
    "level_087.png",
    "level_073.png",
    "level_060.png",
    "level_047.png",
    "level_033.png",
    "level_020.png",
    "level_007.png",
]


def build_blue_mask(bgr: np.ndarray) -> np.ndarray:
    """Conservative hand-drawn blue detection in HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 80, 80], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def build_white_mask(bgr: np.ndarray) -> np.ndarray:
    """White sticker detection: R>235 AND G>235 AND B>235."""
    b, g, r = cv2.split(bgr)
    white = (r > 235) & (g > 235) & (b > 235)
    return (white.astype(np.uint8)) * 255


def clean_one(path: Path, backup_path: Path) -> dict:
    result = {
        "file": path.name,
        "ok": False,
        "blue_pixels": 0,
        "white_pixels": 0,
        "size": None,
        "error": None,
    }

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        result["error"] = f"cv2.imread returned None for {path}"
        return result

    # Backup the original BEFORE overwriting.
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)

    blue_mask = build_blue_mask(img)
    white_mask = build_white_mask(img)

    result["blue_pixels"] = int(np.count_nonzero(blue_mask))
    result["white_pixels"] = int(np.count_nonzero(white_mask))

    combined = cv2.bitwise_or(blue_mask, white_mask)

    cleaned = cv2.inpaint(img, combined, 3, cv2.INPAINT_TELEA)

    ok = cv2.imwrite(str(path), cleaned)
    if not ok:
        result["error"] = f"cv2.imwrite failed for {path}"
        return result

    # Verify written file is valid
    verify = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if verify is None:
        result["error"] = f"Verification failed: cv2.imread returned None after write for {path}"
        return result

    result["size"] = (verify.shape[1], verify.shape[0])  # (width, height)
    result["ok"] = True
    return result


def main() -> int:
    if not REF_DIR.exists():
        print(f"ERROR: reference directory does not exist: {REF_DIR}")
        return 2

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reference dir : {REF_DIR}")
    print(f"Backup dir    : {BACKUP_DIR}")
    print("-" * 72)

    all_results = []
    for name in REFERENCE_FILES:
        src = REF_DIR / name
        backup = BACKUP_DIR / name
        if not src.exists():
            res = {
                "file": name,
                "ok": False,
                "blue_pixels": 0,
                "white_pixels": 0,
                "size": None,
                "error": f"source missing: {src}",
            }
        else:
            try:
                res = clean_one(src, backup)
            except Exception as exc:  # noqa: BLE001
                res = {
                    "file": name,
                    "ok": False,
                    "blue_pixels": 0,
                    "white_pixels": 0,
                    "size": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        all_results.append(res)

        status = "OK   " if res["ok"] else "FAIL "
        size_str = f"{res['size'][0]}x{res['size'][1]}" if res["size"] else "n/a"
        print(
            f"[{status}] {res['file']:<15} "
            f"blue_removed={res['blue_pixels']:>8}  "
            f"white_removed={res['white_pixels']:>8}  "
            f"size={size_str}"
        )
        if res["error"]:
            print(f"         error: {res['error']}")

    print("-" * 72)
    ok_count = sum(1 for r in all_results if r["ok"])
    print(f"Cleaned {ok_count}/{len(REFERENCE_FILES)} images successfully.")

    # Confirm backups
    missing_backups = [
        n for n in REFERENCE_FILES if not (BACKUP_DIR / n).exists()
    ]
    if missing_backups:
        print(f"WARNING: missing backups for: {missing_backups}")
    else:
        print(f"All backups present in: {BACKUP_DIR}")

    return 0 if ok_count == len(REFERENCE_FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
