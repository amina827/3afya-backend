"""
Test oil level detection on all 17 reference images.

Processes each reference image through the detection pipeline and compares
the detected oil percentage against the expected level from the filename.

This script mocks Django settings to avoid requiring the full Django stack.

Usage:
    python test_oil_detection.py
"""

import os
import sys
import time
import types

# --- Mock Django settings so image_processing.py can import ---
# We only need MEDIA_ROOT for the reference cache directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mock_settings = types.ModuleType("django.conf.settings")
mock_settings.MEDIA_ROOT = os.path.join(BASE_DIR, "media")

# Build the mock module hierarchy
django_mod = types.ModuleType("django")
django_conf = types.ModuleType("django.conf")
django_conf.settings = mock_settings
django_mod.conf = django_conf
sys.modules["django"] = django_mod
sys.modules["django.conf"] = django_conf

# Mock the settings object so `from django.conf import settings` works
class _SettingsProxy:
    MEDIA_ROOT = os.path.join(BASE_DIR, "media")
django_conf.settings = _SettingsProxy()

import cv2
import numpy as np
from pathlib import Path

# Now we can import image_processing
from oil.services.image_processing import (
    REFERENCE_DIR,
    REFERENCE_LEVELS,
    _crop_bottle,
    _normalize,
    _detect_label_zone,
    _detect_oil_line_direct,
    _detect_oil_line_side_strips,
    _load_references,
    _find_match,
    _interpolate,
    _fuse_detections,
    _y_rel_to_percentage,
    ProcessingError,
)

# Tolerance: detected level should be within this many percent of expected
TOLERANCE = 8


def test_single_image(image_path, expected_level):
    """Run detection pipeline on a single image and return results."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"Cannot read {image_path}"}

    start = time.time()

    try:
        bottle_crop, (bx, by, bw, bh) = _crop_bottle(img)
        input_norm = _normalize(bottle_crop)
    except ProcessingError as e:
        return {"error": str(e)}

    # 1. Label zone
    label_zone = _detect_label_zone(bottle_crop)

    # 2a. Direct edge detection
    direct_result = None
    edge_y_rel, edge_conf, edge_meta = _detect_oil_line_direct(
        bottle_crop, label_zone=label_zone
    )
    if edge_y_rel is not None:
        direct_result = {
            "percentage": _y_rel_to_percentage(edge_y_rel),
            "confidence": float(edge_conf),
            "y_rel": edge_y_rel,
        }

    # 2b. Side-strip detection
    side_strip_result = None
    try:
        ss_y_rel, ss_conf, side_meta = _detect_oil_line_side_strips(
            bottle_crop, label_zone
        )
        if ss_y_rel is not None:
            side_strip_result = {
                "percentage": _y_rel_to_percentage(ss_y_rel),
                "confidence": float(ss_conf),
                "y_rel": ss_y_rel,
            }
    except Exception:
        pass

    # 2c. Reference comparison
    ref_result = None
    try:
        references = _load_references()
        if len(references) >= 3:
            results = _find_match(input_norm, references)
            ref_level, ref_conf = _interpolate(results)
            ref_result = {
                "percentage": float(np.clip(ref_level, 0, 100)),
                "confidence": float(ref_conf),
            }
    except Exception:
        pass

    # 3. Fuse
    oil_pct, confidence, source = _fuse_detections(
        direct_result, side_strip_result, ref_result, label_zone
    )

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "detected": oil_pct,
        "expected": expected_level,
        "error_pct": abs(oil_pct - expected_level) if oil_pct is not None else None,
        "confidence": confidence,
        "source": source,
        "direct": direct_result["percentage"] if direct_result else None,
        "side_strip": side_strip_result["percentage"] if side_strip_result else None,
        "reference": ref_result["percentage"] if ref_result else None,
        "label_detected": label_zone.get("detected", False),
        "label_top": label_zone.get("top"),
        "label_bottom": label_zone.get("bottom"),
        "time_ms": elapsed_ms,
    }


def main():
    print("=" * 105)
    print("Oil Level Detection Test - All 17 Reference Images")
    print("=" * 105)
    print()

    results = []
    passes = 0
    fails = 0

    header = (
        f"{'Image':<18} {'Expected':>8} {'Detected':>8} {'Error':>6} "
        f"{'Direct':>8} {'Strip':>8} {'Ref':>8} "
        f"{'Conf':>6} {'Label':>6} {'Time':>6} {'Status':>8}"
    )
    print(header)
    print("-" * len(header))

    for filename, expected in REFERENCE_LEVELS:
        img_path = REFERENCE_DIR / filename
        if not img_path.exists():
            print(f"{filename:<18} MISSING")
            continue

        result = test_single_image(img_path, expected)

        if "error" in result and "detected" not in result:
            print(f"{filename:<18} ERROR: {result['error']}")
            fails += 1
            continue

        detected = result["detected"]
        error = result["error_pct"]
        passed = error is not None and error <= TOLERANCE

        if passed:
            passes += 1
            status = "PASS"
        else:
            fails += 1
            status = "FAIL"

        direct_str = f"{result['direct']:.0f}%" if result['direct'] is not None else "---"
        strip_str = f"{result['side_strip']:.0f}%" if result['side_strip'] is not None else "---"
        ref_str = f"{result['reference']:.0f}%" if result['reference'] is not None else "---"
        det_str = f"{detected:.0f}%" if detected is not None else "---"
        err_str = f"{error:.1f}" if error is not None else "---"
        label_str = "yes" if result.get("label_detected") else "no"

        print(
            f"{filename:<18} {expected:>7}% {det_str:>8} {err_str:>6} "
            f"{direct_str:>8} {strip_str:>8} {ref_str:>8} "
            f"{result['confidence']:>5.2f} {label_str:>6} {result['time_ms']:>5}ms "
            f"{'  ' + status:>8}"
        )

        results.append(result)

    print("-" * len(header))
    total = passes + fails
    print(f"\nSummary: {passes}/{total} passed (tolerance: +/-{TOLERANCE}%)")

    if results:
        errors = [r["error_pct"] for r in results if r.get("error_pct") is not None]
        if errors:
            print(f"Mean error: {np.mean(errors):.1f}%")
            print(f"Max error:  {max(errors):.1f}%")
            print(f"Median:     {np.median(errors):.1f}%")

        times = [r["time_ms"] for r in results if "time_ms" in r]
        if times:
            print(f"Avg time:   {np.mean(times):.0f}ms")
            print(f"Max time:   {max(times)}ms")

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
