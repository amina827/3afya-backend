# Afia 1.5L Bottle Calibration

## Bottle Specifications
- **Volume**: 1.5 Liters (1500 ml)
- **Total Cups**: 7.5 cups (200 ml per cup)
- **bottle_id**: `afia-1500`
- **Shape**: Cylinder with side handle

## Reference Images

8 calibrated reference images at `media/reference_levels/`:

| File | Oil Level | Volume | Cups |
|------|-----------|--------|------|
| level_100.png | 100% | 1500 ml | 7.5 |
| level_087.png | 87%  | 1300 ml | 6.5 |
| level_073.png | 73%  | 1100 ml | 5.5 |
| level_060.png | 60%  | 900 ml  | 4.5 |
| level_047.png | 47%  | 700 ml  | 3.5 |
| level_033.png | 33%  | 500 ml  | 2.5 |
| level_020.png | 20%  | 300 ml  | 1.5 |
| level_007.png | 7%   | 100 ml  | 0.5 |

## Detection Pipeline

1. **Bottle bbox detection** (`_detect_bottle`): HSV-based yellow/green/red color detection + proportional padding
2. **Normalization** (`_normalize`): CLAHE on LAB color space, resize to 200x500
3. **Reference comparison** (`_compare`): Multi-metric scoring (brightness, golden, HSV, ORB)
4. **Interpolation**: Best match level + confidence score

## Annotated Originals

Original images with hand-drawn blue outlines and white oil-level stickers are kept at `media/reference_levels_annotated/` for future re-calibration.

## Cache

Processed normalized references are cached at `media/reference_cached/`. Delete this folder to force re-processing.

## Adding New References

1. Photograph the bottle at known oil level
2. Save as `level_NNN.png` (NNN = percentage, zero-padded to 3 digits)
3. Add entry to `REFERENCE_LEVELS` in `oil/services/image_processing.py`
4. Delete `media/reference_cached/` to rebuild cache
5. Restart Django dev server

## Troubleshooting

- If detection bbox is too large → check `_detect_bottle` HSV thresholds
- If oil level is wrong → check confidence score (>0.75 = high)
- If cup count is wrong → verify `BottleSpecification.cup_conversion_ratio = 0.200`
