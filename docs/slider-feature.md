# Oil Volume Slider — Feature Spec (Frontend ↔ Backend)

Replace the `+` / `−` buttons in the oil-volume picker with a horizontal
**snap slider**. Each step on the slider corresponds to half a cup, and
each step is visually mirrored by a horizontal marker line drawn on the
oil-bottle image at the equivalent oil-level height.

## UX

- Remove `+` / `−` buttons completely.
- Horizontal slider with visible tick marks at every snappable step.
- Snap behaviour — the thumb only rests on valid steps (½ cup, 1 cup,
  1½ cup, 2 cup, …).
- Current value displayed inline as both cups and ml
  (e.g. `½ كوب — 100 ml`).
- Moving the slider moves a horizontal line on the bottle image,
  showing *where the oil would reach* at that volume.
- Optional colour fill below the line and smooth animation.

## Backend contract

Two endpoints power the feature.

### 1. `GET /api/bottles/<bottle_id>/slider-config/`

Called once when the screen loads. Returns every snap step for that
bottle, already mapped to its position on the image.

**Response 200**

```json
{
  "bottle_id": "afia-1500",
  "bottle_name": "Afia 1500ml",
  "total_volume_ml": 1500.0,
  "cup_ml": 250.0,
  "step_ml": 125.0,
  "max_cups": 6.0,
  "steps": [
    { "index": 1, "cups": 0.5, "volume_ml": 125.0, "position_percent": 8.33,  "label": "½ كوب" },
    { "index": 2, "cups": 1.0, "volume_ml": 250.0, "position_percent": 16.67, "label": "1 كوب" },
    { "index": 3, "cups": 1.5, "volume_ml": 375.0, "position_percent": 25.0,  "label": "1½ كوب" }
  ]
}
```

- `step_ml` = `cup_ml / 2` (derived from `BottleSpecification.cup_conversion_ratio`).
- `cup_ml` is admin-editable per bottle (Django admin → BottleSpecification).
- `position_percent` is measured from the **bottom** of the bottle upward.
- `404` if the bottle_id is unknown.

### 2. `POST /api/target-level/`  *(updated)*

Now accepts **either** `target_cups` **or** `target_volume_ml` (exactly
one). Returns a server-rendered overlay PNG *and* the numeric data the
client needs to draw its own overlay on top of the live image.

**Request body**

```json
// option A — slider sends ml directly
{ "scan_id": "…uuid…", "target_volume_ml": 125 }

// option B — legacy cups form (still supported)
{ "scan_id": "…uuid…", "target_cups": 0.5 }
```

**Response 201**

```json
{
  "scan": "…uuid…",
  "target_cups": 0.5,
  "target_volume_ml": 125.0,
  "level_position_percent": 8.33,
  "cup_ml": 250.0,
  "target_image_url": "https://…/media/scans/targets/target_xyz.jpg"
}
```

- `level_position_percent` — use this for the client-side line marker.
  Same reference as `steps[].position_percent` (from bottom, 0–100).
- `target_image_url` — pre-rendered overlay ready to display as-is if
  the client doesn't want to draw its own marker.

## Client-side formula

```
position_percent = (selected_volume_ml / total_volume_ml) * 100
y_pixels_from_top = bottle_height_px * (1 - position_percent / 100)
```

The client can compute the marker position itself from the slider
state — a target-level API call is only needed when the user confirms.

## Data source (admin-editable)

The slider re-derives from:

| Field | Owner | Default |
|---|---|---|
| `BottleSpecification.total_volume_liters` | Admin | per bottle |
| `BottleSpecification.cup_conversion_ratio` | Admin | `0.250` L/cup |

Changing `cup_conversion_ratio` in the admin instantly changes the
slider's step size and the labels — no code deploy.

## Notes

- The same snap list is used for the target picker **and** the oil-level
  marker on the bottle image.
- Labels are Arabic by default (`½ كوب`, `1 كوب`, `1½ كوب`). If an English
  locale is needed, map on the client by `cups` rather than trusting
  `label`.
- `max_cups` is a safety cap (never allow the slider past the total
  volume of the bottle).
