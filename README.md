# 3afya Backend

Django backend for Afia bottle analysis. It accepts a bottle photo, detects oil level, returns estimated oil percentage/volume, and generates annotated overlays.

## What This Project Does

- Upload bottle images and create asynchronous scan sessions.
- Detect bottle region and oil surface line from the image.
- Compute oil percentage using image Y coordinates.
- Return remaining/consumed liters and cups.
- Generate overlays for:
  - detected oil level
  - user target level (cups/ml)
- Provide admin-managed reference images for calibration.
- Expose Swagger/ReDoc API docs.

## Tech Stack

- Python + Django 4.2
- Django REST Framework
- OpenCV (computer vision)
- Celery + Redis (async processing)
- SQLite (dev) / Postgres (production)

## Repository Structure

- `core/` Django project config (`settings.py`, `urls.py`, celery wiring)
- `oil/` Main app
- `oil/api/` REST endpoints, serializers
- `oil/services/image_processing.py` CV pipeline and oil computation
- `oil/tasks.py` Celery scan processing task
- `oil/models.py` Bottle, scan, references, feedback, QR models
- `oil/services/reference_data/` bundled fallback reference images
- `oil/CALIBRATION.md` calibration notes and process

## Oil Percentage Formula

The backend computes percentage from detected line and fixed bottle boundaries:

```text
percentage = (bottom_y - oil_level_y) / (bottom_y - top_y) * 100
```

Where image coordinates use top-left origin (Y grows downward), so:

- higher liquid level => smaller `oil_level_y` => larger percentage
- lower liquid level => larger `oil_level_y` => smaller percentage

The result is clamped to `[0, 100]`.

## Current Detection Flow (High Level)

1. Detect bottle bbox from color masks (yellow/green/red label regions).
2. Crop bottle body ROI.
3. Detect stable horizontal oil line using:
   - Gaussian blur
   - Canny edges
   - Hough line segments
   - candidate scoring + weighted stabilization
4. Convert detected line Y to oil percentage using the formula above.
5. Save processed overlay with debug lines (top, bottom, oil level).

## Prerequisites

- Python 3.10+
- Redis (for Celery worker in async mode)
- System libs required by `opencv-python` on your OS

## Environment Variables

Create `.env` in project root.

Minimum for local development:

```env
SECRET_KEY=dev-secret
DEBUG=1
USE_SQLITE=1
DJANGO_TIME_ZONE=UTC
CELERY_TASK_ALWAYS_EAGER=1
```

Optional production-ish values:

```env
ALLOWED_HOSTS=*
DATABASE_URL=postgresql://user:pass@host:5432/dbname
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
FILE_UPLOAD_MAX_MEMORY_SIZE=10485760
DATA_UPLOAD_MAX_MEMORY_SIZE=10485760
```

## Local Setup

1. Create and activate virtual environment.
2. Install dependencies.
3. Run migrations.
4. Seed bottle data and references if needed.
5. Start API server.

Example:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_oil_references
python manage.py runserver
```

Open docs:

- Swagger UI: `http://127.0.0.1:8000/swagger/`
- ReDoc: `http://127.0.0.1:8000/redoc/`

## Running Celery

For real async behavior, disable eager mode and run worker:

```bash
export CELERY_TASK_ALWAYS_EAGER=0
celery -A core worker -l info
```

If you keep `CELERY_TASK_ALWAYS_EAGER=1`, upload requests process immediately in-process (great for local testing).

## Core API Endpoints

Base path: `/api/`

- `POST /api/upload-bottle-image/`
  - starts a scan session (or appends image to existing scan)
- `GET /api/result/<scan_id>/`
  - returns scan status and result when ready
- `POST /api/target-level/`
  - generates a target overlay from `target_cups` or `target_volume_ml`
- `GET /api/bottles/<bottle_id>/slider-config/`
  - returns cup/ml step metadata for frontend slider
- `POST /api/feedback/`
  - submit real-world accuracy feedback
- `POST /api/training/upload/`
  - upload labeled training images
- `GET /api/training/stats/`
  - training image summary
- `POST /api/verify-qr/`
  - QR/label verification flow

## Quick Accuracy Test With Your Image

For your file path (example):

`/home/sohila/Downloads/oil_bottle.jpeg`

### Option A: API workflow

1. Upload image via `POST /api/upload-bottle-image/` with `bottle_id` and image file.
2. Poll `GET /api/result/<scan_id>/` until `status=done`.
3. Compare returned `oil_percentage` with your manually measured ground truth.

### Option B: Python shell (direct function test)

```bash
python manage.py shell
```

```python
from oil.models import BottleSpecification
from oil.services.image_processing import process_bottle_image

bottle = BottleSpecification.objects.get(bottle_id="test-oil-bottle")
result = process_bottle_image("/home/sohila/Downloads/oil_bottle.jpeg", bottle)
print(result)
```

Check:

- `oil_ratio` (0..1)
- `oil_line_debug.top_y`
- `oil_line_debug.bottom_y`
- `oil_line_debug.oil_level_y`
- output image in `media/scans/processed/`

## Interpreting Accuracy

To validate “exact percent” objectively:

1. Define ground truth percent for each test photo (measured volume or calibrated marks).
2. Run batch inference over multiple lighting/background scenarios.
3. Compute MAE (mean absolute error) in percentage points.
4. Track failures where no line is detected (`ProcessingError`).

Recommended practical target for this type of monocular vision setup: low single-digit MAE under controlled capture conditions.

## Troubleshooting

- `No bottle colors detected`
  - ensure bottle is centered, visible, and not strongly color-shifted.
- `Oil level line could not be detected`
  - improve lighting, reduce glare, keep camera level and bottle upright.
- Upload rejected by size
  - increase `FILE_UPLOAD_MAX_MEMORY_SIZE` and `DATA_UPLOAD_MAX_MEMORY_SIZE`.
- Celery task stuck
  - run Redis + worker, or set `CELERY_TASK_ALWAYS_EAGER=1` in local dev.

## Calibration Notes

- Detailed calibration process: `oil/CALIBRATION.md`
- Fallback reference set is bundled in `oil/services/reference_data/`
- Admin can manage `OilReference` records to improve domain accuracy without code redeploy.

## Security/Production Notes

- Use strong `SECRET_KEY`.
- Set restrictive `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS`.
- Use Postgres + persistent media storage.
- Serve media from object storage/CDN for scale.
- Keep `DEBUG=0` in production.

## License

Internal project. Add your organization license terms here.
