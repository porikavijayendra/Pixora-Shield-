# Pixora Shield

Pixora Shield is a full-stack video privacy application. It uploads a video,
detects privacy-sensitive regions, applies Gaussian blur, and returns a
protected MP4 for preview or download.

## Features

- Keeps the main person clear.
- Blurs secondary or additional detected faces.
- Blurs license plates using fallback image analysis or an optional YOLO model.
- Detects phone numbers, email addresses, Aadhaar-like IDs, SSNs, and card-like
  number patterns through OCR when PII detection is enabled.
- Tracks regions between frames to reduce flicker.
- Provides before-and-after video previews and processing metrics.
- Serves the frontend and API from one Flask application.

## Privacy behavior

During the opening frames, the application identifies the largest or most
stable face as the main person. That tracked region is excluded from face blur
and from nearby plate blur. All other detected faces remain privacy regions.

License plates are accepted only when they have plate-like dimensions and
aspect ratios. This prevents whole cars or large rectangular objects from being
treated as plates.

PII OCR runs on sample frames. Enable it with `PII_DETECTION_ENABLED=true`.
Tesseract must be installed on the system.

## Project structure

```text
frontend/
  index.html       Browser interface
  app.js           Upload, API, preview, and download behavior
  styles.css       Interface styles
backend/
  app.py           Flask API and video processing pipeline
  run.py           Local server entry point
  requirements.txt Python dependencies
  models/          Haar cascade model
  src/models/
    detection/     Face, plate, PII, tracking, and blur logic
tests/
  test_plate_detector.py
  test_tracking.py
render.yaml        Render deployment configuration
```

## Requirements

- Python 3.10 or newer
- OpenCV and NumPy dependencies from `backend/requirements.txt`
- Tesseract OCR for PII detection
- A modern browser

## Run locally

### Windows, Git Bash, or MSYS2

```bash
cd ~/Pixora-Shield-
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
PII_DETECTION_ENABLED=true PORT=3000 \
  python -m backend.run
```

Open `http://localhost:3000`.

### Gunicorn

```bash
cd ~/Pixora-Shield-
source .venv/bin/activate
PII_DETECTION_ENABLED=true \
  gunicorn --chdir backend --bind 0.0.0.0:3000 app:app
```

The frontend and API use the same origin, so no separate frontend server or
cross-origin API URL is required.

## Environment variables

Copy `backend/.env.example` as a reference:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `5000` | Local Flask or Gunicorn HTTP port |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allowed browser origins |
| `PII_DETECTION_ENABLED` | `false` | Enables OCR-based sensitive-text detection |
| `PLATE_MODEL_PATH` | unset | Path to optional YOLO plate weights |
| `PLATE_CONF_THRESH` | `0.25` | Minimum YOLO plate confidence |

For a separate YOLO installation:

```bash
pip install ultralytics
export PLATE_MODEL_PATH=/absolute/path/to/plate-model.pt
```

## API

### `GET /api/health`

Returns the service and detector status.

```json
{
  "status": "ok",
  "face_detection": true,
  "plate_detection": true
}
```

### `POST /api/process`

Send a multipart form containing a video file:

```text
video=@example.mp4
```

The response includes the original and processed URLs, metrics, and a
human-readable processing summary.

### `GET /api/original/<filename>`

Serves the uploaded source video.

### `GET /api/output/<filename>`

Serves the protected output video for preview.

### `GET /api/download/<filename>`

Serves the protected output video as a download.

## Testing

Run the backend test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover face filters, face tracking, whole-car rejection, plate
detection, plate stabilization, and secondary-face partitioning.

## Deployment

The repository includes `render.yaml` for a Render web service.

Render configuration:

- Runtime: Python
- Build command: `pip install -r backend/requirements.txt`
- Start command: `gunicorn --chdir backend --bind 0.0.0.0:$PORT app:app`
- Health check: `/api/health`

Deploy through Render Blueprint with `main` and `render.yaml`.

## Troubleshooting

- **Failed to fetch:** Verify the Flask or Gunicorn process is running and open
  `http://localhost:3000/api/health`.
- **No faces detected:** Use a clear, front-facing or profile view. Haar cascade
  detection is weaker with low resolution, motion blur, or heavy occlusion.
- **Plate not detected:** Add an optional YOLO plate model or improve plate
  resolution and contrast.
- **PII not detected:** Install Tesseract and set
  `PII_DETECTION_ENABLED=true`.
- **Push rejected:** Run `git pull --rebase origin main`, resolve conflicts,
  configure Git identity, then push.
