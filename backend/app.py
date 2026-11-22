import os
import re
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, send_from_directory
from flask import request
from flask_cors import CORS

from src.models.detection.blur_utils import apply_blur_regions
from src.models.detection.plate_blur.plate_detector import PlateDetector


BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","))

def find_face_cascade():
    candidates = []
    candidates.append(
        str(BACKEND_DIR / "models" / "haarcascade_frontalface_default.xml")
    )
    if hasattr(cv2, "data"):
        candidates.append(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
    candidates.extend([
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ])
    return next((path for path in candidates if Path(path).is_file()), "")


def find_profile_cascade():
    candidates = []
    if hasattr(cv2, "data"):
        candidates.append(
            str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
        )
    candidates.extend([
        str(BACKEND_DIR / "models" / "haarcascade_profileface.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
        "/usr/share/opencv/haarcascades/haarcascade_profileface.xml",
    ])
    return next((path for path in candidates if Path(path).is_file()), "")


face_cascade_path = find_face_cascade()
profile_cascade_path = find_profile_cascade()
try:
    face_detector = (
        cv2.CascadeClassifier(face_cascade_path)
        if face_cascade_path and hasattr(cv2, "CascadeClassifier")
        else None
    )
    profile_detector = (
        cv2.CascadeClassifier(profile_cascade_path)
        if profile_cascade_path and hasattr(cv2, "CascadeClassifier")
        else None
    )
except cv2.error:
    face_detector = None
    profile_detector = None
plate_detector = PlateDetector(conf_thresh=float(os.getenv("PLATE_CONF_THRESH", "0.25")))
try:
    import pytesseract
except ImportError:
    pytesseract = None


PII_PATTERNS = (
    re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{12}\b"),
    re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
)


def is_visible_face(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    if box_width < 32 or box_height < 32:
        return False
    frame_ratio = (box_width * box_height) / float(width * height)
    aspect_ratio = box_width / float(box_height)
    return frame_ratio >= 0.0008 and 0.60 <= aspect_ratio <= 1.45


def detect_faces(frame):
    if face_detector is None or face_detector.empty():
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detected_faces = []
    for scale_factor, neighbors in ((1.05, 5), (1.10, 7)):
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=neighbors,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        detected_faces.extend([tuple(face) for face in faces])
        if len(faces) >= 2:
            break

    if profile_detector is not None and not profile_detector.empty():
        for view in (gray, cv2.flip(gray, 1)):
            faces = profile_detector.detectMultiScale(
                view,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(32, 32),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for x, y, w, h in faces:
                if view is not gray:
                    x = gray.shape[1] - x - w
                detected_faces.append((int(x), int(y), int(w), int(h)))

    height, width = frame.shape[:2]
    visible_faces = []
    for (x, y, w, h) in detected_faces:
        frame_ratio = (w * h) / float(width * height)
        aspect_ratio = w / float(h)
        if frame_ratio < 0.0008:
            continue
        if not 0.60 <= aspect_ratio <= 1.45:
            continue
        candidate = [int(x), int(y), int(x + w), int(y + h)]
        if not any(overlap_ratio(candidate, kept) >= 0.40 for kept in visible_faces):
            visible_faces.append(candidate)
    return visible_faces


def detect_pii_regions(frame, frame_id):
    if os.getenv("PII_DETECTION_ENABLED", "true").lower() not in ("1", "true", "yes"):
        return []
    if pytesseract is None or frame_id % 30 != 0:
        return []

    try:
        data = pytesseract.image_to_data(
            frame,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []

    regions = []
    for index, text in enumerate(data.get("text", [])):
        text = text.strip()
        if not text or not any(pattern.search(text) for pattern in PII_PATTERNS):
            continue
        x = int(data["left"][index])
        y = int(data["top"][index])
        w = int(data["width"][index])
        h = int(data["height"][index])
        regions.append([x, y, x + w, y + h])
    return regions


def stabilize_regions(previous_regions, current_regions, iou_threshold=0.55):
    if not previous_regions:
        return current_regions

    stabilized = []
    for region in current_regions:
        match = None
        best_iou = iou_threshold
        x1, y1, x2, y2 = region
        for previous in previous_regions:
            px1, py1, px2, py2 = previous
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = max(1, (x2 - x1) * (y2 - y1))
            previous_area = max(1, (px2 - px1) * (py2 - py1))
            iou = intersection / (area + previous_area - intersection)
            if iou >= best_iou:
                best_iou = iou
                match = previous
        stabilized.append(match if match is not None else region)
    return stabilized


def overlap_ratio(first, second):
    x1, y1, x2, y2 = first
    px1, py1, px2, py2 = second
    ix1, iy1 = max(x1, px1), max(y1, py1)
    ix2, iy2 = min(x2, px2), min(y2, py2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(1, (x2 - x1) * (y2 - y1))
    second_area = max(1, (px2 - px1) * (py2 - py1))
    return intersection / min(first_area, second_area)


def deduplicate_blur_regions(regions, overlap_threshold=0.48):
    kept = []
    for region in sorted(
        regions,
        key=lambda box: max(1, (box[2] - box[0]) * (box[3] - box[1]))
        * -1,
    ):
        if not any(
            overlap_ratio(region, existing) >= overlap_threshold
            for existing in kept
        ):
            kept.append(region)
    return kept


def partition_face_regions(detections, main_face):
    if main_face is None:
        return detections
    return [
        box for box in detections
        if overlap_ratio(main_face, box) < 0.40
    ]


def select_main_face(detections, tracker, frame_width):
    if not detections:
        return None
    if not tracker.tracks:
        return max(
            detections,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )

    predicted = tracker.tracks[0]["predicted"]
    predicted_cx = (predicted[0] + predicted[2]) / 2
    predicted_cy = (predicted[1] + predicted[3]) / 2

    def score(box):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        distance = ((cx - predicted_cx) ** 2 + (cy - predicted_cy) ** 2) ** 0.5
        normalized_distance = distance / max(frame_width, 1)
        return overlap_ratio(predicted, box) - normalized_distance

    return max(detections, key=score)


def find_largest_opening_face(capture, width, height, max_frames=90, step=3):
    largest_face = None
    largest_area = 0
    frame_id = 0

    while frame_id < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % step == 0:
            faces = detect_faces(frame)
            for x1, y1, x2, y2 in faces:
                pad_x = int((x2 - x1) * 0.10)
                pad_y = int((y2 - y1) * 0.16)
                box = [
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(width, x2 + pad_x),
                    min(height, y2 + pad_y),
                ]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > largest_area:
                    largest_area = area
                    largest_face = box
        frame_id += 1

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return largest_face


class RegionTracker:
    def __init__(self, max_missed=12, match_threshold=0.18, smooth=0.55):
        self.tracks = []
        self.next_id = 1
        self.max_missed = max_missed
        self.match_threshold = match_threshold
        self.smooth = smooth

    def reset(self):
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def iou(first, second):
        x1, y1, x2, y2 = first
        px1, py1, px2, py2 = second
        ix1, iy1 = max(x1, px1), max(y1, py1)
        ix2, iy2 = min(x2, px2), min(y2, py2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (x2 - x1) * (y2 - y1))
        previous_area = max(1, (px2 - px1) * (py2 - py1))
        return intersection / (area + previous_area - intersection)

    @staticmethod
    def center(box):
        return (
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
        )

    def update(self, detections):
        available = set(range(len(self.tracks)))
        matches = []

        for detection in detections:
            best_track = None
            best_score = self.match_threshold
            dcx, dcy = self.center(detection)
            for index in available:
                track = self.tracks[index]
                predicted = track["predicted"]
                score = self.iou(predicted, detection)
                tcx, tcy = self.center(predicted)
                distance = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                scale = max(1, (predicted[2] - predicted[0]) + (predicted[3] - predicted[1]))
                score -= distance / scale * 0.08
                if score >= best_score:
                    best_score = score
                    best_track = index

            if best_track is not None:
                available.remove(best_track)
                matches.append((best_track, detection))
            else:
                matches.append((None, detection))

        for index, detection in matches:
            if index is None:
                self.tracks.append({
                    "id": self.next_id,
                    "box": detection,
                    "predicted": detection,
                    "velocity": [0.0, 0.0],
                    "missed": 0,
                })
                self.next_id += 1
            else:
                track = self.tracks[index]
                previous = track["box"]
                vx = (detection[0] - previous[0]) * 0.6
                vy = (detection[1] - previous[1]) * 0.6
                box = [
                    int(previous[i] * (1 - self.smooth) + detection[i] * self.smooth)
                    for i in range(4)
                ]
                track["velocity"] = [
                    track["velocity"][0] * 0.45 + vx,
                    track["velocity"][1] * 0.45 + vy,
                ]
                track["box"] = box
                track["predicted"] = [
                    int(box[0] + track["velocity"][0]),
                    int(box[1] + track["velocity"][1]),
                    int(box[2] + track["velocity"][0]),
                    int(box[3] + track["velocity"][1]),
                ]
                track["missed"] = 0

        for index in available:
            track = self.tracks[index]
            box = track["predicted"]
            track["box"] = box
            track["predicted"] = [
                int(box[0] + track["velocity"][0]),
                int(box[1] + track["velocity"][1]),
                int(box[2] + track["velocity"][0]),
                int(box[3] + track["velocity"][1]),
            ]
            track["missed"] += 1

        self.tracks = [track for track in self.tracks if track["missed"] <= self.max_missed]

        return [track["predicted"].copy() for track in self.tracks]


face_tracker = RegionTracker(max_missed=4, match_threshold=0.12, smooth=0.45)
main_face_tracker = RegionTracker(max_missed=15, match_threshold=0.08, smooth=0.35)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_assets(filename):
    if filename.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    asset = FRONTEND_DIR / filename
    if asset.is_file():
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": "Not found"}), 404


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "face_detection": bool(face_detector is not None and not face_detector.empty()),
        "plate_detection": True,
    })


@app.post("/api/process")
def process_video():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "A video file is required"}), 400

    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return jsonify({"error": "Unsupported video format"}), 400

    job_id = uuid.uuid4().hex
    plate_detector.reset()
    face_tracker.reset()
    main_face_tracker.reset()
    input_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    output_name = f"{job_id}_protected.mp4"
    output_path = OUTPUT_DIR / output_name
    video.save(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not open the uploaded video"}), 400

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    locked_main_face = find_largest_opening_face(
        capture,
        width,
        height,
    )
    if locked_main_face is not None:
        main_face_tracker.update([locked_main_face])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not create the output video"}), 500

    started = time.perf_counter()
    processed_frames = 0
    detected_regions = 0
    frame_id = 0
    pii_regions_detected = 0
    plate_regions_detected = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        face_regions = detect_faces(frame)
        padded_face_regions = []
        for x1, y1, x2, y2 in face_regions:
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.16)
            padded_face_regions.append([
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width, x2 + pad_x),
                min(height, y2 + pad_y),
            ])
        main_face = select_main_face(
            padded_face_regions,
            main_face_tracker,
            width,
        )
        main_faces = main_face_tracker.update([main_face] if main_face else [])
        main_face = main_faces[0] if main_faces else None
        secondary_face_regions = partition_face_regions(
            padded_face_regions,
            main_face,
        )
        face_regions = face_tracker.update(secondary_face_regions)
        _, plate_regions = plate_detector.process_frame(frame, frame_id)
        plate_regions = [
            plate for plate in plate_regions
            if (main_face is None or overlap_ratio(main_face, plate) < 0.20)
            and not any(overlap_ratio(face, plate) >= 0.20 for face in face_regions)
        ]
        pii_regions = detect_pii_regions(frame, frame_id)
        regions = face_regions + plate_regions + pii_regions
        regions = deduplicate_blur_regions(regions)
        detected_regions += len(regions)
        plate_regions_detected += len(plate_regions)
        pii_regions_detected += len(pii_regions)
        processed = apply_blur_regions(frame, regions)
        writer.write(processed)
        processed_frames += 1
        frame_id += 1

    capture.release()
    writer.release()

    elapsed = time.perf_counter() - started
    original_url = f"/api/original/{input_path.name}"
    processed_url = f"/api/output/{output_name}"
    return jsonify({
        "job_id": job_id,
        "original_url": original_url,
        "processed_url": processed_url,
        "pipeline_summary": (
            f"Processed {processed_frames} frames: "
            f"{detected_regions} privacy regions, "
            f"{plate_regions_detected} plate detections, "
            f"{pii_regions_detected} PII text detections. "
            "The main person remains clear; secondary faces, plates, and sensitive text are blurred."
        ),
        "metrics": {
            "frames": processed_frames,
            "reported_frames": total_frames,
            "faces_and_plates_detected": detected_regions,
            "processing_seconds": round(elapsed, 2),
            "average_fps": round(processed_frames / elapsed, 2) if elapsed else 0,
            "resolution": f"{width}x{height}",
            "plate_detection_enabled": True,
        },
    })


@app.get("/api/original/<path:filename>")
def serve_original(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/api/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/api/download/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
