import os
from pathlib import Path
import cv2


class PlateDetector:
    def __init__(self, weights_path=None, conf_thresh=0.25):
        self.weights_path = weights_path or os.getenv("PLATE_MODEL_PATH")
        self.conf_thresh = conf_thresh
        self.model = None
        self.previous_regions = []
        self.missed_frames = 0
        self.max_missed_frames = 5
        if self.weights_path and Path(self.weights_path).is_file():
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
            except Exception:
                self.model = None

    def fallback_regions(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []
        frame_area = height * width
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 70 or box_height < 16:
                continue

            aspect = box_width / float(box_height)
            area = box_width * box_height
            if not 2.8 <= aspect <= 6.5:
                continue
            if not 0.0005 * frame_area <= area <= 0.07 * frame_area:
                continue
            if box_width > width * 0.45 or box_height > height * 0.20:
                continue

            contour_area = cv2.contourArea(contour)
            if contour_area / max(area, 1) < 0.12:
                continue

            roi = gray[y:y + box_height, x:x + box_width]
            if roi.size == 0:
                continue

            roi_std = float(cv2.meanStdDev(roi)[1][0][0])
            if roi_std < 8:
                continue

            _, threshold = cv2.threshold(
                roi,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            text_contours, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            text_components = 0
            for text_contour in text_contours:
                _, _, text_width, text_height = cv2.boundingRect(text_contour)
                if (
                    3 <= text_width <= box_width * 0.22
                    and 5 <= text_height <= box_height * 0.82
                ):
                    text_components += 1
            if text_components < 2:
                continue

            padding_x = int(box_width * 0.03)
            padding_y = int(box_height * 0.08)
            regions.append([
                max(0, x - padding_x),
                max(0, y - padding_y),
                min(width, x + box_width + padding_x),
                min(height, y + box_height + padding_y),
            ])

        regions.sort(key=lambda region: (
            (region[2] - region[0]) * (region[3] - region[1])
        ), reverse=True)
        filtered = []
        for region in regions:
            duplicate = False
            x1, y1, x2, y2 = region
            for kept in filtered:
                px1, py1, px2, py2 = kept
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = max(1, (x2 - x1) * (y2 - y1))
                kept_area = max(1, (px2 - px1) * (py2 - py1))
                iou = intersection / (area + kept_area - intersection)
                if iou > 0.25:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(region)
        return filtered[:2]

    def reset(self):
        self.previous_regions = []
        self.missed_frames = 0

    def _refine_plate_region(self, frame, region):
        x1, y1, x2, y2 = region
        roi_width = x2 - x1
        roi_height = y2 - y1
        if roi_width < 70 or roi_height < 16:
            return region

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return region

        _, threshold = cv2.threshold(
            roi,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        text_contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        components = []
        for contour in text_contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if (
                3 <= cw <= roi_width * 0.22
                and 5 <= ch <= roi_height * 0.82
            ):
                components.append((cx, cy, cw, ch))

        if len(components) < 2:
            return region

        text_x1 = min(component[0] for component in components)
        text_y1 = min(component[1] for component in components)
        text_x2 = max(
            component[0] + component[2]
            for component in components
        )
        text_y2 = max(
            component[1] + component[3]
            for component in components
        )

        padding_x = max(3, int(roi_width * 0.025))
        padding_y = max(3, int(roi_height * 0.10))
        refined_x1 = max(0, x1 + text_x1 - padding_x)
        refined_y1 = max(0, y1 + text_y1 - padding_y)
        refined_x2 = min(frame.shape[1], x1 + text_x2 + padding_x)
        refined_y2 = min(frame.shape[0], y1 + text_y2 + padding_y)

        refined_width = refined_x2 - refined_x1
        refined_height = refined_y2 - refined_y1
        if refined_width < 70 or refined_height < 16:
            return region
        return [refined_x1, refined_y1, refined_x2, refined_y2]

    def _valid_plate_region(self, frame, region):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(value))
            for value in region
        )
        x1, y1 = min(x1, width), min(y1, height)
        x2, y2 = min(x2, width), min(y2, height)
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        if box_width < 70 or box_height < 16:
            return None

        aspect = box_width / float(box_height)
        area = box_width * box_height
        frame_area = width * height
        if not 2.8 <= aspect <= 6.5:
            return None
        if not 0.0005 * frame_area <= area <= 0.07 * frame_area:
            return None
        if box_width > width * 0.45 or box_height > height * 0.20:
            return None
        return [x1, y1, x2, y2]

    def _normalize_regions(self, frame, regions):
        normalized = []
        frame_area = frame.shape[0] * frame.shape[1]
        for region in regions:
            box = self._valid_plate_region(frame, region)
            if box is None:
                continue
            box = self._refine_plate_region(frame, box)
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area <= frame_area * 0.07:
                normalized.append(box)

        normalized.sort(
            key=lambda region: (
                (region[2] - region[0]) * (region[3] - region[1])
            ),
            reverse=True,
        )
        filtered = []
        for region in normalized:
            duplicate = False
            x1, y1, x2, y2 = region
            for kept in filtered:
                px1, py1, px2, py2 = kept
                ix1 = max(x1, px1)
                iy1 = max(y1, py1)
                ix2 = min(x2, px2)
                iy2 = min(y2, py2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                region_area = max(1, (x2 - x1) * (y2 - y1))
                kept_area = max(1, (px2 - px1) * (py2 - py1))
                iou = intersection / (region_area + kept_area - intersection)
                if iou > 0.25:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(region)
        return filtered[:3]

    def process_frame(self, frame, frame_id):
        if self.model is None:
            regions = self._normalize_regions(
                frame,
                self.fallback_regions(frame),
            )
        else:
            regions = self._yolo_regions(frame)

        if regions:
            regions = self._smooth_regions(regions)
            self.previous_regions = regions
            self.missed_frames = 0
        elif self.previous_regions and self.missed_frames < self.max_missed_frames:
            regions = self.previous_regions
            self.missed_frames += 1
        else:
            regions = []
            self.previous_regions = []

        return frame_id, regions

    def _yolo_regions(self, frame):
        if self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False,
        )
        rectangles = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                rectangles.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ])
        return self._normalize_regions(frame, rectangles)

    def _smooth_regions(self, regions):
        if not self.previous_regions:
            return regions

        matched = []
        unmatched_previous = list(self.previous_regions)
        for region in regions:
            best_previous = None
            best_iou = 0.20
            x1, y1, x2, y2 = region
            for previous in self.previous_regions:
                px1, py1, px2, py2 = previous
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = max(1, (x2 - x1) * (y2 - y1))
                previous_area = max(1, (px2 - px1) * (py2 - py1))
                iou = intersection / (area + previous_area - intersection)
                if iou >= best_iou:
                    best_iou = iou
                    best_previous = previous

            if best_previous is not None:
                nx1 = int((x1 + best_previous[0]) // 2)
                ny1 = int((y1 + best_previous[1]) // 2)
                nx2 = int((x2 + best_previous[2]) // 2)
                ny2 = int((y2 + best_previous[3]) // 2)
                matched.append([nx1, ny1, nx2, ny2])
                if best_previous in unmatched_previous:
                    unmatched_previous.remove(best_previous)
            else:
                matched.append(region)

        for previous in unmatched_previous:
            if self.missed_frames < self.max_missed_frames:
                matched.append(previous)
        return matched[:3]
