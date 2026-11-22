import cv2
import numpy as np


def clamp_region(frame_shape, region):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (int(value) for value in region)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def apply_blur_regions(frame, rectangles, kernel_size=35):
    result = frame.copy()
    odd_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
    for region in rectangles:
        box = clamp_region(frame.shape, region)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        region_width = x2 - x1
        region_height = y2 - y1
        region_area = region_width * region_height
        if (
            region_area >= frame_area * 0.55
            or region_width >= frame_width * 0.85
            or region_height >= frame_height * 0.85
        ):
            continue
        roi = result[y1:y2, x1:x2]
        if roi.size:
            result[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi, (odd_kernel, odd_kernel), 0
            )
    return result
