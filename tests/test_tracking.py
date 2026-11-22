import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from backend.app import (
    RegionTracker,
    is_visible_face,
    overlap_ratio,
    select_main_face,
    deduplicate_blur_regions,
    partition_face_regions,
)


class RegionTrackerTest(unittest.TestCase):
    def test_face_detection_cascade_is_loaded(self):
        from backend.app import face_detector

        self.assertIsNotNone(face_detector)
        self.assertFalse(face_detector.empty())

    def test_profile_face_detection_cascade_is_loaded(self):
        from backend.app import profile_detector

        self.assertIsNotNone(profile_detector)
        self.assertFalse(profile_detector.empty())

    def test_partition_blurs_only_secondary_faces(self):
        main_face = [100, 100, 220, 220]
        second_face = [700, 120, 820, 240]
        secondary = partition_face_regions(
            [main_face, second_face],
            main_face,
        )
        self.assertEqual(secondary, [second_face])

    def test_deduplicates_overlapping_extra_blur_regions(self):
        regions = deduplicate_blur_regions([
            [100, 100, 200, 200],
            [105, 105, 205, 205],
            [500, 100, 600, 200],
        ])

        self.assertEqual(len(regions), 2)
    def test_main_face_remains_locked_to_nearest_previous_face(self):
        tracker = RegionTracker(max_missed=10, match_threshold=0.12, smooth=0.45)
        tracker.update([[100, 100, 220, 220]])

        main_face = select_main_face(
            [[115, 105, 225, 215], [700, 120, 790, 210]],
            tracker,
            1280,
        )

        self.assertLess(main_face[0], 300)
    def test_visible_face_filter_rejects_partial_or_tiny_faces(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.assertTrue(is_visible_face(frame, [500, 150, 650, 300]))
        self.assertFalse(is_visible_face(frame, [10, 10, 30, 30]))
        self.assertFalse(is_visible_face(frame, [500, 150, 700, 210]))
    def test_excludes_overlapping_main_face_from_blur_tracks(self):
        main_face = [100, 100, 200, 220]
        other_face = [300, 110, 390, 220]
        detections = [
            box for box in [main_face, other_face]
            if overlap_ratio(main_face, box) < 0.40
        ]

        self.assertEqual(detections, [other_face])
    def test_keeps_second_face_track_when_one_detection_is_missing(self):
        tracker = RegionTracker(max_missed=10, match_threshold=0.12, smooth=0.45)
        first = tracker.update([[10, 10, 80, 80], [300, 20, 370, 90]])
        second = tracker.update([[12, 12, 82, 82]])

        self.assertEqual(len(first), 2)
        self.assertGreaterEqual(len(second), 1)
        self.assertTrue(any(box[0] >= 290 for box in second))


if __name__ == "__main__":
    unittest.main()
