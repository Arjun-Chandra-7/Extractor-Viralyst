"""Comprehensive Release Candidate Verification Tests.

Tests:
1. Synthetic freeze-frame tests (A-F)
2. Caption-transcript monotonic sequence alignment
3. Caption emphasis separation
4. Video-level typography clustering
5. Automatic report validation & training safety firewalls
"""
from __future__ import annotations

import unittest
import numpy as np

from backend.editing import DenseFrame, _detect_speed_effects
from backend.alignment import _align_transcript_with_captions
from backend.contract import validate_report, runtime_profile


class TestFreezeFrameDetector(unittest.TestCase):
    def _create_frame(self, t: float, rgb_val: float, face_pos: tuple[float, float] | None = None, flow: float = 0.0) -> DenseFrame:
        arr = np.full((16, 16, 3), rgb_val, dtype=np.float32)
        lum = np.full((16, 16), rgb_val, dtype=np.float32)
        edge = np.zeros((16, 16), dtype=np.float32)
        subjects = []
        if face_pos is not None:
            subjects = [{"center_x": face_pos[0], "center_y": face_pos[1], "box_normalized": [0.4, 0.4, 0.2, 0.2]}]
        frame = DenseFrame(time=t, rgb=arr, luminance=lum, edge=edge, subjects=subjects)
        setattr(frame, "flow_mag", flow)
        return frame

    def test_case_a_static_held_image_for_2s_detects_freeze(self):
        # Case A: 20 consecutive identical frames across 2.0s with zero motion
        frames = [self._create_frame(i * 0.1, 0.50, (0.5, 0.5), flow=0.0) for i in range(20)]
        effects = _detect_speed_effects(frames, fps=10.0)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0]["type"], "freeze_frame")
        self.assertEqual(effects[0]["verification_status"], "verified")
        self.assertTrue(effects[0]["training_eligible"])
        self.assertAlmostEqual(effects[0]["duration"], 1.9, delta=0.15)

    def test_case_b_talking_head_with_subtle_facial_movement_no_freeze(self):
        # Case B: Talking head where subject moves slightly (jitter in coordinates > 0.005)
        frames = [
            self._create_frame(i * 0.1, 0.50 + (i % 2) * 0.002, (0.5 + (i * 0.006), 0.5), flow=0.02)
            for i in range(20)
        ]
        effects = _detect_speed_effects(frames, fps=10.0)
        self.assertEqual(effects, [])

    def test_case_c_static_background_with_mouth_movement_no_freeze(self):
        # Case C: Static background but facial pos delta or flow magnitude present
        frames = [
            self._create_frame(i * 0.1, 0.50 + (0.001 if i % 3 == 0 else 0.0), (0.5, 0.50 + (0.006 if i % 2 == 0 else 0.0)), flow=0.015)
            for i in range(20)
        ]
        effects = _detect_speed_effects(frames, fps=10.0)
        self.assertEqual(effects, [])

    def test_case_d_repeated_exact_frame_sequence_detects_freeze(self):
        # Case D: Repeated exact duplicate frame sequence for 1.5s
        frames = [self._create_frame(i * 0.1, 0.42, (0.5, 0.5), flow=0.0) for i in range(15)]
        effects = _detect_speed_effects(frames, fps=10.0)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0]["verification_status"], "verified")
        self.assertTrue(effects[0]["training_eligible"])

    def test_case_e_very_low_motion_scene_candidate_or_none_not_verified(self):
        # Case E: Low motion scene with small lighting/camera drift
        frames = [
            self._create_frame(i * 0.1, 0.50 + (i * 0.0005), (0.5, 0.5), flow=0.002)
            for i in range(15)
        ]
        effects = _detect_speed_effects(frames, fps=10.0)
        # Must NOT be confidently verified
        for eff in effects:
            self.assertNotEqual(eff.get("verification_status"), "verified")
            self.assertFalse(eff.get("training_eligible"))

    def test_case_f_video_freeze_video_detects_exact_interval(self):
        # Case F: Normal motion (0-1s) -> Freeze (1-2.5s) -> Normal motion (2.5-3.5s)
        frames = []
        # Normal motion
        for i in range(10):
            frames.append(self._create_frame(i * 0.1, 0.20 + (i * 0.01), (0.5 + (i * 0.01), 0.5), flow=0.05))
        # Freeze interval (1.0s to 2.5s = 15 frames)
        for i in range(10, 25):
            frames.append(self._create_frame(i * 0.1, 0.77, (0.5, 0.5), flow=0.0))
        # Normal motion
        for i in range(25, 35):
            frames.append(self._create_frame(i * 0.1, 0.30 + (i * 0.01), (0.5 + (i * 0.01), 0.5), flow=0.05))

        effects = _detect_speed_effects(frames, fps=10.0)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0]["verification_status"], "verified")
        self.assertAlmostEqual(effects[0]["start"], 1.0, delta=0.15)
        self.assertAlmostEqual(effects[0]["end"], 2.4, delta=0.15)


class TestCaptionAlignment(unittest.TestCase):
    def test_local_monotonic_sequence_matching(self):
        # Transcript: "I said it is because I think it helps"
        words = [
            {"word": "I", "display": "I", "start": 0.5, "end": 0.7},
            {"word": "said", "display": "said", "start": 0.75, "end": 1.0},
            {"word": "it", "display": "it", "start": 1.05, "end": 1.25},
            {"word": "is", "display": "is", "start": 1.4, "end": 1.6},
            {"word": "because", "display": "because", "start": 1.65, "end": 2.0},
            {"word": "I", "display": "I", "start": 2.05, "end": 2.2},
            {"word": "think", "display": "think", "start": 2.3, "end": 2.6},
            {"word": "it", "display": "it", "start": 2.65, "end": 2.85},
            {"word": "helps", "display": "helps", "start": 2.9, "end": 3.2},
        ]
        captions = [
            {"text": "I SAID IT", "start": 0.5, "end": 1.3},
            {"text": "IS BECAUSE I", "start": 1.35, "end": 2.25},
            {"text": "THINK IT HELPS", "start": 2.3, "end": 3.25},
        ]

        result = _align_transcript_with_captions(words, captions)
        alignments = result["alignments"]
        self.assertEqual(len(alignments), 3)

        # 1. First caption: indices [0, 1, 2]
        self.assertEqual([r["index"] for r in alignments[0]["spoken_word_refs"]], [0, 1, 2])
        self.assertEqual(alignments[0]["match_score"], 1.0)
        self.assertTrue(alignments[0]["monotonicity_passed"])
        self.assertEqual(alignments[0]["reused_word_count"], 0)
        self.assertTrue(alignments[0]["training_eligible"])

        # 2. Second caption: indices [3, 4, 5] (must NOT grab index 0 "I" or index 2 "it")
        self.assertEqual([r["index"] for r in alignments[1]["spoken_word_refs"]], [3, 4, 5])
        self.assertEqual(alignments[1]["match_score"], 1.0)
        self.assertTrue(alignments[1]["monotonicity_passed"])
        self.assertEqual(alignments[1]["reused_word_count"], 0)
        self.assertTrue(alignments[1]["training_eligible"])

        # 3. Third caption: indices [6, 7, 8]
        self.assertEqual([r["index"] for r in alignments[2]["spoken_word_refs"]], [6, 7, 8])
        self.assertEqual(alignments[2]["match_score"], 1.0)
        self.assertTrue(alignments[2]["monotonicity_passed"])
        self.assertEqual(alignments[2]["reused_word_count"], 0)
        self.assertTrue(alignments[2]["training_eligible"])

    def test_repeated_common_tokens_monotonicity(self):
        # "I think I said I think"
        words = [
            {"word": "I", "display": "I", "start": 0.2, "end": 0.4},
            {"word": "think", "display": "think", "start": 0.45, "end": 0.8},
            {"word": "I", "display": "I", "start": 0.9, "end": 1.1},
            {"word": "said", "display": "said", "start": 1.15, "end": 1.4},
            {"word": "I", "display": "I", "start": 1.5, "end": 1.7},
            {"word": "think", "display": "think", "start": 1.75, "end": 2.1},
        ]
        captions = [
            {"text": "I THINK", "start": 0.2, "end": 0.85},
            {"text": "I SAID", "start": 0.9, "end": 1.45},
            {"text": "I THINK", "start": 1.5, "end": 2.15},
        ]
        result = _align_transcript_with_captions(words, captions)
        alignments = result["alignments"]

        self.assertEqual([r["index"] for r in alignments[0]["spoken_word_refs"]], [0, 1])
        self.assertEqual([r["index"] for r in alignments[1]["spoken_word_refs"]], [2, 3])
        self.assertEqual([r["index"] for r in alignments[2]["spoken_word_refs"]], [4, 5])
        for a in alignments:
            self.assertTrue(0.0 <= a["match_score"] <= 1.0)
            self.assertTrue(a["monotonicity_passed"])
            self.assertEqual(a["reused_word_count"], 0)


class TestCaptionEmphasisSemantics(unittest.TestCase):
    def test_uniform_caption_has_no_emphasized_words(self):
        words = [{"word": "hello", "display": "Hello", "start": 0.1, "end": 0.5}]
        captions = [
            {
                "text": "HELLO WORLD",
                "start": 0.1,
                "end": 0.9,
                "word_highlighting": {"status": "uniform_chunk", "highlighted_words": []},
            }
        ]
        res = _align_transcript_with_captions(words, captions)
        a = res["alignments"][0]
        self.assertEqual(a["displayed_words"], ["HELLO", "WORLD"])
        self.assertEqual(a["highlighted_words"], [])
        self.assertEqual(a["emphasized_displayed_words"], [])

    def test_distinct_highlighted_word_is_captured(self):
        words = [{"word": "important", "display": "important", "start": 0.1, "end": 0.5}]
        captions = [
            {
                "text": "THIS IS IMPORTANT",
                "start": 0.1,
                "end": 0.9,
                "word_highlighting": {"status": "detected", "highlighted_words": ["IMPORTANT"]},
            }
        ]
        res = _align_transcript_with_captions(words, captions)
        a = res["alignments"][0]
        self.assertEqual(a["displayed_words"], ["THIS", "IS", "IMPORTANT"])
        self.assertEqual(a["highlighted_words"], ["IMPORTANT"])
        self.assertEqual(a["emphasized_displayed_words"], ["IMPORTANT"])


class TestAutomaticReportValidation(unittest.TestCase):
    def test_valid_report_passes(self):
        valid_rep = {
            "report_id": "test-123",
            "source": {"duration_seconds": 10.0, "fps": 30.0},
            "processing": {"status": "complete"},
            "transcript": {
                "words": [
                    {"word": "hi", "display": "Hi", "start": 0.0, "end": 0.5, "confidence": 0.95}
                ]
            },
            "visual": {
                "shots": [{"start": 0.0, "end": 10.0}],
                "speed_effects": [],
            },
            "color": {},
            "audio": {},
            "text_overlay": {},
            "editing": {"verified_events": [], "summary": {}},
            "semantic": {},
            "edit_intent": {},
            "cross_modal_events": [],
            "training_features": {},
            "confidence": {},
            "deferred": [],
            "transcript_caption_alignment": {
                "alignments": [
                    {"match_score": 0.95, "spoken_word_refs": [{"index": 0}], "verification_status": "verified", "training_eligible": True}
                ]
            },
        }
        # Should execute cleanly without error
        validate_report(valid_rep)

    def test_out_of_range_match_score_fails_validation(self):
        invalid_rep = {
            "report_id": "test-123",
            "source": {"duration_seconds": 10.0, "fps": 30.0},
            "processing": {"status": "complete"},
            "transcript": {"words": []},
            "visual": {"shots": [], "speed_effects": []},
            "color": {},
            "audio": {},
            "text_overlay": {},
            "editing": {"verified_events": [], "summary": {}},
            "semantic": {},
            "edit_intent": {},
            "cross_modal_events": [],
            "training_features": {},
            "confidence": {},
            "deferred": [],
            "transcript_caption_alignment": {
                "alignments": [{"match_score": 1.333, "spoken_word_refs": []}]
            },
        }
        with self.assertRaises(ValueError):
            validate_report(invalid_rep)


if __name__ == "__main__":
    unittest.main()
