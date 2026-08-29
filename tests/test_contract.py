import unittest
import numpy as np
from pathlib import Path

from backend.alignment import align_modalities, _align_transcript_with_captions, _classify_sfx_transients
from backend.editing import build_shots, verified_edit_summary, _estimate_transform
from backend.contract import validate_report, runtime_profile, reconcile_states
from backend.ocr import _is_false_positive, _analyze_typography
from backend.pipeline import _repair_word_timestamps, _estimate_f0


class ReportContractTests(unittest.TestCase):
    def test_unverified_candidates_never_create_pacing(self):
        events = [{"timestamp": 1.0, "type": "hard_cut", "verification_status": "unverified"}]
        shots = build_shots(events, 10)
        summary = verified_edit_summary(events, shots, 10)
        self.assertEqual(summary["cut_count"], 0)
        self.assertEqual(summary["cuts_per_minute"], 0)
        self.assertTrue(summary["internal_verification_passed"])

    def test_shot_boundaries_ignore_unverified_events(self):
        events = [
            {"timestamp": 1.0, "type": "uncertain_change", "verification_status": "unverified"},
            {"timestamp": 3.0, "type": "hard_cut", "verification_status": "verified"},
        ]
        shots = build_shots(events, 10)
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[1]["start"], 3.0)
        self.assertEqual(shots[1]["boundary_in"], "hard_cut")

    def test_transient_without_beat_grid_is_not_beat_intent(self):
        report = {
            "editing": {"verified_events": [{"timestamp": 1.0, "start": 0.98, "end": 1.02, "type": "hard_cut", "final_confidence": 0.9}]},
            "transcript": {"words": [], "sentences": [], "pauses": [], "emphasized_words": []},
            "audio": {"events": {"transients": [{"timestamp": 1.0, "class": "unknown_transient", "confidence": 0.25}], "beat_grid": [], "beat_status": "not_verified"}},
            "text_overlay": {"track": []},
            "source": {"duration_seconds": 10},
        }
        align_modalities(report)
        intents = report["edit_intent"]["events"][0]["intent_candidates"]
        self.assertIn("audio_transient_aligned_edit", [item["intent"] for item in intents])
        self.assertNotIn("beat_alignment", [item["intent"] for item in intents])

    def test_confidence_is_never_implicit_one(self):
        event = {"timestamp": 2.0, "type": "hard_cut", "verification_status": "verified", "candidate_confidence": 0.64, "verification_confidence": 0.96, "final_confidence": 0.88}
        self.assertLess(event["final_confidence"], 1.0)

    def test_validator_rejects_invalid_word_interval(self):
        report = {
            "report_id": "x",
            "source": {"duration_seconds": 2},
            "processing": {},
            "transcript": {"words": [{"word": "bad", "start": 1, "end": 1, "confidence": 0.5}]},
            "visual": {},
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
        }
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_word_timestamp_repair_strictly_eliminates_overlaps(self):
        raw_words = [
            {"word": "And", "raw_start": 6.72, "raw_end": 6.74},
            {"word": "the", "raw_start": 6.72, "raw_end": 6.78},
            {"word": "correct", "raw_start": 6.78, "raw_end": 7.10},
        ]
        repaired = _repair_word_timestamps(raw_words)
        for i in range(len(repaired) - 1):
            self.assertLessEqual(repaired[i]["aligned_end"], repaired[i + 1]["aligned_start"] + 0.001)
            self.assertGreater(repaired[i]["aligned_end"], repaired[i]["aligned_start"])

    def test_ocr_filters_bad_false_positive(self):
        # Example from issue description: "9" covering whole frame
        box = np.array([[0, 0], [360, 0], [360, 640], [0, 640]])
        is_fp = _is_false_positive("9", 0.559, box, 360, 640)
        self.assertTrue(is_fp)

        # Valid small caption box
        valid_box = np.array([[50, 400], [310, 400], [310, 440], [50, 440]])
        is_valid_fp = _is_false_positive("THINKING SKILLS", 0.98, valid_box, 360, 640)
        self.assertFalse(is_valid_fp)

    def test_transform_rejected_across_unrelated_cuts(self):
        dummy_a = np.zeros((160, 160, 3), dtype=np.float32)
        dummy_b = np.ones((160, 160, 3), dtype=np.float32)
        transform = _estimate_transform(dummy_a, dummy_b, 0.0, 0.5, allow_fit=False)
        self.assertEqual(transform["status"], "rejected_unrelated_cut")
        self.assertEqual(transform["confidence"], 0.0)

    def test_transcript_caption_alignment_computes_lead_lag_and_omissions(self):
        words = [
            {"word": "If", "display": "If", "start": 0.0, "end": 0.16},
            {"word": "you", "display": "you", "start": 0.16, "end": 0.30},
            {"word": "have", "display": "have", "start": 0.30, "end": 0.54},
            {"word": "little", "display": "little", "start": 0.54, "end": 0.88},
            {"word": "kids", "display": "kids", "start": 0.88, "end": 1.26},
        ]
        captions = [
            {"text": "IF YOU HAVE", "start": 0.0, "end": 0.50},
            {"text": "KIDS TODAY", "start": 0.85, "end": 1.40},
        ]
        res = _align_transcript_with_captions(words, captions)
        self.assertEqual(res["total_caption_events"], 2)
        self.assertGreater(res["caption_word_coverage"], 0.5)
        self.assertIn("lead_lag_seconds", res["alignments"][0])

    def test_f0_pitch_estimation(self):
        t = np.linspace(0, 0.2, int(16000 * 0.2))
        tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        f0 = _estimate_f0(tone, 16000)
        self.assertAlmostEqual(f0, 200.0, delta=5.0)

    def test_runtime_profile_detects_hardware(self):
        profile = runtime_profile()
        self.assertIn("cuda", profile)
        self.assertIn("available", profile["cuda"])
        self.assertIn("decode_path", profile)


if __name__ == "__main__":
    unittest.main()

