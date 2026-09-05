"""Regression: summary.cut_count and the contract validator must agree.

The producer (editing.verified_edit_summary) counted three cut subtypes --
hard_cut, jump_cut and scene_change -- while the contract validator counted only
two, silently dropping scene_change. Any real video containing a scene_change
therefore produced a report that was internally consistent but failed
validation and was quarantined.

The detector emits every boundary as hard_cut and then refines the subtype in
place, so all three are cuts by construction; the validator's narrower set was
the defect. Both now derive from one canonical collection.
"""
import copy
import unittest

from backend.contract import CUT_EVENT_TYPES, validate_report, verified_cut_events
from backend.editing import verified_edit_summary


def _event(timestamp, kind="hard_cut", verified=True):
    return {
        "timestamp": timestamp, "start": timestamp - 0.02, "end": timestamp + 0.02,
        "type": kind,
        "verification_status": "verified" if verified else "provisional",
        "candidate_confidence": 0.9, "verification_confidence": 0.9,
        "final_confidence": 0.9, "training_eligible": True,
    }


def _shots(count, duration):
    step = duration / max(count, 1)
    return [{"shot_id": i, "start": i * step, "end": (i + 1) * step,
             "duration": step, "content_class": "talking_head"}
            for i in range(count)]


def _report(events, duration=96.791):
    """Smallest report the contract accepts, carrying a real edit summary.

    Everything outside `editing` is an empty stub: this exercises the
    edit-summary invariant without pulling in a 187 KB fixture.
    """
    summary = verified_edit_summary(events, _shots(len(events) + 1, duration), duration)
    return {
        "report_id": "test", "source": {"duration_seconds": duration, "fps": 24},
        "processing": {}, "transcript": {}, "visual": {}, "color": {}, "audio": {},
        "text_overlay": {}, "semantic": {}, "edit_intent": {},
        "cross_modal_events": [], "training_features": {}, "confidence": {},
        "deferred": {},
        "editing": {"verified_events": events, "summary": summary,
                    "candidate_regions": [], "transforms": []},
    }


class CutSummaryInvariantTests(unittest.TestCase):
    """The invariant: summary.cut_count == number of verified events whose type
    is one of the cut subtypes. Integer equality, no tolerance."""

    def test_canonical_cut_types_cover_every_detector_subtype(self):
        self.assertEqual(CUT_EVENT_TYPES,
                         frozenset({"hard_cut", "jump_cut", "scene_change"}))

    def test_the_real_failure_shape_scene_change_is_counted(self):
        """Exactly the production mismatch: 33 hard + 20 jump + 1 scene = 54,
        which the old validator counted as 53."""
        events = ([_event(1.0 + i, "hard_cut") for i in range(33)]
                  + [_event(40.0 + i, "jump_cut") for i in range(20)]
                  + [_event(96.7292, "scene_change")])
        report = _report(events)
        self.assertEqual(report["editing"]["summary"]["cut_count"], 54)
        self.assertEqual(len(verified_cut_events(events)), 54)
        validate_report(report)   # must not raise

    def test_producer_and_validator_agree_for_every_subtype_alone(self):
        for kind in sorted(CUT_EVENT_TYPES):
            with self.subTest(kind=kind):
                events = [_event(1.0, kind), _event(2.0, kind)]
                report = _report(events)
                self.assertEqual(report["editing"]["summary"]["cut_count"], 2)
                validate_report(report)

    def test_no_cut_video(self):
        report = _report([])
        self.assertEqual(report["editing"]["summary"]["cut_count"], 0)
        validate_report(report)

    def test_one_cut_video(self):
        report = _report([_event(48.0)])
        self.assertEqual(report["editing"]["summary"]["cut_count"], 1)
        validate_report(report)

    def test_boundary_cuts_at_first_and_final_frame_are_counted(self):
        duration = 10.0
        events = [_event(0.0417, "hard_cut"), _event(5.0, "jump_cut"),
                  _event(9.9583, "scene_change")]
        report = _report(events, duration)
        self.assertEqual(report["editing"]["summary"]["cut_count"], 3)
        validate_report(report)

    def test_duplicate_timestamps_are_counted_as_distinct_events(self):
        """Deduplication is not part of this invariant: both sides count the
        same list, so they must agree even when timestamps repeat."""
        events = [_event(5.0, "hard_cut"), _event(5.0, "hard_cut")]
        report = _report(events)
        self.assertEqual(report["editing"]["summary"]["cut_count"], 2)
        validate_report(report)

    def test_non_integer_frame_rate_timestamps(self):
        events = [_event(round(i / 29.97, 4), "hard_cut") for i in range(1, 6)]
        report = _report(events, duration=12.0)
        self.assertEqual(report["editing"]["summary"]["cut_count"], 5)
        validate_report(report)

    def test_unverified_events_are_excluded_from_the_canonical_count(self):
        events = [_event(1.0), _event(2.0, verified=False)]
        self.assertEqual(len(verified_cut_events(events)), 1)

    def test_a_genuinely_inconsistent_summary_is_still_quarantined(self):
        """Strictness must survive the repair."""
        report = _report([_event(1.0), _event(2.0)])
        report["editing"]["summary"]["cut_count"] = 99
        with self.assertRaises(ValueError) as caught:
            validate_report(report)
        self.assertIn("cut summary disagrees", str(caught.exception))

    def test_dropping_a_scene_change_from_the_summary_is_quarantined(self):
        """The inverse of the bug: if a producer ever undercounts, catch it."""
        events = [_event(1.0, "hard_cut"), _event(2.0, "scene_change")]
        report = _report(events)
        report["editing"]["summary"]["cut_count"] = 1
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_cuts_per_minute_derives_from_the_same_count(self):
        """Internal coherence: the old 2-type reading would have made
        cut_count and cuts_per_minute disagree inside one summary."""
        duration = 60.0
        events = [_event(1.0, "hard_cut"), _event(2.0, "jump_cut"),
                  _event(3.0, "scene_change")]
        summary = _report(events, duration)["editing"]["summary"]
        self.assertEqual(summary["cut_count"], 3)
        self.assertAlmostEqual(summary["cuts_per_minute"],
                               round(3 * 60 / duration, 2))


if __name__ == "__main__":
    unittest.main()
