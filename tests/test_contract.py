import unittest

from backend.alignment import align_modalities
from backend.editing import build_shots, verified_edit_summary


class ReportContractTests(unittest.TestCase):
    def test_unverified_candidates_never_create_pacing(self):
        events=[{"timestamp":1.0,"type":"hard_cut","verification_status":"unverified"}]
        shots=build_shots(events,10)
        summary=verified_edit_summary(events,shots,10)
        self.assertEqual(summary["cut_count"],0)
        self.assertEqual(summary["cuts_per_minute"],0)

    def test_shot_boundaries_ignore_unverified_events(self):
        events=[
            {"timestamp":1.0,"type":"uncertain_change","verification_status":"unverified"},
            {"timestamp":3.0,"type":"hard_cut","verification_status":"verified"},
        ]
        shots=build_shots(events,10)
        self.assertEqual(len(shots),2)
        self.assertEqual(shots[1]["start"],3.0)
        self.assertEqual(shots[1]["boundary_in"],"hard_cut")

    def test_transient_without_beat_grid_is_not_beat_intent(self):
        report={"editing":{"verified_events":[{"timestamp":1.0,"start":.98,"end":1.02,"type":"hard_cut","final_confidence":.9}]},"transcript":{"words":[],"sentences":[],"pauses":[],"emphasized_words":[]},"audio":{"events":{"transients":[{"timestamp":1.0,"class":"unknown_transient","confidence":.25}],"beat_grid":[],"beat_status":"not_verified"}},"text_overlay":{"track":[]}}
        align_modalities(report)
        intents=report["edit_intent"]["events"][0]["intent_candidates"]
        self.assertIn("audio_transient_aligned_edit",[item["intent"] for item in intents])
        self.assertNotIn("beat_alignment",[item["intent"] for item in intents])

    def test_confidence_is_never_implicit_one(self):
        event={"timestamp":2.0,"type":"hard_cut","verification_status":"verified","candidate_confidence":.64,"verification_confidence":.96,"final_confidence":.88}
        self.assertLess(event["final_confidence"],1)


if __name__=="__main__":
    unittest.main()
