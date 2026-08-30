import unittest
import numpy as np

from backend.alignment import _align_transcript_with_captions
from backend.corpus import _extract_audio_grading_features, _measure_representative_color


class FinalCorpusCorrectnessTests(unittest.TestCase):
    def test_48khz_stereo_frequency_and_width(self):
        sr=48000; t=np.arange(sr,dtype=np.float32)/sr
        left=np.sin(2*np.pi*1000*t); right=np.sin(2*np.pi*1000*t+.5)
        result=_extract_audio_grading_features(np.stack([left,right]),sr,44100)
        self.assertEqual(result["source_sample_rate"],44100)
        self.assertEqual(result["analysis_sample_rate"],48000)
        self.assertAlmostEqual(result["spectral_centroid_hz"],1000,delta=25)
        self.assertIsNotNone(result["stereo_side_to_mid_ratio"])

    def test_pixel_color_statistics_not_frame_means(self):
        frame=np.zeros((100,100,3),np.uint8); frame[:,50:]=255
        result=_measure_representative_color([frame])
        self.assertAlmostEqual(result["luminance_mean"],.5,delta=.01)
        self.assertLess(result["luminance_p05"],.01); self.assertGreater(result["luminance_p95"],.99)
        self.assertAlmostEqual(result["dark_pixel_fraction"],.5,delta=.01)
        self.assertAlmostEqual(result["bright_pixel_fraction"],.5,delta=.01)
        self.assertGreater(result["contrast_proxy"],.9)

    def test_caption_provenance_and_temporal_safety(self):
        words=[{"word":"There's","display":"There's","start":1.0,"end":1.2}]
        captions=[{"text":"THERE'S","start":1.8,"end":2.0,"text_source":"transcript_assisted","word_highlighting":{}}]
        aligned=_align_transcript_with_captions(words,captions)["alignments"][0]
        self.assertEqual(captions[0]["text_source"],"transcript_assisted")
        self.assertEqual(aligned["match_score"],1.0)
        self.assertFalse(aligned["training_eligible"])


if __name__=="__main__": unittest.main()
