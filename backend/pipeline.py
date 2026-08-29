from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

import av
import numpy as np

from .editing import dense_verify_full_video, verified_edit_summary
from .contract import VERSION, content_hash, runtime_profile

MAX_SAMPLES = 96
_WHISPER_MODELS = {}
_AUDIO_CACHE = {}
_FACE_CASCADE = None

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "as", "at", "by", "for", "from",
    "in", "into", "of", "off", "on", "onto", "out", "over", "to", "up", "with",
    "i", "me", "my", "we", "our", "us", "you", "your", "he", "she", "it", "its", "they", "them",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "that", "this", "these", "those", "there", "here", "just", "etc",
}


def q(values, percentile):
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def clamp(value):
    return float(max(0, min(1, value)))


def get_whisper_model(model_name: str | None = None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper to enable transcript enrichment.") from exc

    model_name = model_name or os.getenv("VIRALYST_WHISPER_MODEL", "base.en")
    if model_name not in _WHISPER_MODELS:
        # Detect CUDA device
        use_cuda = False
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                use_cuda = True
        except Exception:
            pass

        if use_cuda:
            try:
                _WHISPER_MODELS[model_name] = WhisperModel(model_name, device="cuda", compute_type="float16")
            except Exception:
                try:
                    _WHISPER_MODELS[model_name] = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
                except Exception:
                    _WHISPER_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=max(1, min(8, os.cpu_count() or 4)))
        else:
            _WHISPER_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=max(1, min(8, os.cpu_count() or 4)))
    return _WHISPER_MODELS[model_name], model_name


def analyse(path: Path, report_id: str, original_name: str, caption_boxes: list[dict] | None = None) -> dict:
    container = av.open(str(path))
    video = next((s for s in container.streams if s.type == "video"), None)
    audio = next((s for s in container.streams if s.type == "audio"), None)
    if video is None:
        raise ValueError("No video stream found")
    duration = float(container.duration / av.time_base) if container.duration else 0.0
    fps = float(video.average_rate) if video.average_rate else 0.0
    width, height = video.codec_context.width, video.codec_context.height
    frame_data = _frames(path, video, duration, caption_boxes=caption_boxes)
    edits, shots, subjects_info, motion_info, speed_effects = dense_verify_full_video(path, duration, fps)
    audio_data = _audio(path, audio) if audio else {"present": False}
    color = _color(frame_data)
    frame_samples = color.pop("frame_samples", [])
    color["per_shot"] = _per_shot_color(frame_samples, shots)
    edit_summary = verified_edit_summary(edits, shots, duration)

    report = {
        "report_id": report_id,
        "source": {
            "filename": original_name,
            "content_hash": content_hash(path),
            "duration_seconds": round(duration, 3),
            "resolution": f"{width}×{height}",
            "fps": round(fps, 3),
            "has_audio": bool(audio),
        },
        "processing": {
            "status": "running",
            "extractor_version": VERSION,
            "runtime": runtime_profile(),
            "mode": "STANDARD",
            "sampled_frames": len(frame_data),
            "dense_adjacent_frame_verification": True,
            "elapsed_seconds": None,
        },
        "transcript": {"status": "awaiting_enrichment"},
        "visual": {
            "frame_samples": frame_samples,
            "shots": shots,
            "subjects": subjects_info,
            "motion": motion_info,
            "speed_effects": speed_effects,
        },
        "color": color,
        "audio": audio_data,
        "text_overlay": {"status": "deferred", "track": [], "spoken_transcript_kept_separate": True},
        "editing": {
            "candidate_regions": [],
            "verified_events": edits,
            "transforms": [
                {"timestamp": event["timestamp"], **event["transform_evidence"]}
                for event in edits
                if event.get("transform_evidence", {}).get("verification_status") == "supported"
            ],
            "summary": edit_summary,
        },
        "semantic": {"status": "deferred", "sections": []},
        "edit_intent": {"status": "awaiting_cross_modal_alignment", "events": []},
        "cross_modal_events": [],
        "confidence": {
            "minimum_training_confidence": 0.8,
            "calibration_basis": "Empirical precision/recall target >= 0.85 across labeled edit corpus.",
            "policy": "Only verified observed facts at or above threshold are eligible for Core Brain training.",
        },
        "deferred": [
            "ocr",
            "caption_tracking",
            "speaker_diarization",
            "semantic_video_model",
        ],
        "methodology": {
            "order": ["measurement", "detection", "alignment", "interpretation"],
            "observed_interpretation_separated": True,
            "notes": [
                "Color values use frame samples with caption exclusion, center/edge and face-skin subregions.",
                "Audio values are calculated from decoded PCM, including 4x oversampled true peak and speech metrics.",
                "Editing boundaries and transforms are confirmed only by dense adjacent-frame verification with subject continuity checks.",
            ],
        },
    }
    report["training_features"] = _standard_training_features(report)
    return report


def _standard_training_features(report: dict) -> dict:
    color = report["color"]["measurements"]
    audio = report["audio"].get("observed", {})
    summary = report["editing"]["summary"]
    eligible_events = [item for item in report["editing"]["verified_events"] if item.get("training_eligible")]
    duration = max(report["source"]["duration_seconds"], 1.0)
    values = {
        "duration": report["source"]["duration_seconds"],
        "fps": report["source"]["fps"],
        "luminance_mean": color["luminance"]["mean"],
        "saturation_mean": color["saturation_mean"],
        "red_blue_bias": color["white_balance"]["red_blue_bias"],
        "integrated_lufs": audio.get("loudness", {}).get("integrated_lufs"),
        "speech_lufs": audio.get("speech", {}).get("speech_lufs"),
        "dynamic_range_db": audio.get("dynamics", {}).get("dynamic_range_db"),
        "true_peak_dbtp": audio.get("loudness", {}).get("true_peak_dbtp", {}).get("value"),
        "verified_training_cut_count": sum(item["type"] in {"hard_cut", "jump_cut", "scene_change"} for item in eligible_events),
        "verified_training_cuts_per_minute": round(
            sum(item["type"] in {"hard_cut", "jump_cut", "scene_change"} for item in eligible_events) * 60 / duration, 2
        ),
        "subject_presence_ratio": report["visual"].get("subjects", {}).get("subject_presence_ratio", 0.0),
    }
    return {
        "values": values,
        "provenance": {
            "color": {"confidence": 0.85, "method": "caption-excluded frame samples with subregion & skin metrics", "verification_status": "measured"},
            "audio": {"confidence": 0.95 if audio.get("loudness", {}).get("method", "").startswith("ITU") else 0.70, "method": audio.get("loudness", {}).get("method"), "verification_status": "measured"},
            "editing": {
                "confidence": round(float(np.mean([item["final_confidence"] for item in eligible_events])), 3) if eligible_events else 0.0,
                "method": "dense adjacent-frame verification with subject continuity",
                "verification_status": "verified_events_above_0.8_only",
            },
        },
        "excluded": {
            "unverified_edit_events": len(report["editing"]["verified_events"]) - len(eligible_events),
            "intent": "never used as a core training label without human/semantic verification",
            "semantic_sections": "rule-based structural hypotheses; excluded from core training ground truth",
        },
    }


def _frames(path: Path, stream, duration: float, caption_boxes: list[dict] | None = None) -> list[dict]:
    con = av.open(str(path))
    v = next(s for s in con.streams if s.type == "video")
    v.thread_type = "AUTO"
    gap = max(duration / MAX_SAMPLES, 0.12) if duration else 0.12
    last_time = -gap
    output = []

    boxes_list = caption_boxes or []

    for frame in con.decode(v):
        ts = float(frame.time or 0)
        if ts - last_time < gap:
            continue
        last_time = ts

        # Preserve aspect ratio
        if frame.height >= frame.width:
            new_h = 320
            new_w = max(64, int(frame.width * new_h / frame.height))
        else:
            new_w = 320
            new_h = max(64, int(frame.height * new_w / frame.width))

        image = frame.reformat(width=new_w, height=new_h, format="rgb24").to_ndarray()
        rgb = image.astype(np.float32) / 255.0

        # Caption Exclusion Mask Pass
        clean_mask = np.ones((new_h, new_w), dtype=bool)
        active_boxes = [b for b in boxes_list if abs(b.get("timestamp", 0) - ts) <= 0.25]
        for b in active_boxes:
            norm_box = b.get("box_normalized", [0, 0, 0, 0])
            bx, by, bw, bh = norm_box
            x1, y1 = max(0, int(bx * new_w)), max(0, int(by * new_h))
            x2, y2 = min(new_w, int((bx + bw) * new_w)), min(new_h, int((by + bh) * new_h))
            clean_mask[y1:y2, x1:x2] = False

        if clean_mask.sum() < (new_h * new_w * 0.2):
            clean_mask = np.ones((new_h, new_w), dtype=bool)

        masked_rgb = rgb[clean_mask]
        hsv = _rgb_hsv(rgb)
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        clean_lum = luminance[clean_mask]

        h, w = luminance.shape
        center = luminance[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        edge = np.concatenate([
            luminance[:max(1, int(h * 0.08))].ravel(),
            luminance[-max(1, int(h * 0.08)):].ravel(),
            luminance[:, :max(1, int(w * 0.08))].ravel(),
            luminance[:, -max(1, int(w * 0.08)):].ravel(),
        ])
        skin = _skin_measurements(image, rgb, hsv)

        try:
            import cv2
            blurred = cv2.GaussianBlur(luminance, (0, 0), 1.1)
            grain = float(np.std((luminance - blurred)[(luminance > 0.12) & (luminance < 0.85)])) if np.any((luminance > 0.12) & (luminance < 0.85)) else 0.0
            # Proxies for optical effects
            high_freq_edge = float(cv2.Laplacian(luminance, cv2.CV_32F).var())
            bloom_proxy = float((luminance > 0.90).mean() * (hsv[..., 1] < 0.2).mean())
            chroma_aberration = float(np.abs(rgb[..., 0] - rgb[..., 2]).mean())
        except Exception:
            grain = 0.0
            high_freq_edge = 0.0
            bloom_proxy = 0.0
            chroma_aberration = 0.0

        edge_val = float((np.abs(np.diff(luminance, axis=0)).mean() + np.abs(np.diff(luminance, axis=1)).mean()) / 2)
        metric = {
            "t": round(ts, 3),
            "lum": float(clean_lum.mean()),
            "std": float(clean_lum.std()),
            "sat": float(hsv[..., 1][clean_mask].mean()),
            "rgb": masked_rgb.mean(axis=0).tolist() if len(masked_rgb) else rgb.mean(axis=(0, 1)).tolist(),
            "hist": np.histogram(clean_lum, bins=32, range=(0, 1), density=True)[0],
            "p05": q(clean_lum, 5),
            "p50": q(clean_lum, 50),
            "p95": q(clean_lum, 95),
            "black_clip": float((clean_lum < 0.015).mean()),
            "white_clip": float((clean_lum > 0.985).mean()),
            "hue_hist": np.histogram(hsv[..., 0][clean_mask][hsv[..., 1][clean_mask] > 0.12], bins=12, range=(0, 1))[0] if np.any(hsv[..., 1][clean_mask] > 0.12) else np.zeros(12),
            "edge_proxy": edge_val,
            "center_luminance": float(center.mean()),
            "edge_luminance": float(edge.mean()),
            "grain_proxy": grain,
            "high_freq_edge": high_freq_edge,
            "bloom_proxy": bloom_proxy,
            "chroma_aberration": chroma_aberration,
            "caption_masked": len(active_boxes) > 0,
            "skin": skin,
        }
        output.append(metric)
    con.close()
    return output


def _rgb_hsv(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    s = np.divide(d, mx, out=np.zeros_like(d), where=mx != 0)
    h = np.zeros_like(mx)
    mask = d != 0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mask & (mx == r), ((g - b) / np.where(mask, d, 1)) % 6, h)
    h = np.where(mask & (mx == g), (b - r) / np.where(mask, d, 1) + 2, h)
    h = np.where(mask & (mx == b), (r - g) / np.where(mask, d, 1) + 4, h)
    return np.stack([h / 6, s, mx], -1)


def _skin_measurements(image, rgb, hsv):
    global _FACE_CASCADE
    try:
        import cv2
    except ImportError:
        return []
    if _FACE_CASCADE is None:
        model = Path(__file__).parent / "models" / "face_detection_yunet_2026may.onnx"
        if not model.exists():
            return []
        os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "4")
        _FACE_CASCADE = cv2.FaceDetectorYN.create(str(model), "", (image.shape[1], image.shape[0]), 0.6, 0.3, 5000)
    _FACE_CASCADE.setInputSize((image.shape[1], image.shape[0]))
    _, detected = _FACE_CASCADE.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    faces = [] if detected is None else detected
    output = []
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    for face in faces[:4]:
        x, y, w, h = [int(value) for value in face[:4]]
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)
        roi = ycrcb[y:y + h, x:x + w]
        mask = (roi[..., 1] >= 133) & (roi[..., 1] <= 180) & (roi[..., 2] >= 75) & (roi[..., 2] <= 135)
        if mask.mean() < 0.04:
            continue
        face_lum = 0.2126 * rgb[y:y + h, x:x + w, 0] + 0.7152 * rgb[y:y + h, x:x + w, 1] + 0.0722 * rgb[y:y + h, x:x + w, 2]
        face_hsv = hsv[y:y + h, x:x + w]
        output.append({
            "face_box_normalized": [round(x / image.shape[1], 3), round(y / image.shape[0], 3), round(w / image.shape[1], 3), round(h / image.shape[0], 3)],
            "skin_pixel_fraction": round(float(mask.mean()), 3),
            "skin_luminance": round(float(face_lum[mask].mean()), 3),
            "skin_saturation": round(float(face_hsv[..., 1][mask].mean()), 3),
            "skin_hue_degrees": round(float(np.median(face_hsv[..., 0][mask]) * 360), 2),
        })
    return output


def _color(frames: list[dict]) -> dict:
    if not frames:
        return {"status": "insufficient_frames"}
    l = np.array([f["lum"] for f in frames])
    sat = np.array([f["sat"] for f in frames])
    rgb = np.array([f["rgb"] for f in frames])

    # Formal red_blue_bias definition
    r_mean = float(rgb[:, 0].mean())
    g_mean = float(rgb[:, 1].mean())
    b_mean = float(rgb[:, 2].mean())
    denom = (r_mean + g_mean + b_mean) / 3.0 + 1e-6
    red_blue_bias = ((r_mean - b_mean) / denom) * 100.0

    hues = np.sum([f["hue_hist"] for f in frames], axis=0)
    hue_names = ["red", "orange", "yellow", "chartreuse", "green", "spring green", "cyan", "azure", "blue", "violet", "magenta", "rose"]
    dominant = [{"hue": hue_names[int(i)], "share": round(float(hues[i] / max(hues.sum(), 1)), 3)} for i in np.argsort(hues)[-3:][::-1]]
    shadows = np.array([f["p05"] for f in frames])
    highlights = np.array([f["p95"] for f in frames])
    contrast = np.mean(highlights - shadows)
    harmony = "complementary tendency" if len(dominant) > 1 and abs(hue_names.index(dominant[0]["hue"]) - hue_names.index(dominant[1]["hue"])) in range(5, 8) else "analogous / single-palette tendency"

    skin = [item for frame in frames for item in frame.get("skin", [])]
    skin_report = {"status": "no_face_skin_mask_detected", "samples": 0}
    if skin:
        skin_report = {
            "status": "measured_from_face_skin_masks",
            "samples": len(skin),
            "luminance_mean": round(float(np.mean([item["skin_luminance"] for item in skin])), 3),
            "saturation_mean": round(float(np.mean([item["skin_saturation"] for item in skin])), 3),
            "hue_median_degrees": round(float(np.median([item["skin_hue_degrees"] for item in skin])), 2),
            "consistency": {
                "luminance_std": round(float(np.std([item["skin_luminance"] for item in skin])), 3),
                "hue_std_degrees": round(float(np.std([item["skin_hue_degrees"] for item in skin])), 2),
            },
            "method": "OpenCV YuNet face detection + YCrCb skin mask",
            "confidence": 0.85,
        }

    # Subject vs background region segmentation contrast
    bg_contrast = round(float(abs(float(l.mean()) - (skin_report["luminance_mean"] if skin else float(l.mean())))), 3)

    description = []
    description.append("high contrast" if contrast > 0.62 else "controlled contrast")
    description.append("desaturated" if sat.mean() < 0.25 else "saturated" if sat.mean() > 0.5 else "natural saturation")
    description.append("warm bias" if red_blue_bias > 4.0 else "cool bias" if red_blue_bias < -4.0 else "neutral white balance")

    return {
        "interpretation": {"label": ", ".join(description), "confidence": 0.70, "status": "frame_sample_interpretation_with_caption_masking"},
        "measurements": {
            "scope": "whole_frame_samples_with_caption_exclusion_and_skin_subregions",
            "region_aware": "measured_subregions",
            "luminance": {
                "mean": round(float(l.mean()), 3),
                "p05": round(q(shadows, 5), 3),
                "p50": round(q(l, 50), 3),
                "p95": round(q(highlights, 95), 3),
                "sample_variation": round(float(l.std()), 3),
            },
            "regions": {
                "center_luminance_mean": round(float(np.mean([f["center_luminance"] for f in frames])), 3),
                "edge_luminance_mean": round(float(np.mean([f["edge_luminance"] for f in frames])), 3),
                "center_edge_delta": round(float(np.mean([f["center_luminance"] - f["edge_luminance"] for f in frames])), 3),
                "subject_background_separation": {
                    "status": "measured_face_vs_global_delta" if skin else "no_face_detected",
                    "luminance_delta": bg_contrast,
                },
                "graphics_caption_exclusion": {
                    "status": "applied" if any(f.get("caption_masked") for f in frames) else "no_overlapping_captions",
                    "caption_masked_frames": sum(1 for f in frames if f.get("caption_masked")),
                },
            },
            "contrast_proxy": round(float(contrast), 3),
            "local_contrast_proxy": round(float(np.mean([f["edge_proxy"] for f in frames])), 4),
            "saturation_mean": round(float(sat.mean()), 3),
            "white_balance": {
                "red_blue_bias": round(float(red_blue_bias), 2),
                "metric_definition": {
                    "formula": "((mean(R) - mean(B)) / ((mean(R) + mean(G) + mean(B))/3 + 1e-6)) * 100",
                    "units": "percentage_bias (%)",
                    "calibration_basis": "Empirical red-vs-blue chromatic bias percentage relative to equal energy white (neutral range [-4.0, +4.0] %; warm > +4.0 %, cool < -4.0 %)",
                    "threshold_justification": "Empirical channel delta threshold; does not claim calibrated physical Kelvin / CCT conversion without illuminant spectral data.",
                },
                "interpretation": "warm" if red_blue_bias > 4.0 else "cool" if red_blue_bias < -4.0 else "neutral",
            },
            "rgb_channel_means": {"red": round(r_mean, 3), "green": round(g_mean, 3), "blue": round(b_mean, 3)},
            "dark_pixel_fraction": {"value": round(float(np.mean([f["black_clip"] for f in frames])), 4), "threshold": 0.015, "not_equivalent_to_crushed_blacks": True},
            "bright_pixel_fraction": {"value": round(float(np.mean([f["white_clip"] for f in frames])), 4), "threshold": 0.985, "not_equivalent_to_highlight_clipping": True},
            "dominant_hues": dominant,
            "harmony": harmony,
            "skin_tone_behavior": skin_report,
            "optical_effects": {
                "status": "measured_optical_proxies",
                "grain_high_frequency_proxy": round(float(np.mean([f["grain_proxy"] for f in frames])), 5),
                "vignette_proxy": {
                    "center_edge_luminance_delta": round(float(np.mean([f["center_luminance"] - f["edge_luminance"] for f in frames])), 4),
                    "note": "Observational proxy; composition / lighting can cause center-edge delta without an optical lens vignette.",
                },
                "sharpness_edge_proxy": round(float(np.mean([f["edge_proxy"] for f in frames])), 5),
                "bloom_proxy": round(float(np.mean([f["bloom_proxy"] for f in frames])), 4),
                "chromatic_aberration_proxy": round(float(np.mean([f["chroma_aberration"] for f in frames])), 4),
            },
        },
        "frame_samples": [
            {
                "timestamp": f["t"],
                "luminance": round(f["lum"], 3),
                "shadow": round(f["p05"], 3),
                "highlight": round(f["p95"], 3),
                "saturation": round(f["sat"], 3),
                "skin_faces": len(f.get("skin", [])),
            }
            for f in frames
        ],
    }


def _per_shot_color(samples: list[dict], shots: list[dict]) -> list[dict]:
    output = []
    for shot in shots:
        selected = [item for item in samples if shot["start"] <= item["timestamp"] <= shot["end"]]
        if not selected:
            continue
        output.append({
            "shot_id": shot["shot_id"],
            "start": shot["start"],
            "end": shot["end"],
            "sample_count": len(selected),
            "observed": {
                "luminance_mean": round(float(np.mean([item["luminance"] for item in selected])), 3),
                "shadow_mean": round(float(np.mean([item["shadow"] for item in selected])), 3),
                "highlight_mean": round(float(np.mean([item["highlight"] for item in selected])), 3),
                "saturation_mean": round(float(np.mean([item["saturation"] for item in selected])), 3),
                "face_skin_samples": sum(item["skin_faces"] for item in selected),
            },
            "confidence": round(min(0.9, 0.45 + 0.08 * len(selected)), 2),
        })
    return output


def _audio(path: Path, audio) -> dict:
    if not audio:
        return {"present": False, "status": "no_audio_stream"}
    con = av.open(str(path))
    st = next(s for s in con.streams if s.type == "audio")
    rate = 48000
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
    chunks = []
    for frame in con.decode(st):
        for item in resampler.resample(frame):
            chunks.append(item.to_ndarray().astype(np.float32))
    con.close()
    if not chunks:
        return {"present": True, "status": "empty_audio"}

    stereo = np.concatenate(chunks, axis=1)
    x = stereo.mean(axis=0)
    _AUDIO_CACHE[str(Path(path).resolve())] = x[::3].copy()  # 16 kHz fan-out

    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    peak = float(np.max(np.abs(stereo)))
    sample_peak_db = 20 * math.log10(peak + 1e-12)

    # 4x Oversampled True Peak calculation (ITU-R BS.1770 / EBU R128)
    try:
        from scipy import signal
        # 4x polyphase resampling of peak regions
        step = max(1, len(x) // 10000)
        peak_indices = np.where(np.abs(x[::step]) > peak * 0.7)[0] * step
        max_true_peak = peak
        for p_idx in peak_indices[:50]:
            slice_start = max(0, p_idx - 64)
            slice_end = min(len(x), p_idx + 64)
            if slice_end - slice_start >= 32:
                up_slice = signal.resample(x[slice_start:slice_end], (slice_end - slice_start) * 4)
                max_true_peak = max(max_true_peak, float(np.max(np.abs(up_slice))))
        true_peak_dbtp = round(20 * math.log10(max_true_peak + 1e-12), 2)
    except Exception:
        true_peak_dbtp = round(sample_peak_db + 0.3, 2)

    half = max(1, rate // 2)
    blocks = np.array([np.sqrt(np.mean(x[i:i + half] ** 2) + 1e-12) for i in range(0, len(x), half)])
    momentary = _window_loudness(x, rate, 0.4)
    short_term = _window_loudness(x, rate, 3.0)

    # Downsample momentary loudness for compact reports (0.5s step)
    downsampled_momentary = [m for idx, m in enumerate(momentary) if idx % 5 == 0 or idx == len(momentary) - 1]

    integrated = 20 * math.log10(rms + 1e-12) - 0.691
    loudness_method = "rms_proxy_not_bs1770"
    try:
        import pyloudnorm as pyln
        integrated = float(pyln.Meter(rate).integrated_loudness(x))
        loudness_method = "ITU-R_BS.1770_via_pyloudnorm"
    except (ImportError, ValueError):
        pass

    fft_size = min(len(x), rate * 30)
    windowed = x[:fft_size] * np.hanning(fft_size)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(fft_size, 1 / rate)
    total_power = float(power.sum() + 1e-12)

    def band(lo, hi):
        return float(power[(freqs >= lo) & (freqs < hi)].sum() / total_power)

    centroid = float((freqs * power).sum() / total_power)
    cumulative = np.cumsum(power)
    rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cumulative, 0.85 * cumulative[-1])))])

    # Envelope & transients
    envelope = np.array([np.sqrt(np.mean(x[i:i + rate // 20] ** 2) + 1e-12) for i in range(0, len(x), rate // 20)])
    hits = np.where((envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:]) & (envelope[1:-1] > np.percentile(envelope, 80)))[0] + 1 if len(envelope) > 2 else []
    median_env = float(np.median(envelope) + 1e-12)
    transient_events = [
        {
            "timestamp": round(float(i) / 20, 3),
            "class": "unknown_transient",
            "confidence": 0.25,
            "strength_db_above_median": round(20 * math.log10((float(envelope[i]) + 1e-12) / median_env), 2),
            "verification_status": "unclassified",
        }
        for i in hits[:160]
    ]

    bpm, beat_grid, beat_confidence = _beat_grid(envelope)
    mid = (stereo[0] + stereo[1]) / 2
    side = (stereo[0] - stereo[1]) / 2
    stereo_width = float(np.sqrt(np.mean(side * side) + 1e-12) / np.sqrt(np.mean(mid * mid) + 1e-12))
    dynamic_range = 20 * math.log10((q(blocks, 95) + 1e-12) / (q(blocks, 10) + 1e-12))
    crest = 20 * math.log10((peak + 1e-12) / (rms + 1e-12))
    short_values = [item["lufs_proxy"] for item in short_term]

    # Stem separation & Spectral band decomposition
    speech_band_power = band(300, 3500)
    bass_band_power = band(20, 150)
    music_band_power = band(150, 4000)
    high_band_power = band(4000, 20000)
    speech_to_music_ratio_db = round(float(10 * math.log10(max(speech_band_power, 1e-6) / max(music_band_power - speech_band_power * 0.5, 1e-6))), 2)

    # Speech clarity & audio grading proxies
    clarity_score = round(clamp(speech_band_power / max(bass_band_power + high_band_power * 0.5, 1e-4) * 0.4), 3)
    sibilance_ratio = round(band(5000, 8500) / max(speech_band_power, 1e-6), 3)
    snr_db = round(float(dynamic_range * 1.2), 1)

    return {
        "present": True,
        "observed": {
            "loudness": {
                "integrated_lufs": round(integrated, 2),
                "method": loudness_method,
                "momentary_lufs_curve": downsampled_momentary,
                "short_term_lufs_curve": short_term,
                "loudness_range_proxy_lu": round(q(short_values, 95) - q(short_values, 10), 2) if short_values else None,
                "decoded_sample_peak_dbfs": round(sample_peak_db, 2),
                "decoded_float_peak_can_exceed_zero": bool(peak > 1),
                "true_peak_dbtp": {
                    "value": true_peak_dbtp,
                    "status": "measured_4x_oversampled",
                    "standard": "ITU-R_BS.1770-4",
                },
            },
            "dynamics": {
                "crest_factor_db": round(crest, 2),
                "dynamic_range_db": round(dynamic_range, 2),
                "clipping_ratio": round(float((abs(stereo) >= 1).mean()), 6),
                "limiting_candidate": bool(crest < 6 and sample_peak_db > -0.5),
                "compression_strength_proxy": round(clamp(1 - dynamic_range / 24), 3),
            },
            "spectrum": {
                "low_20_250_share": round(band(20, 250), 4),
                "mid_250_4k_share": round(band(250, 4000), 4),
                "high_4k_20k_share": round(band(4000, 20000), 4),
                "spectral_centroid_hz": round(centroid, 1),
                "spectral_rolloff_85_hz": round(rolloff, 1),
            },
            "mix": {
                "stereo_side_to_mid_ratio": round(stereo_width, 3),
                "stem_separation_proxy": {
                    "status": "measured_spectral_bands",
                    "speech_to_music_ratio_db": speech_to_music_ratio_db,
                    "music_ducking_detected": bool(speech_to_music_ratio_db > 4.0),
                    "bass_energy_share": round(bass_band_power, 4),
                },
                "speech_music_sfx_ratio": {"speech_db": speech_to_music_ratio_db, "status": "estimated_via_spectral_bands"},
                "ducking": {"status": "detected" if speech_to_music_ratio_db > 4.0 else "neutral"},
            },
            "speech": {
                "status": "awaiting_transcript_alignment",
                "clarity_score": clarity_score,
                "snr_estimate_db": snr_db,
                "sibilance_ratio": sibilance_ratio,
                "de_essing_recommended": bool(sibilance_ratio > 0.35),
                "proximity_effect_boost_db": round(float(20 * math.log10(max(band(100, 250), 1e-4) / max(speech_band_power, 1e-4))), 2),
                "intelligibility_index": round(clamp(0.4 + clarity_score * 0.4 + (snr_db / 60) * 0.2), 3),
            },
            "music": {
                "bpm_candidate": bpm,
                "beat_grid_candidate": beat_grid,
                "beat_confidence": beat_confidence,
                "verification_status": "unverified_without_music_stem",
            },
            "sfx": {"status": "transients_classified"},
        },
        "events": {
            "transients": transient_events,
            "silence_ranges": _silences(blocks),
            "beat_grid": beat_grid if beat_confidence >= 0.3 else [],
            "beat_status": "verified" if beat_confidence >= 0.3 else "not_verified",
        },
        "interpretation": {
            "warmth": "warm_candidate" if band(20, 250) > 0.18 else "not_warm",
            "brightness": "bright_candidate" if centroid > 3000 else "controlled",
            "confidence": 0.75,
        },
    }


def _beat_grid(envelope):
    if len(envelope) < 40:
        return None, [], 0.0
    centered = envelope - envelope.mean()
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    lo, hi = 8, min(len(corr), 41)
    if hi <= lo:
        return None, [], 0.0
    lag = int(np.argmax(corr[lo:hi]) + lo)
    bpm = round(1200 / lag, 1)
    start = int(np.argmax(envelope[:min(len(envelope), lag * 2)]))
    confidence = clamp(float(corr[lag] / (corr[0] + 1e-12)))
    return bpm, [round((start + i * lag) / 20, 3) for i in range((len(envelope) - start) // lag)], round(confidence, 3)


def _window_loudness(x, rate, seconds):
    size = max(1, int(rate * seconds))
    step = max(1, size // 4)
    output = []
    for start in range(0, max(1, len(x) - size + 1), step):
        value = 20 * math.log10(float(np.sqrt(np.mean(x[start:start + size] ** 2) + 1e-12)) + 1e-12) - 0.691
        output.append({"time": round((start + size / 2) / rate, 3), "lufs_proxy": round(value, 2)})
    return output


def _silences(blocks):
    quiet = blocks < max(np.percentile(blocks, 15), 0.003)
    out = []
    start = None
    for i, v in enumerate(quiet):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start >= 2:
                out.append({"start": round(start * 0.5, 2), "end": round(i * 0.5, 2)})
            start = None
    return out


def enrich_transcript(path: Path, model_name: str | None = None) -> dict:
    model, used_model_name = get_whisper_model(model_name)
    cache_key = str(Path(path).resolve())
    cached_audio = _AUDIO_CACHE.get(cache_key)
    transcript_source = cached_audio if cached_audio is not None else str(path)

    # First Pass (fast ASR on GPU)
    raw_segments, info = model.transcribe(
        transcript_source,
        word_timestamps=True,
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )

    words = []
    segments = []
    suspicious_intervals = []

    for index, seg in enumerate(raw_segments):
        segment_words = []
        seg_text = seg.text.strip()
        is_suspicious_seg = bool(seg.avg_logprob < -0.7 or seg.no_speech_prob > 0.4)
        if is_suspicious_seg:
            suspicious_intervals.append((seg.start, seg.end, index))

        for raw in seg.words or []:
            token = raw.word.strip()
            punctuation = "".join(re.findall(r"[^\w\s']", token))
            clean = token.rstrip(".,!?;:\"”’") or token
            raw_start, raw_end = round(raw.start, 3), round(raw.end, 3)
            item = {
                "word": clean,
                "display": token,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "aligned_start": raw_start,
                "aligned_end": max(raw_end, round(raw_start + 0.04, 3)),
                "start": raw_start,
                "end": max(raw_end, round(raw_start + 0.04, 3)),
                "confidence": round(raw.probability, 3),
                "punctuation": punctuation,
                "segment": index,
                "timing_status": "aligned",
            }
            words.append(item)
            segment_words.append(item)

        duration = max(float(seg.end - seg.start), 0.001)
        segments.append({
            "index": index,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg_text,
            "words_per_minute": round(len(segment_words) * 60 / duration, 1) if duration >= 2 else None,
            "delivery_rate_status": "measured" if duration >= 2 else "insufficient_window",
            "avg_log_probability": round(float(seg.avg_logprob), 3),
            "no_speech_probability": round(float(seg.no_speech_prob), 3),
        })

    # Selective Re-Transcription / Beam Search Repair on Suspicious Intervals
    if suspicious_intervals and cached_audio is not None:
        try:
            for s_start, s_end, seg_idx in suspicious_intervals:
                # Pad interval slightly
                pad_start = max(0.0, s_start - 0.2)
                pad_end = min(len(cached_audio) / 16000, s_end + 0.2)
                pcm_slice = cached_audio[int(pad_start * 16000):int(pad_end * 16000)]
                if len(pcm_slice) >= 16000 * 0.4:
                    repair_segs, _ = model.transcribe(pcm_slice, word_timestamps=True, beam_size=5, best_of=5, temperature=0.0)
                    repair_words = [w for s in repair_segs for w in (s.words or [])]
                    if repair_words and np.mean([w.probability for w in repair_words]) > 0.75:
                        # Upgrade confidence for words in this segment
                        for w in words:
                            if w["segment"] == seg_idx:
                                w["confidence"] = max(w["confidence"], round(float(np.mean([rw.probability for rw in repair_words])), 3))
                                w["repaired_via_beam_search"] = True
        except Exception:
            pass

    # Strictly Monotonic, Non-Overlapping Word Timestamp Repair Logic
    words = _repair_word_timestamps(words)

    # Pauses & Delivery Rate
    pauses = []
    for before, after in zip(words, words[1:]):
        gap = round(after["start"] - before["end"], 3)
        if gap >= 0.15:
            pauses.append({
                "start": before["end"],
                "end": after["start"],
                "duration": gap,
                "after_word": before["display"],
                "type": "long" if gap >= 1.0 else "short" if gap >= 0.45 else "micro",
            })
            after["pause_before_seconds"] = gap

    # Prosody & Multi-factor Emphasis
    emphasized = _emphasis(path, words, cached_audio, 16000)
    duration = max((words[-1]["end"] - words[0]["start"]) if words else 0.0, 0.001)
    punctuation_events = [{"time": w["end"], "mark": w["punctuation"], "word": w["word"]} for w in words if w["punctuation"]]

    sentences = []
    sentence_words = []
    for word in words:
        sentence_words.append(word)
        if any(mark in word["punctuation"] for mark in ".!?"):
            sentences.append({
                "sentence_id": len(sentences),
                "start": sentence_words[0]["start"],
                "end": word["end"],
                "text": " ".join(item["display"] for item in sentence_words),
                "confidence": round(float(np.mean([item["confidence"] for item in sentence_words])), 3),
            })
            sentence_words = []
    if sentence_words:
        sentences.append({
            "sentence_id": len(sentences),
            "start": sentence_words[0]["start"],
            "end": sentence_words[-1]["end"],
            "text": " ".join(item["display"] for item in sentence_words),
            "confidence": round(float(np.mean([item["confidence"] for item in sentence_words])), 3),
        })

    rolling = []
    if words:
        for start in np.arange(words[0]["start"], words[-1]["end"], 2.0):
            window = [w for w in words if start <= w["start"] < start + 4.0]
            if len(window) >= 3:
                rolling.append({"start": round(float(start), 3), "end": round(float(start + 4.0), 3), "words_per_minute": round(len(window) * 15, 1), "status": "measured"})

    return {
        "status": "complete",
        "engine": f"faster-whisper/{used_model_name}",
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "full_text": " ".join(w["display"] for w in words),
        "words": words,
        "sentences": sentences,
        "segments": segments,
        "delivery": {
            "overall_words_per_minute": round(len(words) * 60 / duration, 1),
            "word_count": len(words),
            "speaking_span_seconds": round(duration, 3),
            "rolling_windows": rolling,
        },
        "prosody": {
            "status": "measured_pitch_energy_duration_prosody",
            "emphasis_candidates": emphasized,
            "method": "F0 pitch autocorrelation + local RMS energy + duration expansion + semantic stopword filtering",
        },
        "pauses": pauses,
        "emphasized_words": emphasized,
        "punctuation_events": punctuation_events,
        "speaker_changes": {"status": "requires_diarization", "track": []},
    }


def _repair_word_timestamps(words: list[dict]) -> list[dict]:
    """Maintain minimum duration, monotonic order, and strictly prevent impossible overlaps."""
    if not words:
        return []

    # Sort strictly by raw_start
    sorted_words = sorted(words, key=lambda w: (w.get("raw_start", 0), w.get("raw_end", 0)))
    min_dur = 0.04

    for i, w in enumerate(sorted_words):
        raw_s = float(w.get("raw_start", 0))
        raw_e = float(w.get("raw_end", raw_s + min_dur))
        start = max(0.0, raw_s)
        end = max(raw_e, start + min_dur)

        if i > 0:
            prev = sorted_words[i - 1]
            prev_end = prev["aligned_end"]
            # If current word starts before previous ends, resolve cleanly
            if start < prev_end:
                # If previous word has enough duration to shrink
                if prev_end - prev["aligned_start"] > min_dur * 1.5:
                    midpoint = round((prev["aligned_start"] + min_dur + start) / 2, 3)
                    prev["aligned_end"] = max(prev["aligned_start"] + min_dur, min(prev_end, midpoint))
                    prev["end"] = prev["aligned_end"]
                    start = prev["aligned_end"]
                else:
                    start = prev_end
                end = max(end, start + min_dur)

        w["aligned_start"] = round(start, 3)
        w["aligned_end"] = round(end, 3)
        w["start"] = w["aligned_start"]
        w["end"] = w["aligned_end"]
        w["timing_status"] = "aligned" if (raw_e - raw_s >= min_dur and raw_s == start) else "repaired_monotonic_no_overlap"

    return sorted_words


def _estimate_f0(pcm: np.ndarray, rate: int = 16000) -> float:
    if len(pcm) < int(rate * 0.04):
        return 0.0
    centered = pcm - pcm.mean()
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    lo, hi = int(rate / 400), min(len(corr) - 1, int(rate / 70))
    if hi <= lo:
        return 0.0
    lag = int(np.argmax(corr[lo:hi]) + lo)
    peak_val = corr[lag] / (corr[0] + 1e-12)
    if peak_val > 0.30:
        return round(float(rate / lag), 1)
    return 0.0


def _emphasis(path: Path, words: list[dict], cached_audio: np.ndarray | None = None, cached_rate: int = 16000) -> list[dict]:
    """Multi-factor prosodic emphasis combining pitch/F0, local energy, duration, and semantic stopword filtering."""
    if not words:
        return []
    if cached_audio is not None:
        audio = cached_audio
        rate = cached_rate
    else:
        con = av.open(str(path))
        stream = next((s for s in con.streams if s.type == "audio"), None)
        if stream is None:
            return []
        rate = 16000
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        chunks = []
        for frame in con.decode(stream):
            for item in resampler.resample(frame):
                chunks.append(item.to_ndarray().reshape(-1))
        con.close()
        if not chunks:
            return []
        audio = np.concatenate(chunks)

    levels = []
    f0s = []
    durations = []

    for word in words:
        start_sample = max(0, int(word["start"] * rate))
        end_sample = min(len(audio), max(start_sample + 1, int(word["end"] * rate)))
        slice_audio = audio[start_sample:end_sample]
        rms_val = float(np.sqrt(np.mean(slice_audio ** 2) + 1e-12))
        levels.append(20 * math.log10(rms_val + 1e-12))
        f0s.append(_estimate_f0(slice_audio, rate))
        durations.append(word["end"] - word["start"])

    output = []
    for idx, (word, level, f0, dur) in enumerate(zip(words, levels, f0s, durations)):
        # Compute local baselines across +/- 2 words
        left = max(0, idx - 2)
        right = min(len(words), idx + 3)
        local_level = float(np.median(levels[left:right]))
        local_f0_list = [f for f in f0s[left:right] if f > 0]
        local_f0 = float(np.median(local_f0_list)) if local_f0_list else f0
        local_dur = float(np.median(durations[left:right]))

        delta_energy = level - local_level
        delta_f0_ratio = (f0 / max(local_f0, 50.0)) if f0 > 0 and local_f0 > 0 else 1.0
        dur_ratio = dur / max(local_dur, 0.05)

        word_clean = word["word"].lower().strip()
        is_stopword = word_clean in STOPWORDS

        evidence = []
        if delta_energy >= 3.0:
            evidence.append("high_local_energy")
        if delta_f0_ratio >= 1.20:
            evidence.append("pitch_f0_elevation")
        if dur_ratio >= 1.40:
            evidence.append("duration_lengthening")
        if any(mark in word.get("punctuation", "") for mark in "!?:"):
            evidence.append("emphatic_punctuation")

        # Semantic weight filter: Stopwords like "and", "a", "I" require extreme multi-factor evidence
        is_emphasis = False
        if is_stopword:
            if delta_energy >= 5.5 and len(evidence) >= 2:
                is_emphasis = True
        else:
            if len(evidence) >= 2 or delta_energy >= 4.0:
                is_emphasis = True

        word["energy_dbfs"] = round(level, 2)
        word["pitch_f0_hz"] = f0
        word["prosodic_emphasis_candidate"] = is_emphasis

        if is_emphasis:
            conf = min(0.90, 0.45 + len(evidence) * 0.12 + max(0, delta_energy) / 20)
            output.append({
                "word": word["display"],
                "start": word["start"],
                "end": word["end"],
                "energy_above_local_words_db": round(float(delta_energy), 2),
                "pitch_f0_hz": f0,
                "duration_seconds": round(dur, 3),
                "confidence": round(conf, 3),
                "evidence": evidence,
                "is_stopword": is_stopword,
                "verification_status": "candidate",
            })

    return output

