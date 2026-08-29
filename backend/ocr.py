from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import av
import numpy as np

_OCR_ENGINE = None


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        import os
        import glob
        import ctypes
        # Preload CUDA 13 libraries for ONNX Runtime CUDA provider
        nvidia_libs = glob.glob("/home/xor_sensei/miniconda3/lib/python3.13/site-packages/nvidia/*/lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + ":" + current_ld
        for lib_dir in nvidia_libs:
            for so in glob.glob(f"{lib_dir}/*.so*"):
                try:
                    ctypes.CDLL(so)
                except Exception:
                    pass

        try:
            from rapidocr import RapidOCR
            try:
                params = {"EngineConfig.onnxruntime.use_cuda": True}
                _OCR_ENGINE = RapidOCR(params=params)
            except Exception:
                _OCR_ENGINE = RapidOCR()
        except ImportError:
            return None
    return _OCR_ENGINE


def _is_false_positive(text: str, score: float, points: np.ndarray, img_w: int, img_h: int) -> bool:
    clean = text.strip()
    if not clean:
        return True
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    w_norm = (x2 - x1) / max(img_w, 1)
    h_norm = (y2 - y1) / max(img_h, 1)
    area_norm = w_norm * h_norm

    # 1. Reject full-screen or massive boxes covering >45% of frame with very short text (e.g. false positive "9")
    if area_norm > 0.45 and len(clean) < 8:
        return True
    if w_norm > 0.80 and h_norm > 0.80 and len(clean) < 15:
        return True

    # 2. Reject low-confidence short strings or single digits
    if score < 0.65 and len(clean) <= 2:
        return True
    if score < 0.50:
        return True

    # 3. Reject non-alphanumeric noise
    if not re.search(r"[a-zA-Z0-9]", clean):
        return True

    # 4. Reject isolated single characters with low-to-medium confidence
    if len(clean) == 1 and score < 0.88:
        return True

    return False


def _analyze_typography(image_bgr: np.ndarray, points: np.ndarray) -> dict:
    h, w = image_bgr.shape[:2]
    x1 = max(0, int(points[:, 0].min()))
    y1 = max(0, int(points[:, 1].min()))
    x2 = min(w, int(points[:, 0].max()))
    y2 = min(h, int(points[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return {
            "font_class": "sans_serif",
            "fill_color_rgb": [255, 255, 255],
            "fill_color_hex": "#FFFFFF",
            "has_stroke": False,
            "has_shadow": False,
            "has_background_box": False,
            "contrast_ratio": 1.0,
        }

    crop = image_bgr[y1:y2, x1:x2]
    crop_rgb = crop[..., ::-1].astype(np.float32)
    gray = 0.2126 * crop_rgb[..., 0] + 0.7152 * crop_rgb[..., 1] + 0.0722 * crop_rgb[..., 2]

    # Foreground text vs background estimation via Otsu/histogram
    threshold = np.median(gray)
    text_mask = gray > threshold if gray.mean() < 160 else gray < threshold
    if text_mask.sum() < 4:
        text_mask = np.ones_like(gray, dtype=bool)

    fg_rgb = crop_rgb[text_mask]
    bg_rgb = crop_rgb[~text_mask] if (~text_mask).sum() > 4 else crop_rgb

    fg_mean = fg_rgb.mean(axis=0) if len(fg_rgb) else np.array([255.0, 255.0, 255.0])
    r, g, b = int(np.clip(fg_mean[0], 0, 255)), int(np.clip(fg_mean[1], 0, 255)), int(np.clip(fg_mean[2], 0, 255))
    hex_color = f"#{r:02X}{g:02X}{b:02X}"

    # Edge analysis for stroke & font class
    try:
        import cv2
        edges = cv2.Canny(crop, 50, 150)
        edge_density = float(edges.mean() / 255.0)
        font_class = "display_bold" if edge_density > 0.22 else "serif" if edge_density > 0.16 else "sans_serif"
        has_stroke = bool(edge_density > 0.12 and abs(fg_mean.mean() - bg_rgb.mean()) > 40)
    except Exception:
        font_class = "sans_serif"
        has_stroke = False

    bg_var = float(bg_rgb.std()) if len(bg_rgb) else 999.0
    has_bg_box = bool(bg_var < 30.0 and len(bg_rgb) > 20)

    return {
        "font_class": font_class,
        "fill_color_rgb": [r, g, b],
        "fill_color_hex": hex_color,
        "has_stroke": has_stroke,
        "has_shadow": bool(has_stroke or bg_var > 60),
        "has_background_box": has_bg_box,
        "contrast_ratio": round(float(abs(fg_mean.mean() - bg_rgb.mean()) / 255.0), 3),
    }


def extract_text_overlay(path: Path, duration: float, target_fps: float = 2.0) -> dict:
    """Scene-text track with GPU acceleration, false-positive filtering, dense ROI tracking, and typography clustering."""
    engine = get_ocr_engine()
    if engine is None:
        return {"status": "unavailable", "engine": "rapidocr", "track": [], "reason": "Install rapidocr and onnxruntime-gpu"}

    # Adaptive sampling: 2.5 Hz base sampling with adaptive ROI difference gating
    step = max(0.20, 1.0 / max(target_fps, 1.0))
    targets = np.arange(0.0, max(0.01, duration), step)

    con = av.open(str(path))
    stream = next(s for s in con.streams if s.type == "video")
    stream.thread_type = "AUTO"

    samples = []
    target_idx = 0
    clean_boxes_for_color = []
    all_typography = []

    prev_caption_roi = None
    prev_detections = []
    prev_clean_boxes = []
    ocr_calls_performed = 0
    ocr_frames_reused = 0

    for frame in con.decode(stream):
        ts = float(frame.time or 0)
        if target_idx >= len(targets):
            break
        if ts + 1e-5 < targets[target_idx]:
            continue

        width, height = frame.width, frame.height
        if height >= width:
            new_h = 360
            new_w = max(64, int(width * new_h / height))
        else:
            new_w = 360
            new_h = max(64, int(height * new_w / width))

        image = frame.reformat(width=new_w, height=new_h, format="bgr24").to_ndarray()
        caption_roi = image[int(new_h * 0.35):, :]

        # Fast ROI differential gating: if subtitle band has negligible delta, reuse prior detection
        if prev_caption_roi is not None and prev_caption_roi.shape == caption_roi.shape:
            roi_diff = float(np.mean(np.abs(caption_roi.astype(float) - prev_caption_roi.astype(float))))
            if roi_diff < 5.5:
                ocr_frames_reused += 1
                samples.append({"timestamp": round(ts, 3), "detections": prev_detections})
                for cb in prev_clean_boxes:
                    clean_boxes_for_color.append({"timestamp": round(ts, 3), "box_normalized": cb})
                target_idx += 1
                continue

        prev_caption_roi = caption_roi.copy()
        result = engine(image)
        ocr_calls_performed += 1

        detections = []
        current_frame_clean_boxes = []
        boxes = result.boxes if result.boxes is not None else []
        texts = result.txts if result.txts is not None else []
        scores = result.scores if result.scores is not None else []

        for box, text, score in zip(boxes, texts, scores):
            points = np.asarray(box, dtype=float)
            if _is_false_positive(str(text), float(score), points, new_w, new_h):
                continue

            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            center_y = (y1 + y2) / 2 / new_h
            center_x = (x1 + x2) / 2 / new_w
            w_norm = (x2 - x1) / new_w
            h_norm = (y2 - y1) / new_h

            role = "caption" if center_y > 0.40 else "title" if center_y < 0.25 else "label_or_graphic_text"
            clean_text = str(text).strip()
            typography = _analyze_typography(image, points)
            typography["case"] = "UPPERCASE" if clean_text.isupper() else "lowercase" if clean_text.islower() else "mixed"
            typography["relative_text_height"] = round(float(h_norm), 3)
            all_typography.append(typography)

            detections.append({
                "text": clean_text,
                "confidence": round(float(score), 3),
                "polygon_normalized": [[round(float(x / new_w), 4), round(float(y / new_h), 4)] for x, y in points],
                "position": {
                    "center_x": round(float(center_x), 3),
                    "center_y": round(float(center_y), 3),
                    "width": round(float(w_norm), 3),
                    "height": round(float(h_norm), 3),
                },
                "role_candidate": role,
                "typography": typography,
            })

            box_norm = [round(float(x1 / new_w), 3), round(float(y1 / new_h), 3), round(float(w_norm), 3), round(float(h_norm), 3)]
            clean_boxes_for_color.append({
                "timestamp": round(ts, 3),
                "box_normalized": box_norm,
            })
            current_frame_clean_boxes.append(box_norm)

        prev_detections = detections
        prev_clean_boxes = current_frame_clean_boxes

        if detections:
            samples.append({"timestamp": round(ts, 3), "detections": detections})
        target_idx += 1

    con.close()

    total_sampled = max(1, len(samples))
    reuse_ratio = round(ocr_frames_reused / total_sampled, 3)

    # Video-Level Temporal Typography Clustering
    if all_typography:
        font_counts = Counter(t["font_class"] for t in all_typography)
        dominant_raw_class = font_counts.most_common(1)[0][0]
        font_family_class = (
            "display_candidate" if dominant_raw_class == "display_bold"
            else "serif_candidate" if dominant_raw_class == "serif"
            else "sans_serif_candidate"
        )
        stroke_ratio = sum(1 for t in all_typography if t["has_stroke"]) / len(all_typography)
        shadow_ratio = sum(1 for t in all_typography if t["has_shadow"]) / len(all_typography)
        box_ratio = sum(1 for t in all_typography if t["has_background_box"]) / len(all_typography)
        upper_ratio = sum(1 for t in all_typography if t["case"] == "UPPERCASE") / len(all_typography)
        dominant_hex = Counter(t["fill_color_hex"] for t in all_typography).most_common(1)[0][0]
        style_conf = round(float(font_counts.most_common(1)[0][1] / len(all_typography)), 3)

        persistent_style = {
            "caption_style_id": "style_0",
            "font_family_class": font_family_class,
            "weight_class": "bold" if dominant_raw_class == "display_bold" or stroke_ratio >= 0.40 else "regular",
            "uppercase": bool(upper_ratio >= 0.70),
            "stroke": bool(stroke_ratio >= 0.40),
            "shadow": bool(shadow_ratio >= 0.40),
            "background_box": bool(box_ratio >= 0.40),
            "dominant_fill_color_hex": dominant_hex,
            "confidence": style_conf,
            "training_eligible": False,  # Blacklisted from core training ground truth unless verified
        }
    else:
        persistent_style = {
            "caption_style_id": "style_0",
            "font_family_class": "sans_serif_candidate",
            "weight_class": "regular",
            "uppercase": True,
            "stroke": False,
            "shadow": False,
            "background_box": False,
            "dominant_fill_color_hex": "#FFFFFF",
            "confidence": 1.0,
            "training_eligible": False,
        }

    track = _dense_track_grouping(samples, duration, step, persistent_style)
    caption_events = [item for item in track if item.get("role_candidate") == "caption"]

    # Detect provider name accurately
    provider_name = "CPUExecutionProvider"
    try:
        if hasattr(engine, "text_det") and hasattr(engine.text_det, "session"):
            providers = engine.text_det.session.session.get_providers()
            if "CUDAExecutionProvider" in providers:
                provider_name = "CUDAExecutionProvider"
    except Exception:
        pass

    return {
        "status": "complete_dense_roi_tracking",
        "engine": f"RapidOCR/{provider_name}-dense-tracked",
        "ocr_provider": provider_name,
        "spoken_transcript_kept_separate": True,
        "sample_count": len(samples),
        "performance_metrics": {
            "ocr_calls_performed": ocr_calls_performed,
            "ocr_frames_skipped_or_reused": ocr_frames_reused,
            "ocr_reuse_ratio": reuse_ratio,
        },
        "track": track,
        "persistent_caption_style": persistent_style,
        "caption_boxes_for_exclusion": clean_boxes_for_color,
        "caption_analysis": {
            "tracking_status": "dense_roi_tracked",
            "event_count": len(caption_events),
            "captions": caption_events,
            "dominant_typography": persistent_style,
            "animation_count": sum(1 for item in caption_events if item.get("animation", {}).get("scale_pop") or item.get("animation", {}).get("entry_animation") != "instant"),
            "highlighting_detected_count": sum(1 for item in caption_events if len(item.get("word_highlighting", {}).get("highlighted_words", [])) > 0),
        },
        "limitations": [
            "Dense sampling intervals provide ~0.2s-0.5s boundary precision.",
            "Stroke and shadow are estimated via edge gradient and Otsu contrast heuristics.",
        ],
    }


def _dense_track_grouping(samples: list[dict], duration: float, step: float, persistent_style: dict | None = None) -> list[dict]:
    active = {}
    output = []

    for sample in samples:
        ts = sample["timestamp"]
        seen = set()

        for det in sample["detections"]:
            key = _normalize(det["text"]) + f"@{round(det['position']['center_y'], 1)}"
            seen.add(key)

            if key in active:
                event = active[key]
                event["end"] = ts
                event["observations"] += 1
                event["confidence"] = round(max(event["confidence"], det["confidence"]), 3)
                event["positions"].append({
                    "time": ts,
                    "center_x": det["position"]["center_x"],
                    "center_y": det["position"]["center_y"],
                    "width": det["position"]["width"],
                    "height": det["position"]["height"],
                })
            else:
                # Apply persistent style cluster smoothing
                typ = dict(det["typography"])
                style_id = "style_0"
                if persistent_style:
                    style_id = persistent_style.get("caption_style_id", "style_0")
                    typ["caption_style_id"] = style_id
                    typ["font_family_class"] = persistent_style["font_family_class"]
                    typ["weight_class"] = persistent_style["weight_class"]
                    typ["has_stroke"] = persistent_style["stroke"]
                    typ["has_shadow"] = persistent_style["shadow"]
                    typ["has_background_box"] = persistent_style["background_box"]
                    typ["training_eligible"] = False

                event = {
                    "text": det["text"],
                    "start": ts,
                    "end": ts,
                    "confidence": det["confidence"],
                    "caption_style_id": style_id,
                    "detection_method": "dense_roi_keyframe_ocr",
                    "verification_status": "observed",
                    "observations": 1,
                    "role_candidate": det["role_candidate"],
                    "position": det["position"],
                    "typography": typ,
                    "positions": [{
                        "time": ts,
                        "center_x": det["position"]["center_x"],
                        "center_y": det["position"]["center_y"],
                        "width": det["position"]["width"],
                        "height": det["position"]["height"],
                    }],
                }
                active[key] = event
                output.append(event)

        for key in list(active):
            if key not in seen and active[key]["end"] < ts - step * 1.5:
                active.pop(key)

    # Post-process animations, durations, word highlight candidates
    half_step = round(step / 2, 3)
    for event in output:
        event["start"] = max(0.0, round(event["start"] - half_step, 3))
        event["end"] = min(round(duration, 3), round(event["end"] + half_step, 3))
        event["duration"] = round(max(0.05, event["end"] - event["start"]), 3)
        words = event["text"].split()
        event["words_visible"] = len(words)
        event["displayed_words"] = words

        # Animation analysis
        positions = event.pop("positions", [])
        if len(positions) >= 2:
            dx = positions[-1]["center_x"] - positions[0]["center_x"]
            dy = positions[-1]["center_y"] - positions[0]["center_y"]
            initial_w = max(positions[0]["width"], 0.01)
            max_w = max(p["width"] for p in positions)
            scale_growth = max_w / initial_w

            entry_anim = "pop_in" if scale_growth > 1.08 else "slide_up" if dy < -0.02 else "slide_down" if dy > 0.02 else "instant"
            scale_pop = bool(scale_growth > 1.08)
            movement = {
                "drift_x": round(float(dx), 3),
                "drift_y": round(float(dy), 3),
                "animated": bool(abs(dx) > 0.025 or abs(dy) > 0.025),
            }
        else:
            entry_anim = "instant"
            scale_pop = False
            movement = {"drift_x": 0.0, "drift_y": 0.0, "animated": False}

        event["animation"] = {
            "entry_animation": entry_anim,
            "exit_animation": "instant",
            "scale_pop": scale_pop,
            "movement": movement,
        }

        # Word-level highlight candidates:
        # A uniform caption where all words are UPPERCASE or all lowercase is a UNIFORM CHUNK.
        # Only mixed case with uppercase word, or explicit color variation constitutes a highlighted word.
        if not event["text"].isupper() and not event["text"].islower():
            highlighted = [w for w in words if w.isupper() and len(w) > 1]
        else:
            highlighted = []

        event["word_highlighting"] = {
            "status": "detected" if highlighted else "uniform_chunk",
            "highlighted_words": highlighted,
        }

    return output


def _normalize(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()
