from __future__ import annotations

import re
from pathlib import Path

import av
import numpy as np

_OCR_ENGINE = None


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR
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

    # 1. Reject full-screen or massive boxes covering >50% of frame with very short text (e.g. false positive "9")
    if area_norm > 0.50 and len(clean) < 8:
        return True
    if w_norm > 0.85 and h_norm > 0.85 and len(clean) < 15:
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
        # Font classification proxy: high edge density relative to area indicates display bold or serif
        font_class = "display_bold" if edge_density > 0.20 else "serif" if edge_density > 0.14 else "sans_serif"
        has_stroke = bool(edge_density > 0.12 and abs(fg_mean.mean() - bg_rgb.mean()) > 40)
    except Exception:
        font_class = "sans_serif"
        has_stroke = False

    # Background box detection (low variance in background region)
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


def extract_text_overlay(path: Path, duration: float, target_fps: float = 4.0) -> dict:
    """Dense scene-text track with false-positive rejection, ROI tracking, animation & typography."""
    engine = get_ocr_engine()
    if engine is None:
        return {"status": "unavailable", "engine": "rapidocr", "track": [], "reason": "Install rapidocr and onnxruntime"}

    step = max(0.12, 1.0 / max(target_fps, 1.0))
    targets = np.arange(0.0, max(0.01, duration), step)

    con = av.open(str(path))
    stream = next(s for s in con.streams if s.type == "video")
    stream.thread_type = "AUTO"

    samples = []
    target_idx = 0
    clean_boxes_for_color = []

    for frame in con.decode(stream):
        ts = float(frame.time or 0)
        if target_idx >= len(targets):
            break
        if ts + 1e-5 < targets[target_idx]:
            continue

        width, height = frame.width, frame.height
        if height >= width:
            new_h = 540
            new_w = max(64, int(width * new_h / height))
        else:
            new_w = 540
            new_h = max(64, int(height * new_w / width))

        image = frame.reformat(width=new_w, height=new_h, format="bgr24").to_ndarray()
        result = engine(image)

        detections = []
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

            role = "caption" if center_y > 0.45 else "title" if center_y < 0.25 else "label_or_graphic_text"
            clean_text = str(text).strip()
            typography = _analyze_typography(image, points)
            typography["case"] = "UPPERCASE" if clean_text.isupper() else "lowercase" if clean_text.islower() else "mixed"
            typography["relative_text_height"] = round(float(h_norm), 3)

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

            clean_boxes_for_color.append({
                "timestamp": round(ts, 3),
                "box_normalized": [round(float(x1 / new_w), 3), round(float(y1 / new_h), 3), round(float(w_norm), 3), round(float(h_norm), 3)],
            })

        if detections:
            samples.append({"timestamp": round(ts, 3), "detections": detections})
        target_idx += 1

    con.close()

    track = _dense_track_grouping(samples, duration, step)
    caption_events = [item for item in track if item.get("role_candidate") == "caption"]

    return {
        "status": "complete_dense_roi_tracking",
        "engine": "RapidOCR/ONNX-dense-tracked",
        "spoken_transcript_kept_separate": True,
        "sample_count": len(samples),
        "track": track,
        "caption_boxes_for_exclusion": clean_boxes_for_color,
        "caption_analysis": {
            "tracking_status": "dense_roi_tracked",
            "event_count": len(caption_events),
            "captions": caption_events,
            "animation_count": sum(1 for item in caption_events if item.get("animation", {}).get("scale_pop") or item.get("animation", {}).get("entry_animation") != "instant"),
            "highlighting_detected_count": sum(1 for item in caption_events if len(item.get("word_highlighting", {}).get("highlighted_words", [])) > 0),
        },
        "limitations": [
            "Dense sampling intervals provide ~0.2s boundary precision.",
            "Stroke/shadow and font class are estimated via edge gradient and Otsu contrast heuristics.",
        ],
    }


def _dense_track_grouping(samples: list[dict], duration: float, step: float) -> list[dict]:
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
                event = {
                    "text": det["text"],
                    "start": ts,
                    "end": ts,
                    "confidence": det["confidence"],
                    "detection_method": "dense_roi_keyframe_ocr",
                    "verification_status": "observed",
                    "observations": 1,
                    "role_candidate": det["role_candidate"],
                    "position": det["position"],
                    "typography": det["typography"],
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

        # Word-level highlight candidates (e.g. UPPERCASE / distinct styling in short chunks)
        highlighted = [w for w in words if w.isupper() and len(w) > 1] if not event["text"].isupper() else []
        event["word_highlighting"] = {
            "status": "detected" if highlighted else "uniform_chunk",
            "highlighted_words": highlighted,
        }

    return output


def _normalize(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()
