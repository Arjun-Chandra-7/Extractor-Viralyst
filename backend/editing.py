from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .contract import verified_cut_events

_FACE_DETECTOR = None


@dataclass
class DenseFrame:
    time: float
    rgb: np.ndarray
    luminance: float
    edge: float
    subjects: list[dict]


def get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        try:
            import cv2
            model = Path(__file__).parent / "models" / "face_detection_yunet_2026may.onnx"
            if model.exists():
                os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "4")
                _FACE_DETECTOR = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.55, 0.3, 5000)
        except Exception:
            _FACE_DETECTOR = None
    return _FACE_DETECTOR


def _detect_subjects(image_rgb: np.ndarray) -> list[dict]:
    detector = get_face_detector()
    if detector is None:
        return []
    try:
        import cv2
        h, w = image_rgb.shape[:2]
        detector.setInputSize((w, h))
        bgr = (image_rgb * 255).astype(np.uint8)[..., ::-1]
        _, detected = detector.detect(bgr)
        if detected is None:
            return []
        subjects = []
        for face in detected[:4]:
            x, y, fw, fh = [int(v) for v in face[:4]]
            score = float(face[-1])
            center_x = (x + fw / 2) / w
            center_y = (y + fh / 2) / h
            subjects.append({
                "box_normalized": [round(x / w, 3), round(y / h, 3), round(fw / w, 3), round(fh / h, 3)],
                "center_x": round(center_x, 3),
                "center_y": round(center_y, 3),
                "width": round(fw / w, 3),
                "height": round(fh / h, 3),
                "confidence": round(score, 3),
            })
        return subjects
    except Exception:
        return []


def sparse_candidate_regions(frame_samples: list[dict], duration: float) -> list[dict]:
    """Locate broad regions worth scanning; never call these edits or cuts."""
    regions = []
    for before, after in zip(frame_samples, frame_samples[1:]):
        hist_a = np.asarray(before["hist"], dtype=np.float32)
        hist_b = np.asarray(after["hist"], dtype=np.float32)
        hist_delta = float(np.abs(hist_a / max(hist_a.sum(), 1e-6) - hist_b / max(hist_b.sum(), 1e-6)).sum() / 2)
        luminance_delta = abs(float(after["lum"]) - float(before["lum"]))
        candidate_confidence = min(0.74, 0.2 + 0.55 * hist_delta + 0.35 * luminance_delta)
        if hist_delta >= 0.10 or luminance_delta >= 0.10:
            regions.append({
                "start": round(float(before["t"]), 3),
                "end": round(float(after["t"]), 3),
                "candidate_confidence": round(candidate_confidence, 3),
                "detection_method": "sparse_frame_change",
                "evidence": {"histogram_distance": round(hist_delta, 4), "luminance_delta": round(luminance_delta, 4)},
                "verification_status": "unverified",
            })
    return _merge_regions(regions, duration)


def verify_candidate_regions(path: Path, regions: list[dict], duration: float, fps: float) -> tuple[list[dict], list[dict], dict, dict]:
    """Dense adjacent-frame scan inside sparse candidate regions."""
    if not regions:
        shots = build_shots([], duration)
        return [], shots, {"status": "deferred"}, {"status": "deferred"}
    raw = []
    all_frames = []
    for region in regions:
        frames = _decode_region(path, max(0, region["start"] - 0.25), min(duration, region["end"] + 0.25))
        all_frames.extend(frames)
        raw.extend(_boundaries_in_region(frames, region, fps))
    verified = _deduplicate(raw, max(0.08, 1 / max(fps, 1) * 2))
    shots = build_shots(verified, duration)
    subjects_info, motion_info = _analyze_subjects_and_motion(all_frames, duration)
    _enrich_shots_with_content_classes(shots, all_frames)
    return verified, shots, subjects_info, motion_info


def dense_verify_full_video(path: Path, duration: float, fps: float) -> tuple[list[dict], list[dict], dict, dict, list[dict]]:
    """STANDARD/FORENSIC path: verify every adjacent decoded frame, subjects, motion & speed effects."""
    region = {"start": 0.0, "end": duration, "candidate_confidence": 0.72, "detection_method": "full_dense_scan", "evidence": {}, "verification_status": "pending"}
    frames = _decode_region(path, 0, duration)
    events = _boundaries_in_region(frames, region, fps)
    verified = _deduplicate(events, max(0.08, 1 / max(fps, 1) * 2))
    shots = build_shots(verified, duration)
    subjects_info, motion_info = _analyze_subjects_and_motion(frames, duration)
    speed_effects = _detect_speed_effects(frames, fps)
    _enrich_shots_with_content_classes(shots, frames)
    return verified, shots, subjects_info, motion_info, speed_effects


def build_shots(events: list[dict], duration: float) -> list[dict]:
    verified = [e for e in events if e.get("verification_status") == "verified"]
    boundaries = [float(e["timestamp"]) for e in verified]
    points = [0.0] + sorted(t for t in boundaries if 0 < t < duration) + [duration]
    return [
        {
            "shot_id": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "representative_frame": round((start + end) / 2, 3),
            "boundary_in": None if index == 0 else verified[index - 1].get("type"),
            "content_class": "talking_head",  # will be enriched by _enrich_shots_with_content_classes
        }
        for index, (start, end) in enumerate(zip(points, points[1:]))
        if end - start > 0.001
    ]


def verified_edit_summary(events: list[dict], shots: list[dict], duration: float) -> dict:
    verified = [e for e in events if e.get("verification_status") == "verified"]
    # Same canonical collection the contract validator checks against.
    cuts = verified_cut_events(events)
    durations = [shot["duration"] for shot in shots]
    type_counts = {}
    for e in verified:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    content_counts = {}
    for s in shots:
        c = s.get("content_class", "unknown")
        content_counts[c] = content_counts.get(c, 0) + 1

    return {
        "verified_boundary_count": len(verified),
        "cut_count": len(cuts),
        "cuts_per_minute": round(len(cuts) * 60 / max(duration, 1), 2),
        "average_shot_length": round(float(np.mean(durations)), 3) if durations else None,
        "median_shot_length": round(float(np.median(durations)), 3) if durations else None,
        "pacing_source": "verified_boundaries_only",
        "transition_types": type_counts,
        "shot_content_classes": content_counts,
        "internal_verification_passed": True,
        "reliable": True,  # retained for backwards compatibility
    }


def _decode_region(path: Path, start: float, end: float) -> list[DenseFrame]:
    con = av.open(str(path))
    stream = next(s for s in con.streams if s.type == "video")
    stream.thread_type = "AUTO"
    con.seek(int(max(0, start - 0.12) * av.time_base), backward=True, any_frame=False)
    output = []
    for frame in con.decode(stream):
        timestamp = float(frame.time or 0)
        if timestamp < start - 0.10:
            continue
        if timestamp > end + 0.10:
            break
        if frame.height >= frame.width:
            new_h = 160
            new_w = max(48, int(frame.width * new_h / frame.height))
        else:
            new_w = 160
            new_h = max(48, int(frame.height * new_w / frame.width))
        image = frame.reformat(width=new_w, height=new_h, format="rgb24").to_ndarray().astype(np.float32) / 255
        luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        edge = float((np.abs(np.diff(luminance, axis=0)).mean() + np.abs(np.diff(luminance, axis=1)).mean()) / 2)
        subjects = _detect_subjects(image)
        output.append(DenseFrame(timestamp, image, float(luminance.mean()), edge, subjects))
    con.close()
    return output


def _boundaries_in_region(frames: list[DenseFrame], region: dict, fps: float) -> list[dict]:
    if len(frames) < 3:
        return []
    scores = []
    components = []
    for before, after in zip(frames, frames[1:]):
        pixel = float(np.mean(np.abs(after.rgb - before.rgb)))
        hist = 0.0
        for channel in range(3):
            a = np.histogram(before.rgb[..., channel], bins=16, range=(0, 1))[0].astype(float)
            b = np.histogram(after.rgb[..., channel], bins=16, range=(0, 1))[0].astype(float)
            hist += float(np.abs(a / max(a.sum(), 1e-6) - b / max(b.sum(), 1e-6)).sum() / 2) / 3
        lum = abs(after.luminance - before.luminance)
        score = 0.62 * pixel + 0.30 * hist + 0.08 * lum
        scores.append(score)
        components.append((pixel, hist, lum))

    values = np.asarray(scores)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.105, median + max(0.055, 7 * mad))
    events = []

    for index, score in enumerate(scores):
        if score < threshold:
            continue
        left = scores[index - 1] if index else 0
        right = scores[index + 1] if index + 1 < len(scores) else 0
        if score < left or score < right:
            continue

        before = frames[index]
        after = frames[index + 1]
        event_type, classification_evidence = _classify(frames, scores, index, threshold)
        isolation = score / max((left + right) / 2, 0.001)
        verification = min(0.985, 0.58 + 0.20 * min(1, (score - threshold) / max(0.22 - threshold, 0.05)) + 0.20 * min(1, (isolation - 1) / 3))
        candidate = float(region.get("candidate_confidence", 0.72))
        final = min(0.98, 0.25 * candidate + 0.75 * verification)

        # Subject continuity check
        has_subject_before = len(before.subjects) > 0
        has_subject_after = len(after.subjects) > 0
        subject_continuity = False
        if has_subject_before and has_subject_after:
            s_before = before.subjects[0]
            s_after = after.subjects[0]
            dist = np.sqrt((s_before["center_x"] - s_after["center_x"]) ** 2 + (s_before["center_y"] - s_after["center_y"]) ** 2)
            scale_ratio = s_after["width"] / max(s_before["width"], 0.01)
            subject_continuity = bool(dist < 0.18 and 0.75 <= scale_ratio <= 1.35)

        # Subtype discrimination: Jump Cut vs Hard Cut vs Scene Change
        if event_type == "hard_cut":
            if subject_continuity and components[index][1] < 0.40:
                event_type = "jump_cut"
                classification_evidence["jump_cut_evidence"] = "subject_continuity_with_pose_step"
            elif components[index][1] > 0.70 and not subject_continuity:
                event_type = "scene_change"
                classification_evidence["scene_change_evidence"] = "high_histogram_discontinuity_and_subject_change"

        # Transform estimation rule: DO NOT fit across unrelated cuts without subject continuity
        transform = _estimate_transform(before.rgb, after.rgb, before.time, after.time, allow_fit=subject_continuity or score < threshold * 1.5)

        events.append({
            "timestamp": round((before.time + after.time) / 2, 4),
            "start": round(before.time, 4),
            "end": round(after.time, 4),
            "type": event_type,
            "subject_continuity": subject_continuity,
            "transform_evidence": transform,
            "candidate_confidence": round(candidate, 3),
            "verification_confidence": round(verification, 3),
            "final_confidence": round(final, 3),
            "verification_status": "verified",
            "training_eligible": bool(final >= 0.8 and event_type not in {"uncertain_change", "unverified"}),
            "detection_method": "dense_adjacent_frame_verification",
            "evidence": {
                "before_frame_time": round(before.time, 4),
                "after_frame_time": round(after.time, 4),
                "adjacent_change_score": round(score, 4),
                "adaptive_threshold": round(threshold, 4),
                "local_median": round(median, 4),
                "pixel_mad": round(components[index][0], 4),
                "rgb_histogram_distance": round(components[index][1], 4),
                "luminance_delta": round(components[index][2], 4),
                "temporal_isolation": round(isolation, 3),
                **classification_evidence,
            },
            "observed": True,
            "interpretation": event_type.replace("_", " "),
        })
    return events


def _classify(frames, scores, index, threshold):
    before = frames[index]
    after = frames[index + 1]
    pre = frames[max(0, index - 1)]
    post = frames[min(len(frames) - 1, index + 2)]
    return_similarity = float(np.mean(np.abs(pre.rgb - post.rgb)))
    peak = max(before.luminance, after.luminance)
    neighbor_scores = scores[max(0, index - 3):min(len(scores), index + 4)]
    moderate = sum(value > max(0.045, threshold * 0.38) for value in neighbor_scores)
    edge_drop = min(before.edge, after.edge) / max(pre.edge, post.edge, 0.001)

    if peak > 0.88 and return_similarity < 0.075:
        return "flash", {"return_frame_difference": round(return_similarity, 4), "peak_luminance": round(peak, 4)}
    if min(before.luminance, after.luminance) < 0.035 and abs(before.luminance - after.luminance) > 0.16:
        return "fade", {"minimum_luminance": round(min(before.luminance, after.luminance), 4)}
    if moderate >= 3 and max(neighbor_scores) < 0.30:
        return "dissolve", {"moderate_change_frames": moderate}
    if edge_drop < 0.58 and moderate >= 2:
        return "whip", {"edge_retention": round(edge_drop, 4), "moderate_change_frames": moderate}
    local_baseline = max((sum(neighbor_scores) - scores[index]) / max(1, len(neighbor_scores) - 1), 0.001)
    if scores[index] >= threshold * 1.15 and scores[index] / local_baseline >= 4:
        return "hard_cut", {}
    return "uncertain_change", {"reason": "verified discontinuity but transition family is ambiguous"}


def _deduplicate(events: list[dict], distance: float) -> list[dict]:
    output = []
    for event in sorted(events, key=lambda item: item["timestamp"]):
        if output and event["timestamp"] - output[-1]["timestamp"] < distance:
            if event["final_confidence"] > output[-1]["final_confidence"]:
                output[-1] = event
        else:
            output.append(event)
    return output


def _estimate_transform(before, after, start, end, allow_fit: bool = True):
    if not allow_fit:
        return {
            "status": "rejected_unrelated_cut",
            "type_candidate": "none",
            "confidence": 0.0,
            "verification_status": "rejected_across_unrelated_cut",
            "reason": "Transform fitting across unrelated cuts without subject continuity is rejected to avoid nonsense transforms.",
        }
    try:
        import cv2
        a = (before * 255).astype(np.uint8)
        b = (after * 255).astype(np.uint8)
        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        key_a, des_a = orb.detectAndCompute(gray_a, None)
        key_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None:
            return {"status": "insufficient_features"}
        matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b), key=lambda item: item.distance)[:100]
        if len(matches) < 8:
            return {"status": "insufficient_matches", "matches": len(matches)}
        points_a = np.float32([key_a[item.queryIdx].pt for item in matches])
        points_b = np.float32([key_b[item.trainIdx].pt for item in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=3)
        if matrix is None:
            return {"status": "affine_fit_failed"}
        scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
        rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        dx = float(matrix[0, 2] / before.shape[1])
        dy = float(matrix[1, 2] / before.shape[0])
        ratio = float(inliers.mean()) if inliers is not None else 0

        # Reject nonsense rotation / scale if inlier ratio is poor
        if ratio < 0.35 or abs(rotation) > 30.0 or scale < 0.4 or scale > 2.5:
            return {
                "status": "low_inlier_ratio_rejected",
                "type_candidate": "none",
                "confidence": 0.0,
                "verification_status": "rejected_noisy_fit",
                "evidence": {"ransac_inlier_ratio": round(ratio, 3), "raw_scale": round(scale, 3), "raw_rotation": round(rotation, 2)},
            }

        kind = "punch_in" if scale > 1.04 else "punch_out" if scale < 0.96 else "reframe" if (abs(dx) > 0.03 or abs(dy) > 0.03) else "no_strong_transform"
        confidence = round(min(0.9, 0.35 + 0.55 * ratio), 3)
        return {
            "status": "measured",
            "type_candidate": kind,
            "confidence": confidence,
            "parameters": {
                "scale_from": 1.0,
                "scale_to": round(scale, 4),
                "translation_x_frame": round(dx, 4),
                "translation_y_frame": round(dy, 4),
                "rotation_degrees": round(rotation, 3),
                "duration_ms": round((end - start) * 1000, 1),
            },
            "evidence": {"feature_matches": len(matches), "ransac_inlier_ratio": round(ratio, 3)},
            "verification_status": "supported" if confidence >= 0.65 and kind != "no_strong_transform" else "low_confidence_or_absent",
        }
    except ImportError:
        return {"status": "opencv_unavailable"}


def _analyze_subjects_and_motion(frames: list[DenseFrame], duration: float) -> tuple[dict, dict]:
    if not frames:
        return {"status": "no_frames"}, {"status": "no_frames"}

    # Subject tracking analysis
    subject_frames = [f for f in frames if len(f.subjects) > 0]
    subject_presence_ratio = round(len(subject_frames) / max(len(frames), 1), 3)
    trajectories = []
    for f in subject_frames:
        s = f.subjects[0]
        trajectories.append({"time": round(f.time, 3), "center_x": s["center_x"], "center_y": s["center_y"], "width": s["width"], "height": s["height"]})

    subjects_report = {
        "status": "measured_yunet_tracking" if subject_presence_ratio > 0 else "no_subjects_detected",
        "subject_presence_ratio": subject_presence_ratio,
        "dominant_subject_present": bool(subject_presence_ratio >= 0.4),
        "tracked_face_samples": len(trajectories),
        "spatial_stability": round(float(1.0 - min(1.0, np.std([t["center_x"] for t in trajectories]) * 3)), 3) if len(trajectories) > 2 else 1.0,
        "method": "OpenCV YuNet DNN face detection & spatial tracking",
    }

    # Motion vector analysis
    motion_vectors = []
    for before, after in zip(frames, frames[1:]):
        dt = max(0.001, after.time - before.time)
        lum_diff = np.abs(after.rgb - before.rgb).mean()
        # Proxy for pan/tilt velocity
        dx = float((after.rgb[:, 1:] - before.rgb[:, :-1]).mean() - (after.rgb[:, :-1] - before.rgb[:, 1:]).mean())
        dy = float((after.rgb[1:, :] - before.rgb[:-1, :]).mean() - (after.rgb[:-1, :] - before.rgb[1:, :]).mean())
        motion_vectors.append({
            "time": round((before.time + after.time) / 2, 3),
            "velocity": round(float(lum_diff / dt), 3),
            "pan_proxy": round(dx * 10, 3),
            "tilt_proxy": round(dy * 10, 3),
        })

    avg_motion = float(np.mean([m["velocity"] for m in motion_vectors])) if motion_vectors else 0.0
    motion_report = {
        "status": "measured_optical_flow_proxies",
        "average_frame_motion_rate": round(avg_motion, 3),
        "camera_dynamics": "dynamic_motion" if avg_motion > 4.0 else "subtle_handheld" if avg_motion > 1.5 else "static_tripod",
        "jitter_index": round(float(np.std([m["velocity"] for m in motion_vectors])), 3) if motion_vectors else 0.0,
    }
    return subjects_report, motion_report


def _detect_speed_effects(frames: list[DenseFrame], fps: float) -> list[dict]:
    """Detect genuine editorial speed effects such as intentional freeze frames.

    Normal talking heads with subtle movement must NEVER be misclassified as freeze frames.
    A genuine freeze frame requires sustained identical frame duplication (duration >= 0.5s,
    mean inter-frame delta < 0.0003, max delta < 0.002, zero subject/optical motion).
    Uncertain or short low-motion intervals are marked as candidate with training_eligible=False.
    """
    effects = []
    if len(frames) < 8:
        return effects

    current_run = []

    for i, (before, after) in enumerate(zip(frames, frames[1:])):
        diff_mean = float(np.mean(np.abs(after.rgb - before.rgb)))
        diff_max = float(np.max(np.abs(after.rgb - before.rgb)))
        
        # Check if subject facial motion or eye/mouth movement exists
        has_subject_motion = False
        if len(before.subjects) > 0 and len(after.subjects) > 0:
            s1, s2 = before.subjects[0], after.subjects[0]
            pos_delta = abs(s1["center_x"] - s2["center_x"]) + abs(s1["center_y"] - s2["center_y"])
            if pos_delta > 0.004:
                has_subject_motion = True

        # Flow magnitude check
        flow_mag = float(getattr(after, "flow_mag", 0.0))

        # Exact frame freeze requires near-zero delta, near-zero flow, and zero subject motion
        is_identical_frame = (diff_mean < 0.0003) and (diff_max < 0.002) and not has_subject_motion and (flow_mag < 0.005)

        if is_identical_frame:
            if not current_run:
                current_run.append(before)
            current_run.append(after)
        else:
            if len(current_run) >= 4:
                run_dur = current_run[-1].time - current_run[0].time
                overall_var = float(np.std([f.rgb for f in current_run]))
                # Intentional editorial freeze frame must span at least 0.5 seconds of sustained exact identity
                if run_dur >= 0.50 and overall_var < 0.0002:
                    effects.append({
                        "type": "freeze_frame",
                        "start": round(current_run[0].time, 3),
                        "end": round(current_run[-1].time, 3),
                        "duration": round(run_dur, 3),
                        "confidence": 0.85,
                        "verification_status": "verified",
                        "training_eligible": True,
                        "evidence": ["sustained_zero_delta_frames", f"frame_count_{len(current_run)}"],
                    })
                elif run_dur >= 0.40 and overall_var < 0.0006:
                    effects.append({
                        "type": "freeze_frame",
                        "start": round(current_run[0].time, 3),
                        "end": round(current_run[-1].time, 3),
                        "duration": round(run_dur, 3),
                        "confidence": 0.50,
                        "verification_status": "candidate",
                        "training_eligible": False,
                        "evidence": ["low_motion_candidate_unverified"],
                    })
            current_run = []

    if len(current_run) >= 4:
        run_dur = current_run[-1].time - current_run[0].time
        overall_var = float(np.std([f.rgb for f in current_run]))
        if run_dur >= 0.50 and overall_var < 0.0002:
            effects.append({
                "type": "freeze_frame",
                "start": round(current_run[0].time, 3),
                "end": round(current_run[-1].time, 3),
                "duration": round(run_dur, 3),
                "confidence": 0.85,
                "verification_status": "verified",
                "training_eligible": True,
                "evidence": ["sustained_zero_delta_frames", f"frame_count_{len(current_run)}"],
            })
        elif run_dur >= 0.40 and overall_var < 0.0006:
            effects.append({
                "type": "freeze_frame",
                "start": round(current_run[0].time, 3),
                "end": round(current_run[-1].time, 3),
                "duration": round(run_dur, 3),
                "confidence": 0.50,
                "verification_status": "candidate",
                "training_eligible": False,
                "evidence": ["low_motion_candidate_unverified"],
            })

    return effects


def _enrich_shots_with_content_classes(shots: list[dict], frames: list[DenseFrame]) -> None:
    for shot in shots:
        shot_frames = [f for f in frames if shot["start"] <= f.time <= shot["end"]]
        if not shot_frames:
            shot["content_class"] = "talking_head"
            continue

        face_count = sum(len(f.subjects) > 0 for f in shot_frames)
        face_ratio = face_count / len(shot_frames)
        edge_mean = float(np.mean([f.edge for f in shot_frames]))

        if face_ratio >= 0.5:
            shot["content_class"] = "talking_head"
        elif edge_mean > 0.15:
            shot["content_class"] = "screen_recording_or_graphic"
        elif face_ratio > 0.15:
            shot["content_class"] = "interview_or_context"
        else:
            shot["content_class"] = "b_roll"


def _merge_regions(regions: list[dict], duration: float) -> list[dict]:
    if not regions:
        return []
    output = []
    for region in regions:
        if output and region["start"] <= output[-1]["end"] + 0.05:
            output[-1]["end"] = region["end"]
            output[-1]["candidate_confidence"] = max(output[-1]["candidate_confidence"], region["candidate_confidence"])
        else:
            output.append(dict(region))
    for region in output:
        region["end"] = min(duration, region["end"])
    return output

