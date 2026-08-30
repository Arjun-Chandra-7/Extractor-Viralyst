"""Report lifecycle, validation, and hardware capability provenance."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

LIFECYCLES = {"not_requested", "queued", "running", "complete", "partial", "deferred", "failed"}
VERSION = "2026.08.evidence-first"


def content_hash(path: Path) -> str:
    digest = hashlib.blake2s(digest_size=16)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_profile() -> dict:
    profile = {
        "decode_path": "pyav_software",
        "acceleration_path": "cpu_fallback",
        "cuda": {
            "available": False,
            "status": "not_detected",
            "device_count": 0,
            "devices": [],
            "torch_cuda": False,
            "ctranslate2_cuda": False,
            "onnxruntime_providers": [],
        },
    }

    # 1. Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            profile["cuda"]["torch_cuda"] = True
            profile["cuda"]["device_count"] = torch.cuda.device_count()
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                profile["cuda"]["devices"].append({
                    "id": idx,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024 * 1024), 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except Exception as err:
        profile["cuda"]["torch_error"] = str(err)

    # 2. Check CTranslate2 (Faster-Whisper engine) CUDA support
    try:
        import ctranslate2
        ct2_count = ctranslate2.get_cuda_device_count()
        profile["cuda"]["ctranslate2_cuda"] = bool(ct2_count > 0)
        profile["cuda"]["ctranslate2_device_count"] = ct2_count
    except Exception as err:
        profile["cuda"]["ctranslate2_error"] = str(err)

    # 3. Check ONNX Runtime execution providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        profile["cuda"]["onnxruntime_providers"] = providers
    except Exception as err:
        profile["cuda"]["onnxruntime_error"] = str(err)

    # 4. Check nvidia-smi
    smi_detected = False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            smi_detected = True
            profile["cuda"]["nvidia_smi_gpus"] = [line.strip() for line in result.stdout.splitlines()]
        else:
            profile["cuda"]["nvidia_smi_reason"] = (result.stderr or result.stdout).strip()[:240]
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        profile["cuda"]["nvidia_smi_reason"] = str(error)

    # Determine overall status
    is_available = profile["cuda"]["torch_cuda"] or profile["cuda"]["ctranslate2_cuda"] or smi_detected
    profile["cuda"]["available"] = is_available
    if is_available:
        profile["cuda"]["status"] = "detected"
        profile["decode_path"] = "software_decode_with_cuda_accelerated_workers"
        profile["acceleration_path"] = "cuda_accelerated"
    else:
        profile["cuda"]["status"] = "not_detected"

    # Subsystem-level execution mapping
    ocr_ep = "CUDAExecutionProvider" if "CUDAExecutionProvider" in profile["cuda"]["onnxruntime_providers"] else "CPUExecutionProvider"
    profile["subsystems"] = {
        "video_decode": "software",
        "asr_device": "cuda" if profile["cuda"]["ctranslate2_cuda"] else "cpu",
        "asr_engine": "faster-whisper-ctranslate2",
        "ocr_provider": ocr_ep,
        "gpu_name": profile["cuda"]["devices"][0]["name"] if profile["cuda"]["devices"] else "none",
        "gpu_available": is_available,
    }

    return profile


def validate_report(report: dict) -> None:
    """Comprehensive automatic report validation for Core Brain training safety."""
    required = {
        "report_id",
        "source",
        "processing",
        "transcript",
        "visual",
        "color",
        "audio",
        "text_overlay",
        "editing",
        "semantic",
        "edit_intent",
        "cross_modal_events",
        "training_features",
        "confidence",
        "deferred",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"missing report keys: {sorted(missing)}")
    duration = float(report["source"].get("duration_seconds", 0))
    if duration < 0:
        raise ValueError("negative duration")

    # 1. Validate words
    words = report.get("transcript", {}).get("words", [])
    for i, word in enumerate(words):
        start = float(word.get("aligned_start", word.get("start", 0)))
        end = float(word.get("aligned_end", word.get("end", 0)))
        if end <= start:
            raise ValueError(f"invalid word interval: {word.get('display', word.get('word'))}")
        if start < 0 or end > duration + 0.5:
            raise ValueError(f"word timestamp outside source: {start} - {end} (dur: {duration})")
        confidence = word.get("confidence")
        if confidence is not None and not 0 <= float(confidence) <= 1.0:
            raise ValueError(f"word confidence out of range: {confidence}")

    # 2. Validate verified editing events
    for event in report.get("editing", {}).get("verified_events", []):
        if event.get("verification_status") != "verified":
            raise ValueError("unverified event in verified_events")
        for key in ("candidate_confidence", "verification_confidence", "final_confidence"):
            if key in event and not 0 <= float(event[key]) <= 1.0:
                raise ValueError(f"invalid {key}: {event[key]}")

    # 3. Validate edit summary
    summary = report["editing"].get("summary", {})
    if summary.get("cut_count") is not None:
        count = sum(item.get("type") in {"hard_cut", "jump_cut"} for item in report["editing"].get("verified_events", []))
        if count != summary["cut_count"]:
            raise ValueError("cut summary disagrees with verified events")

    # 4. Validate caption alignments
    alignments = report.get("transcript_caption_alignment", {}).get("alignments", [])
    for a in alignments:
        score = a.get("match_score", 0.0)
        if not (0.0 <= float(score) <= 1.0):
            raise ValueError(f"caption match_score out of [0, 1] range: {score}")
        refs = a.get("spoken_word_refs", [])
        if refs:
            is_monotonic = all(refs[i]["index"] <= refs[i + 1]["index"] for i in range(len(refs) - 1))
            if not is_monotonic:
                raise ValueError("non-monotonic spoken word references within caption")
        # Training eligibility safety gate
        if a.get("training_eligible") and a.get("verification_status") != "verified":
            raise ValueError("unverified caption alignment marked training_eligible")
        if a.get("training_eligible") and float(a.get("temporal_error_seconds",0)) > .5:
            raise ValueError("mistimed caption alignment marked training_eligible")

    # 5. Validate speed effects training eligibility
    for effect in report.get("visual", {}).get("speed_effects", []):
        if effect.get("training_eligible") and effect.get("verification_status") != "verified":
            raise ValueError("unverified speed effect marked training_eligible")

    # 6. Validate shots
    shots = report.get("visual", {}).get("shots", [])
    for shot in shots:
        s_start = float(shot.get("start", 0))
        s_end = float(shot.get("end", 0))
        if s_end < s_start:
            raise ValueError(f"negative shot duration: {shot}")
        if s_end > duration + 0.5:
            raise ValueError(f"shot end exceeds video duration: {s_end} > {duration}")


def atomic_json_write(path: Path, report: dict, compact: bool = False, compress_gzip: bool = False) -> None:
    validate_report(report)
    target_path = path.with_suffix(path.suffix + ".gz") if compress_gzip and not path.name.endswith(".gz") else path
    temporary = target_path.with_suffix(target_path.suffix + ".partial")
    json_bytes = json.dumps(
        report,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    ).encode("utf-8")

    if compress_gzip:
        with gzip.open(temporary, "wb") as gz_file:
            gz_file.write(json_bytes)
        # Test read
        with gzip.open(temporary, "rb") as gz_file:
            json.loads(gz_file.read().decode("utf-8"))
    else:
        temporary.write_bytes(json_bytes)
        json.loads(temporary.read_text(encoding="utf-8"))

    temporary.replace(target_path)


def reconcile_states(report: dict) -> dict:
    """Prevent a completed subsystem from simultaneously being advertised as deferred."""
    deferred = set(report.get("deferred", []))
    if report.get("transcript", {}).get("status") == "complete":
        report["audio"].get("observed", {}).get("speech", {}).update({"status": "aligned_to_transcript"})
    if report.get("text_overlay", {}).get("status", "").startswith("complete"):
        deferred.discard("ocr")
        if report.get("text_overlay", {}).get("caption_analysis", {}).get("tracking_status") == "dense_roi_tracked":
            deferred.discard("caption_tracking")
    if report.get("semantic", {}).get("status") not in {"deferred", "not_inferred"}:
        deferred.discard("semantic_video_model")
    if report.get("visual", {}).get("subjects", {}).get("status") not in {"deferred", None}:
        deferred.discard("subject_tracking")
    if report.get("visual", {}).get("motion", {}).get("status") not in {"deferred", None}:
        deferred.discard("motion_analysis")
    if report.get("color", {}).get("measurements", {}).get("graphics_caption_exclusion", {}).get("status") == "applied":
        deferred.discard("graphics_caption_exclusion")
    if report.get("audio", {}).get("observed", {}).get("mix", {}).get("stem_separation_proxy", {}).get("status") == "measured_spectral_bands":
        deferred.discard("stem_separation")
    report["deferred"] = sorted(deferred)
    return report
