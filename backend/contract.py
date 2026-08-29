"""Report lifecycle, validation, and hardware capability provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

LIFECYCLES={"not_requested","queued","running","complete","partial","deferred","failed"}
VERSION="2026.08.evidence-first"

def content_hash(path: Path) -> str:
    digest=hashlib.blake2s(digest_size=16)
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def runtime_profile() -> dict:
    profile={"decode_path":"pyav_software","cuda":{"available":False,"status":"not_detected"}}
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],capture_output=True,text=True,timeout=3)
        if result.returncode==0 and result.stdout.strip():
            profile["cuda"]={"available":True,"status":"detected","gpus":[line.strip() for line in result.stdout.splitlines()]}
            profile["decode_path"]="software_fallback_cuda_runtime_detected"
        else: profile["cuda"]["reason"]=(result.stderr or result.stdout).strip()[:240]
    except (FileNotFoundError,subprocess.TimeoutExpired) as error: profile["cuda"]["reason"]=str(error)
    return profile

def validate_report(report: dict) -> None:
    required={"report_id","source","processing","transcript","visual","color","audio","text_overlay","editing","semantic","edit_intent","cross_modal_events","training_features","confidence","deferred"}
    missing=required-set(report)
    if missing: raise ValueError(f"missing report keys: {sorted(missing)}")
    duration=float(report["source"].get("duration_seconds",0))
    if duration<0: raise ValueError("negative duration")
    for word in report.get("transcript",{}).get("words",[]):
        start=float(word.get("aligned_start",word.get("start",0))); end=float(word.get("aligned_end",word.get("end",0)))
        if end<=start: raise ValueError(f"invalid word interval: {word.get('display',word.get('word'))}")
        if start<0 or end>duration+.25: raise ValueError("word timestamp outside source")
        confidence=word.get("confidence")
        if confidence is not None and not 0<=float(confidence)<=1: raise ValueError("word confidence out of range")
    for event in report.get("editing",{}).get("verified_events",[]):
        if event.get("verification_status")!="verified": raise ValueError("unverified event in verified_events")
        for key in ("candidate_confidence","verification_confidence","final_confidence"):
            if key not in event or not 0<float(event[key])<1: raise ValueError(f"invalid {key}")
    if report["editing"].get("summary",{}).get("cut_count") is not None:
        count=sum(item.get("type") in {"hard_cut","jump_cut"} for item in report["editing"].get("verified_events",[]))
        if count!=report["editing"]["summary"]["cut_count"]: raise ValueError("cut summary disagrees with verified events")

def atomic_json_write(path: Path, report: dict, compact: bool=False) -> None:
    validate_report(report)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(report,separators=(",",":") if compact else None,indent=None if compact else 2),encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)

def reconcile_states(report: dict) -> dict:
    """Prevent a completed subsystem from simultaneously being advertised as deferred."""
    deferred=set(report.get("deferred",[]))
    if report.get("transcript",{}).get("status")=="complete":
        report["audio"].get("observed",{}).get("speech",{}).update({"status":"aligned_to_transcript"})
    if report.get("text_overlay",{}).get("status","").startswith("complete"):
        deferred.discard("ocr")
    if report.get("semantic",{}).get("status")!="deferred": deferred.discard("semantic_video_model")
    report["deferred"]=sorted(deferred)
    return report
