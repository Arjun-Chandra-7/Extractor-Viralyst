from __future__ import annotations

import time
from pathlib import Path

import av
import numpy as np

from .editing import sparse_candidate_regions
from .pipeline import _color, _rgb_hsv, q

TURBO_VIDEO_SAMPLES = 16
TURBO_AUDIO_HZ = 8_000


def analyse_turbo(path: Path, report_id: str, original_name: str) -> dict:
    """Sparse, fixed-work analysis for very large corpora.

    This pass is for triage/training features. Deep reports are scheduled only for
    interesting samples; otherwise 50K workloads waste most compute on redundant video.
    """
    started = time.perf_counter()
    with av.open(str(path)) as con:
        video = next((s for s in con.streams if s.type == "video"), None)
        audio = next((s for s in con.streams if s.type == "audio"), None)
        if video is None:
            raise ValueError("No video stream found")
        duration = float(con.duration / av.time_base) if con.duration else 0.0
        source = {
            "filename": original_name,
            "duration_seconds": round(duration, 3),
            "resolution": f"{video.codec_context.width}×{video.codec_context.height}",
            "fps": round(float(video.average_rate), 3) if video.average_rate else 0,
            "codec": video.codec_context.name,
            "has_audio": audio is not None,
        }
    frames = _sparse_frames(path, duration)
    candidate_regions = sparse_candidate_regions(frames, duration)
    audio_report = _turbo_audio(path) if source["has_audio"] else {"present": False}
    color = _color(frames)
    report = {
        "report_id": report_id,
        "source": source,
        "processing": {
            "mode": "TURBO",
            "runtime": runtime_profile(),
            "sampled_frames": len(frames),
            "bounded_work": True,
            "capabilities": {"cut_detection": "candidates_only", "pacing_metrics": "not_calculated", "intent": "not_inferred"},
        },
        "transcript": {"status": "deferred_until_watcher_or_enrichment"},
        "visual": {"frame_samples": color.pop("frame_samples", []), "shots": [], "subjects": {"status": "deferred"}, "motion": {"status": "deferred"}},
        "color": color,
        "audio": audio_report,
        "text_overlay": {"status": "deferred", "track": []},
        "editing": {
            "candidate_regions": candidate_regions,
            "verified_events": [],
            "summary": {
                "internal_verification_passed": False,
                "reliable": False,
                "reason": "TURBO performs sparse triage only; no cuts or pacing values are claimed.",
            },
        },
        "edit_intent": {"status": "not_inferred", "events": [], "reason": "No verified edits or sufficient synchronized semantic evidence in TURBO."},
        "cross_modal_events": [],
        "confidence": {"policy": "Candidates are excluded from training cut/pacing targets until dense verification.", "minimum_training_confidence": 0.8},
        "deferred": ["dense_cut_verification", "ocr", "speaker_diarization", "stem_separation", "semantic_video_model"],
    }
    report["training_features"] = feature_vector(report)
    report["processing"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report


def _sparse_frames(path: Path, duration: float):
    con = av.open(str(path)); stream = next(s for s in con.streams if s.type == "video")
    stream.thread_type = "AUTO"
    targets = np.linspace(0, max(0, duration - .01), TURBO_VIDEO_SAMPLES)
    frames = []
    for target in targets:
        con.seek(int(target * av.time_base), backward=True, any_frame=False)
        selected = None
        for frame in con.decode(stream):
            selected = frame
            if float(frame.time or 0) >= target:
                break
        if selected is None:
            continue
        if selected.height>=selected.width:new_h=160; new_w=max(48,int(selected.width*new_h/selected.height))
        else:new_w=160; new_h=max(48,int(selected.height*new_w/selected.width))
        image = selected.reformat(width=new_w,height=new_h,format="rgb24").to_ndarray()
        rgb=image.astype(np.float32)/255; hsv=_rgb_hsv(rgb)
        lum=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
        hist=np.histogram(lum,bins=24,range=(0,1),density=True)[0]
        h,w=lum.shape; center=lum[int(h*.1):int(h*.9),int(w*.1):int(w*.9)]; edge=np.concatenate([lum[:max(1,int(h*.08))].ravel(),lum[-max(1,int(h*.08)):].ravel(),lum[:,:max(1,int(w*.08))].ravel(),lum[:,-max(1,int(w*.08)):].ravel()])
        metric={"t":round(float(selected.time or target),3),"lum":float(lum.mean()),"std":float(lum.std()),"sat":float(hsv[...,1].mean()),"rgb":rgb.mean(axis=(0,1)).tolist(),"hist":hist,"p05":q(lum,5),"p50":q(lum,50),"p95":q(lum,95),"black_clip":float((lum<.015).mean()),"white_clip":float((lum>.985).mean()),"hue_hist":np.histogram(hsv[...,0][hsv[...,1]>.12],bins=12,range=(0,1))[0],"edge_proxy":float((np.abs(np.diff(lum,axis=0)).mean()+np.abs(np.diff(lum,axis=1)).mean())/2),"center_luminance":float(center.mean()),"edge_luminance":float(edge.mean()),"grain_proxy":0.0,"skin":[]}
        frames.append(metric)
    con.close(); return frames


def _turbo_audio(path: Path):
    con=av.open(str(path)); stream=next((s for s in con.streams if s.type=="audio"),None)
    if stream is None:return {"present":False}
    resampler=av.AudioResampler(format="fltp",layout="mono",rate=TURBO_AUDIO_HZ)
    chunks=[]
    for frame in con.decode(stream):
        converted=resampler.resample(frame)
        for item in converted:
            chunks.append(item.to_ndarray().reshape(-1))
    con.close()
    if not chunks:return {"present":True,"status":"empty"}
    x=np.concatenate(chunks).astype(np.float32); peak=float(np.max(np.abs(x))); rms=float(np.sqrt(np.mean(x*x)+1e-12))
    block=max(1,TURBO_AUDIO_HZ//10); env=np.array([np.sqrt(np.mean(x[i:i+block]**2)+1e-12) for i in range(0,len(x),block)])
    transient_idx=np.where((env[1:-1]>env[:-2])&(env[1:-1]>env[2:])&(env[1:-1]>np.percentile(env,80)))[0]+1 if len(env)>2 else []
    transient_events=[{"timestamp":round(float(i)/10,2),"class":"unknown_transient","confidence":.25,"strength_db_above_median":round(float(20*np.log10((env[i]+1e-12)/(np.median(env)+1e-12))),2),"verification_status":"unclassified"} for i in transient_idx[:80]]
    peak_db=20*np.log10(peak+1e-12)
    return {"present":True,"measurements":{"rms_dbfs":round(20*np.log10(rms+1e-12),2),"decoded_sample_peak_dbfs":round(float(peak_db),2),"decoded_float_peak_can_exceed_zero":bool(peak>1),"true_peak_dbtp":{"status":"not_measured"},"crest_factor_db":round(20*np.log10((peak+1e-12)/(rms+1e-12)),2)},"events":{"transients":transient_events,"beat_grid":[],"beat_status":"not_detected"},"quality":"triage_proxy"}


def feature_vector(report: dict) -> dict:
    c=report.get("color",{}).get("measurements",{}); a=report.get("audio",{}).get("measurements",{}); src=report["source"]
    lum=c.get("luminance",{}); rgb=c.get("rgb_channel_means",{})
    return {"duration":src.get("duration_seconds",0),"fps":src.get("fps",0),"luma_mean":lum.get("mean",0),"luma_p05":lum.get("p05",0),"luma_p95":lum.get("p95",0),"contrast":c.get("contrast_proxy",0),"local_contrast":c.get("local_contrast_proxy",0),"saturation":c.get("saturation_mean",0),"red":rgb.get("red",0),"green":rgb.get("green",0),"blue":rgb.get("blue",0),"dark_pixel_fraction":c.get("dark_pixel_fraction",{}).get("value",0),"bright_pixel_fraction":c.get("bright_pixel_fraction",{}).get("value",0),"rms_dbfs":a.get("rms_dbfs",-120),"decoded_sample_peak_dbfs":a.get("decoded_sample_peak_dbfs",-120),"crest_db":a.get("crest_factor_db",0)}
