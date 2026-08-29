from __future__ import annotations

import time
from pathlib import Path

import av
import numpy as np

from .pipeline import _color, _edit_summary, _intent, _rgb_hsv, q

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
    frames, edits = _sparse_frames(path, duration)
    audio_report = _turbo_audio(path) if source["has_audio"] else {"present": False}
    color = _color(frames)
    report = {
        "report_id": report_id,
        "source": source,
        "processing": {"mode": "turbo-corpus", "sampled_frames": len(frames), "bounded_work": True},
        "color": color,
        "audio": audio_report,
        "editing": {"events": edits, "summary": _edit_summary(edits, duration)},
        "intent": _intent(edits, audio_report, duration),
        "deferred": ["word_transcript", "ocr", "speaker_diarization", "stem_separation", "semantic_video_model"],
    }
    report["training_features"] = feature_vector(report)
    report["processing"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report


def _sparse_frames(path: Path, duration: float):
    con = av.open(str(path)); stream = next(s for s in con.streams if s.type == "video")
    stream.thread_type = "AUTO"
    targets = np.linspace(0, max(0, duration - .01), TURBO_VIDEO_SAMPLES)
    frames = []; edits = []; previous = None
    for target in targets:
        con.seek(int(target * av.time_base), backward=True, any_frame=False)
        selected = None
        for frame in con.decode(stream):
            selected = frame
            if float(frame.time or 0) >= target:
                break
        if selected is None:
            continue
        image = selected.to_ndarray(format="rgb24")
        image = image[::max(1,image.shape[0]//90), ::max(1,image.shape[1]//160)]
        rgb=image.astype(np.float32)/255; hsv=_rgb_hsv(rgb)
        lum=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
        hist=np.histogram(lum,bins=24,range=(0,1),density=True)[0]
        metric={"t":round(float(selected.time or target),3),"lum":float(lum.mean()),"std":float(lum.std()),"sat":float(hsv[...,1].mean()),"rgb":rgb.mean(axis=(0,1)).tolist(),"hist":hist,"p05":q(lum,5),"p50":q(lum,50),"p95":q(lum,95),"black_clip":float((lum<.015).mean()),"white_clip":float((lum>.985).mean()),"hue_hist":np.histogram(hsv[...,0][hsv[...,1]>.12],bins=12,range=(0,1))[0],"edge_proxy":float((np.abs(np.diff(lum,axis=0)).mean()+np.abs(np.diff(lum,axis=1)).mean())/2)}
        if previous:
            delta=float(np.abs(hist-previous["hist"]).sum())
            if delta>.16:
                edits.append({"start":metric["t"],"end":metric["t"],"type":"cut_candidate","confidence":round(min(1,delta/.38),2),"evidence":{"sparse_histogram_delta":round(delta,3)},"verification":"deep_pass_required"})
        frames.append(metric); previous=metric
    con.close(); return frames, edits


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
    return {"present":True,"measurements":{"rms_dbfs":round(20*np.log10(rms+1e-12),2),"sample_peak_dbfs":round(20*np.log10(peak+1e-12),2),"crest_factor_db":round(20*np.log10((peak+1e-12)/(rms+1e-12)),2)},"events":{"transients":[round(float(i)/10,2) for i in transient_idx[:80]],"beat_grid":[]},"quality":"triage_proxy"}


def feature_vector(report: dict) -> dict:
    c=report.get("color",{}).get("measurements",{}); a=report.get("audio",{}).get("measurements",{}); src=report["source"]; edit=report["editing"]["summary"]
    lum=c.get("luminance",{}); rgb=c.get("rgb_channel_means",{}); clip=c.get("clipping",{})
    return {"duration":src.get("duration_seconds",0),"fps":src.get("fps",0),"luma_mean":lum.get("mean",0),"luma_p05":lum.get("p05",0),"luma_p95":lum.get("p95",0),"contrast":c.get("contrast_proxy",0),"local_contrast":c.get("local_contrast_proxy",0),"saturation":c.get("saturation_mean",0),"red":rgb.get("red",0),"green":rgb.get("green",0),"blue":rgb.get("blue",0),"black_clip":clip.get("black",0),"white_clip":clip.get("white",0),"rms_dbfs":a.get("rms_dbfs",-120),"peak_dbfs":a.get("sample_peak_dbfs",-120),"crest_db":a.get("crest_factor_db",0),"cut_candidates":edit.get("cut_count",0),"cuts_per_minute":edit.get("pace_cuts_per_minute",0)}
