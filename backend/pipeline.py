from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

import av
import numpy as np

MAX_SAMPLES = 96
_WHISPER_MODELS = {}


def q(values, percentile):
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def clamp(value):
    return float(max(0, min(1, value)))


def analyse(path: Path, report_id: str, original_name: str) -> dict:
    container = av.open(str(path))
    video = next((s for s in container.streams if s.type == "video"), None)
    audio = next((s for s in container.streams if s.type == "audio"), None)
    if video is None:
        raise ValueError("No video stream found")
    duration = float(container.duration / av.time_base) if container.duration else 0.0
    fps = float(video.average_rate) if video.average_rate else 0.0
    width, height = video.codec_context.width, video.codec_context.height
    frame_data, edits = _frames(path, video, duration)
    audio_data = _audio(path, audio) if audio else {"present": False}
    color = _color(frame_data)
    return {
        "report_id": report_id,
        "source": {"filename": original_name, "duration_seconds": round(duration, 3), "resolution": f"{width}×{height}", "fps": round(fps, 3), "has_audio": bool(audio)},
        "processing": {"mode": "fast-pass", "sampled_frames": len(frame_data), "sampling_cap": MAX_SAMPLES, "elapsed_seconds": None},
        "transcript": {"status": "not_enriched", "spoken": [], "speakers": [], "pauses": [], "note": "Run transcript enrichment for word timestamps, confidence, punctuation and delivery metrics."},
        "on_screen_text": {"status": "not_configured", "track": [], "note": "Kept separate from spoken transcript by design. Connect a local OCR engine to populate this track."},
        "color": color,
        "audio": audio_data,
        "editing": {"events": edits, "summary": _edit_summary(edits, duration)},
        "intent": _intent(edits, audio_data, duration),
        "methodology": {"measurement_first": True, "notes": ["Color values are calculated from sampled decoded frames.", "Audio values are calculated from decoded PCM, not an opinion score.", "Edit events are heuristic detections and include evidence/confidence."]},
    }


def _frames(path, stream, duration):
    con = av.open(str(path)); v = next(s for s in con.streams if s.type == "video")
    gap = max(duration / MAX_SAMPLES, 0.12) if duration else 0.12
    last_time = -gap; output = []; previous = None; events = []
    for frame in con.decode(v):
        ts = float(frame.time or 0)
        if ts - last_time < gap: continue
        last_time = ts
        image = frame.to_ndarray(format="rgb24")
        # Downsample: strong enough for grading descriptors, fast enough for high-res uploads.
        image = image[::max(1, image.shape[0]//180), ::max(1, image.shape[1]//320)]
        gray = image.mean(axis=2)
        rgb = image.astype(np.float32) / 255
        hsv = _rgb_hsv(rgb)
        luminance = .2126 * rgb[..., 0] + .7152 * rgb[..., 1] + .0722 * rgb[..., 2]
        metric = {"t": round(ts, 3), "lum": float(luminance.mean()), "std": float(luminance.std()), "sat": float(hsv[...,1].mean()), "rgb": rgb.mean(axis=(0,1)).tolist(), "hist": np.histogram(luminance, bins=32, range=(0,1), density=True)[0], "p05": q(luminance,5), "p50": q(luminance,50), "p95": q(luminance,95), "black_clip":float((luminance < .015).mean()), "white_clip":float((luminance > .985).mean()), "hue_hist":np.histogram(hsv[...,0][hsv[...,1]>.12], bins=12, range=(0,1))[0], "edge_proxy":float((np.abs(np.diff(luminance,axis=0)).mean()+np.abs(np.diff(luminance,axis=1)).mean())/2)}
        if previous:
            hist_delta = float(np.abs(metric["hist"] - previous["hist"]).sum())
            lum_delta = abs(metric["lum"] - previous["lum"])
            if hist_delta > 0.13:
                kind = "hard_cut" if hist_delta > .23 else "possible_transition"
                events.append({"start": round(ts,3), "end": round(ts,3), "type": kind, "confidence": round(clamp((hist_delta-.08)/.25),2), "evidence": {"histogram_delta": round(hist_delta,3), "luminance_delta": round(lum_delta,3)}})
        output.append(metric); previous = metric
    con.close(); return output, events


def _rgb_hsv(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1); d = mx-mn
    s = np.divide(d, mx, out=np.zeros_like(d), where=mx!=0)
    h = np.zeros_like(mx)
    mask = d != 0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    h = np.where(mask & (mx == r), ((g-b)/np.where(mask,d,1)) % 6, h)
    h = np.where(mask & (mx == g), (b-r)/np.where(mask,d,1) + 2, h)
    h = np.where(mask & (mx == b), (r-g)/np.where(mask,d,1) + 4, h)
    return np.stack([h / 6, s, mx], -1)


def _color(frames):
    if not frames: return {"status": "insufficient_frames"}
    l = np.array([f["lum"] for f in frames]); sat = np.array([f["sat"] for f in frames]); rgb=np.array([f["rgb"] for f in frames])
    temp = (rgb[:,0]-rgb[:,2]).mean() * 100
    hues = np.sum([f["hue_hist"] for f in frames],axis=0)
    hue_names = ["red","orange","yellow","chartreuse","green","spring green","cyan","azure","blue","violet","magenta","rose"]
    dominant = [{"hue":hue_names[int(i)],"share":round(float(hues[i]/max(hues.sum(),1)),3)} for i in np.argsort(hues)[-3:][::-1]]
    shadows = np.array([f["p05"] for f in frames]); highlights=np.array([f["p95"] for f in frames])
    contrast = np.mean(highlights-shadows)
    harmony = "complementary tendency" if len(dominant)>1 and abs(hue_names.index(dominant[0]["hue"])-hue_names.index(dominant[1]["hue"])) in range(5,8) else "analogous / single-palette tendency"
    description = []
    description.append("high contrast" if contrast > .62 else "controlled contrast")
    description.append("desaturated" if sat.mean() < .25 else "saturated" if sat.mean() > .5 else "natural saturation")
    description.append("warm bias" if temp > 4 else "cool bias" if temp < -4 else "neutral white balance")
    return {"semantic": ", ".join(description), "measurements": {"luminance": {"mean":round(float(l.mean()),3),"p05":round(q(shadows,5),3),"p50":round(q(l,50),3),"p95":round(q(highlights,95),3),"shot_variation":round(float(l.std()),3)}, "contrast_proxy": round(float(contrast),3), "local_contrast_proxy":round(float(np.mean([f['edge_proxy'] for f in frames])),4), "saturation_mean":round(float(sat.mean()),3), "white_balance": {"red_blue_bias":round(float(temp),2),"temperature": "warm" if temp>4 else "cool" if temp<-4 else "neutral"}, "rgb_channel_means": {"red":round(float(rgb[:,0].mean()),3),"green":round(float(rgb[:,1].mean()),3),"blue":round(float(rgb[:,2].mean()),3)}, "clipping": {"black":round(float(np.mean([f['black_clip'] for f in frames])),4),"white":round(float(np.mean([f['white_clip'] for f in frames])),4)}, "dominant_hues":dominant, "harmony":harmony, "skin_tone_behavior":{"status":"requires_face_detection","value":None}, "bloom_halation":{"status":"requires_high_resolution_optics_pass","value":None}, "chromatic_aberration":{"status":"requires_high_resolution_optics_pass","value":None}}, "shot_track": [{"t":f["t"],"luminance":round(f["lum"],3),"shadow":round(f["p05"],3),"highlight":round(f["p95"],3),"saturation":round(f["sat"],3)} for f in frames]}


def _audio(path, audio):
    if not audio: return {"present": False, "status": "no_audio_stream"}
    con=av.open(str(path)); st=next(s for s in con.streams if s.type=="audio"); chunks=[]; rate=st.rate or 44100
    for fr in con.decode(st):
        raw = fr.to_ndarray()
        a = raw.astype(np.float32)
        if raw.dtype.kind in "iu": a /= np.iinfo(raw.dtype).max
        chunks.append(a.mean(axis=0) if a.ndim>1 else a)
    con.close()
    if not chunks:return {"present":True,"status":"empty_audio"}
    x=np.concatenate(chunks); x=x[::max(1,rate//22050)]; rate=min(rate,22050)
    rms=np.sqrt(np.mean(x*x)+1e-12); peak=float(np.max(abs(x))); lufs=20*math.log10(rms+1e-12)-0.691
    blocks=np.array([np.sqrt(np.mean(b*b)+1e-12) for b in np.array_split(x, max(1,len(x)//(rate//2)))])
    fft=np.abs(np.fft.rfft(x[:min(len(x),rate*20)]*np.hanning(min(len(x),rate*20))))**2; freqs=np.fft.rfftfreq(min(len(x),rate*20),1/rate)
    def band(lo,hi):return float(fft[(freqs>=lo)&(freqs<hi)].sum()/(fft.sum()+1e-12))
    envelope=np.array([np.sqrt(np.mean(b*b)+1e-12) for b in np.array_split(x,max(1,len(x)//(rate//20)))])
    hits=np.where((envelope[1:-1]>envelope[:-2])&(envelope[1:-1]>envelope[2:])&(envelope[1:-1]>np.percentile(envelope,75)))[0]+1
    bpm, beat_grid = _beat_grid(envelope)
    return {"present":True,"measurements":{"integrated_loudness_lufs_proxy":round(lufs,2),"true_peak_dbfs":round(20*math.log10(peak+1e-12),2),"dynamic_range_db":round(20*math.log10((q(blocks,95)+1e-12)/(q(blocks,10)+1e-12)),2),"clipping_ratio":round(float((abs(x)>.99).mean()),5),"noise_floor_proxy_dbfs":round(20*math.log10(q(blocks,10)+1e-12),2),"spectral_balance":{"bass_20_250":round(band(20,250),3),"mids_250_4k":round(band(250,4000),3),"treble_4k_16k":round(band(4000,16000),3)},"speech_music_sfx_balance":{"status":"requires_stem_separation","value":None},"stereo_width":{"status":"requires_stereo_preservation_pass","value":None},"reverb_distortion":{"status":"requires_deep_audio_pass","value":None}},"events":{"transients":[round(float(i)/20,3) for i in hits[:160]],"silence_ranges":_silences(blocks),"bpm_proxy":bpm,"beat_grid":beat_grid},"semantic":{"impact":"high" if q(blocks,95)/max(q(blocks,50),1e-6)>3 else "controlled","warmth":"bass-forward" if band(20,250)>band(4000,16000) else "bright"}}


def _beat_grid(envelope):
    """Lightweight onset-envelope autocorrelation. Deep lane should use madmom/DBN."""
    if len(envelope) < 40: return None, []
    centered = envelope - envelope.mean()
    corr = np.correlate(centered, centered, mode="full")[len(centered)-1:]
    lo, hi = 8, min(len(corr), 41)  # 300–60 BPM at 20 Hz envelope
    if hi <= lo: return None, []
    lag = int(np.argmax(corr[lo:hi]) + lo)
    bpm = round(1200 / lag, 1)
    start = int(np.argmax(envelope[:min(len(envelope), lag*2)]))
    return bpm, [round((start + i * lag) / 20, 3) for i in range((len(envelope)-start)//lag)]


def _silences(blocks):
    quiet=blocks < max(np.percentile(blocks,15), .003); out=[]; start=None
    for i,v in enumerate(quiet):
        if v and start is None:start=i
        if not v and start is not None:
            if i-start>=2:out.append({"start":round(start*.5,2),"end":round(i*.5,2)})
            start=None
    return out


def _edit_summary(events,duration):
    cuts=sum(e['type'] in {'hard_cut','cut_candidate'} for e in events)
    return {"cut_count":cuts,"pace_cuts_per_minute":round(cuts*60/max(duration,1),1),"timeline_coverage":"sampled"}


def _intent(edits,audio,duration):
    events = audio.get('events') or {}
    transients=events.get('transients',[]); beats=events.get('beat_grid',[]); out=[]
    for event in edits:
        nearby=[t for t in transients if abs(t-event['start'])<.2]
        on_beat=[t for t in beats if abs(t-event['start'])<.12]
        beat_supported = bool(nearby or on_beat)
        out.append({"time":event['start'],"event":event['type'],"likely_intent":"beat-driven pacing / retention interrupt" if beat_supported else "pacing reset or subject change","confidence":round(.78 if beat_supported else .52,2),"evidence":{"nearby_transient":nearby[:1],"nearby_beat":on_beat[:1],"edit_confidence":event['confidence']}})
    return out


def enrich_transcript(path, model_name=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc: raise RuntimeError("Install faster-whisper to enable transcript enrichment.") from exc
    model_name=model_name or os.getenv("VIRALYST_WHISPER_MODEL", "base.en")
    if model_name not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_name]=WhisperModel(model_name, device="auto", compute_type="int8", cpu_threads=max(1, min(8, os.cpu_count() or 4)))
    model=_WHISPER_MODELS[model_name]
    raw_segments, info=model.transcribe(str(path), word_timestamps=True, vad_filter=True, beam_size=1, best_of=1, condition_on_previous_text=False)
    words=[]; segments=[]
    for index, seg in enumerate(raw_segments):
        segment_words=[]
        for raw in seg.words or []:
            token=raw.word.strip(); punctuation="".join(re.findall(r"[^\w\s']",token))
            clean=token.rstrip(".,!?;:\"”’") or token
            item={"word":clean,"display":token,"start":round(raw.start,3),"end":round(raw.end,3),"confidence":round(raw.probability,3),"punctuation":punctuation,"segment":index}
            words.append(item); segment_words.append(item)
        duration=max(float(seg.end-seg.start),.001)
        segments.append({"index":index,"start":round(seg.start,3),"end":round(seg.end,3),"text":seg.text.strip(),"words_per_minute":round(len(segment_words)*60/duration,1),"avg_log_probability":round(float(seg.avg_logprob),3),"no_speech_probability":round(float(seg.no_speech_prob),3)})
    pauses=[]
    for before, after in zip(words,words[1:]):
        gap=round(after["start"]-before["end"],3)
        if gap>=.2:
            pauses.append({"start":before["end"],"end":after["start"],"duration":gap,"after_word":before["display"],"type":"long" if gap>=1 else "short" if gap>=.45 else "micro"})
            after["pause_before_seconds"]=gap
    emphasized=_emphasis(path,words)
    duration=max((words[-1]["end"]-words[0]["start"]) if words else 0,.001)
    punctuation_events=[{"time":w["end"],"mark":w["punctuation"],"word":w["word"]} for w in words if w["punctuation"]]
    return {"status":"complete","engine":f"faster-whisper/{model_name}","language":info.language,"language_probability":round(info.language_probability,3),"full_text":" ".join(w["display"] for w in words),"spoken":words,"segments":segments,"delivery":{"overall_words_per_minute":round(len(words)*60/duration,1),"word_count":len(words),"speaking_span_seconds":round(duration,3)},"pauses":pauses,"emphasized_words":emphasized,"punctuation_events":punctuation_events,"speaker_changes":{"status":"requires_diarization","track":[]}}


def _emphasis(path, words):
    """Align word spans with source energy; punctuation also supplies emphasis evidence."""
    if not words:return []
    con=av.open(str(path)); stream=next((s for s in con.streams if s.type=="audio"),None)
    if stream is None:return []
    rate=8000; resampler=av.AudioResampler(format="fltp",layout="mono",rate=rate); chunks=[]
    for frame in con.decode(stream):
        for item in resampler.resample(frame):chunks.append(item.to_ndarray().reshape(-1))
    con.close()
    if not chunks:return []
    audio=np.concatenate(chunks); levels=[]
    for word in words:
        start=max(0,int(word["start"]*rate)); end=min(len(audio),max(start+1,int(word["end"]*rate)))
        levels.append(20*math.log10(float(np.sqrt(np.mean(audio[start:end]**2)+1e-12))+1e-12))
    median=float(np.median(levels)); output=[]
    for word,level in zip(words,levels):
        delta=level-median; punctuation_emphasis=any(mark in word["punctuation"] for mark in "!?")
        word["energy_dbfs"]=round(level,2); word["emphasized"]=bool(delta>=3 or punctuation_emphasis)
        if word["emphasized"]:output.append({"word":word["display"],"start":word["start"],"end":word["end"],"energy_above_median_db":round(delta,2),"evidence":"punctuation_and_energy" if punctuation_emphasis else "energy"})
    return output
