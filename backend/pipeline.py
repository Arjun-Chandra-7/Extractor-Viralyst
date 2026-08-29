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
    frame_data = _frames(path, video, duration)
    edits, shots = dense_verify_full_video(path, duration, fps)
    audio_data = _audio(path, audio) if audio else {"present": False}
    color = _color(frame_data)
    frame_samples=color.pop("frame_samples",[])
    color["per_shot"]=_per_shot_color(frame_samples,shots)
    report={
        "report_id": report_id,
        "source": {"filename": original_name, "content_hash":content_hash(path), "duration_seconds": round(duration, 3), "resolution": f"{width}×{height}", "fps": round(fps, 3), "has_audio": bool(audio)},
        "processing": {"status":"running","extractor_version":VERSION,"runtime":runtime_profile(),"mode": "STANDARD", "sampled_frames": len(frame_data), "dense_adjacent_frame_verification":True,"elapsed_seconds": None},
        "transcript": {"status": "awaiting_enrichment"},
        "visual":{"frame_samples":frame_samples,"shots":shots,"subjects":{"status":"deferred"},"motion":{"status":"deferred"}},
        "color": color,
        "audio": audio_data,
        "text_overlay":{"status":"deferred","track":[],"spoken_transcript_kept_separate":True},
        "editing": {"candidate_regions":[],"verified_events":edits,"transforms":[{"timestamp":event["timestamp"],**event["transform_evidence"]} for event in edits if event.get("transform_evidence",{}).get("verification_status")=="supported"],"summary":verified_edit_summary(edits,shots,duration)},
        "semantic":{"status":"deferred","sections":[]},
        "edit_intent":{"status":"awaiting_cross_modal_alignment","events":[]},
        "cross_modal_events":[],
        "confidence":{"minimum_training_confidence":.8,"policy":"Only verified observed facts at or above threshold are eligible for Core Brain training."},
        "deferred":["ocr","caption_tracking","speaker_diarization","stem_separation","semantic_video_model","region_segmentation","optical_effects"],
        "methodology": {"order":["measurement","detection","alignment","interpretation"],"observed_interpretation_separated":True,"notes": ["Color values use frame samples plus center/edge and face-skin subregions; full semantic segmentation remains deferred.", "Audio values are calculated from decoded PCM.", "Editing boundaries are confirmed only by dense adjacent-frame verification."]},
    }
    report["training_features"]=_standard_training_features(report)
    return report


def _standard_training_features(report):
    color=report["color"]["measurements"]; audio=report["audio"].get("observed",{}); summary=report["editing"]["summary"]; eligible_events=[item for item in report["editing"]["verified_events"] if item.get("training_eligible")]
    values={"duration":report["source"]["duration_seconds"],"fps":report["source"]["fps"],"luminance_mean":color["luminance"]["mean"],"saturation_mean":color["saturation_mean"],"integrated_lufs":audio.get("loudness",{}).get("integrated_lufs"),"dynamic_range_db":audio.get("dynamics",{}).get("dynamic_range_db"),"verified_training_cut_count":sum(item["type"]=="hard_cut" for item in eligible_events),"verified_training_cuts_per_minute":round(sum(item["type"]=="hard_cut" for item in eligible_events)*60/max(report["source"]["duration_seconds"],1),2)}
    return {"values":values,"provenance":{"color":{"confidence":.55,"method":"whole-frame samples with partial region metrics","verification_status":"measured"},"audio":{"confidence":.9 if audio.get("loudness",{}).get("method","").startswith("ITU") else .55,"method":audio.get("loudness",{}).get("method"),"verification_status":"measured"},"editing":{"confidence":round(float(np.mean([item["final_confidence"] for item in eligible_events])),3) if eligible_events else 0,"method":"dense adjacent-frame verification","verification_status":"verified_events_above_0.8_only"}},"excluded":{"unverified_edit_events":len(report["editing"]["verified_events"])-len(eligible_events),"intent":"never used as a core training label without human/semantic verification"}}


def _frames(path, stream, duration):
    con = av.open(str(path)); v = next(s for s in con.streams if s.type == "video")
    gap = max(duration / MAX_SAMPLES, 0.12) if duration else 0.12
    last_time = -gap; output = []
    for frame in con.decode(v):
        ts = float(frame.time or 0)
        if ts - last_time < gap: continue
        last_time = ts
        # Preserve aspect ratio; independent array strides distort portrait footage.
        if frame.height>=frame.width:new_h=320; new_w=max(64,int(frame.width*new_h/frame.height))
        else:new_w=320; new_h=max(64,int(frame.height*new_w/frame.width))
        image = frame.reformat(width=new_w,height=new_h,format="rgb24").to_ndarray()
        gray = image.mean(axis=2)
        rgb = image.astype(np.float32) / 255
        hsv = _rgb_hsv(rgb)
        luminance = .2126 * rgb[..., 0] + .7152 * rgb[..., 1] + .0722 * rgb[..., 2]
        h,w=luminance.shape; center=luminance[int(h*.1):int(h*.9),int(w*.1):int(w*.9)]; edge=np.concatenate([luminance[:max(1,int(h*.08))].ravel(),luminance[-max(1,int(h*.08)):].ravel(),luminance[:, :max(1,int(w*.08))].ravel(),luminance[:, -max(1,int(w*.08)):].ravel()])
        skin=_skin_measurements(image,rgb,hsv)
        try:
            import cv2
            blurred=cv2.GaussianBlur(luminance,(0,0),1.1); grain=float(np.std((luminance-blurred)[(luminance>.12)&(luminance<.85)])) if np.any((luminance>.12)&(luminance<.85)) else 0
        except ImportError:grain=0
        metric = {"t": round(ts, 3), "lum": float(luminance.mean()), "std": float(luminance.std()), "sat": float(hsv[...,1].mean()), "rgb": rgb.mean(axis=(0,1)).tolist(), "hist": np.histogram(luminance, bins=32, range=(0,1), density=True)[0], "p05": q(luminance,5), "p50": q(luminance,50), "p95": q(luminance,95), "black_clip":float((luminance < .015).mean()), "white_clip":float((luminance > .985).mean()), "hue_hist":np.histogram(hsv[...,0][hsv[...,1]>.12], bins=12, range=(0,1))[0], "edge_proxy":float((np.abs(np.diff(luminance,axis=0)).mean()+np.abs(np.diff(luminance,axis=1)).mean())/2),"center_luminance":float(center.mean()),"edge_luminance":float(edge.mean()),"grain_proxy":grain,"skin":skin}
        output.append(metric)
    con.close(); return output


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


def _skin_measurements(image,rgb,hsv):
    global _FACE_CASCADE
    try:
        import cv2
    except ImportError:return []
    if _FACE_CASCADE is None:
        model=Path(__file__).parent/"models"/"face_detection_yunet_2026may.onnx"
        if not model.exists():return []
        os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE","4")
        _FACE_CASCADE=cv2.FaceDetectorYN.create(str(model),"",(image.shape[1],image.shape[0]),.6,.3,5000)
    _FACE_CASCADE.setInputSize((image.shape[1],image.shape[0])); _,detected=_FACE_CASCADE.detect(cv2.cvtColor(image,cv2.COLOR_RGB2BGR)); faces=[] if detected is None else detected
    output=[]
    ycrcb=cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)
    for face in faces[:4]:
        x,y,w,h=[int(value) for value in face[:4]]; x=max(0,x); y=max(0,y); w=min(image.shape[1]-x,w); h=min(image.shape[0]-y,h)
        roi=ycrcb[y:y+h,x:x+w]; mask=(roi[...,1]>=133)&(roi[...,1]<=180)&(roi[...,2]>=75)&(roi[...,2]<=135)
        if mask.mean()<.04:continue
        face_lum=.2126*rgb[y:y+h,x:x+w,0]+.7152*rgb[y:y+h,x:x+w,1]+.0722*rgb[y:y+h,x:x+w,2]; face_hsv=hsv[y:y+h,x:x+w]
        output.append({"face_box_normalized":[round(x/image.shape[1],3),round(y/image.shape[0],3),round(w/image.shape[1],3),round(h/image.shape[0],3)],"skin_pixel_fraction":round(float(mask.mean()),3),"skin_luminance":round(float(face_lum[mask].mean()),3),"skin_saturation":round(float(face_hsv[...,1][mask].mean()),3),"skin_hue_degrees":round(float(np.median(face_hsv[...,0][mask])*360),2)})
    return output


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
    skin=[item for frame in frames for item in frame.get("skin",[])]; skin_report={"status":"no_face_skin_mask_detected","samples":0}
    if skin:
        skin_report={"status":"measured_from_face_skin_masks","samples":len(skin),"luminance_mean":round(float(np.mean([item["skin_luminance"] for item in skin])),3),"saturation_mean":round(float(np.mean([item["skin_saturation"] for item in skin])),3),"hue_median_degrees":round(float(np.median([item["skin_hue_degrees"] for item in skin])),2),"consistency":{"luminance_std":round(float(np.std([item["skin_luminance"] for item in skin])),3),"hue_std_degrees":round(float(np.std([item["skin_hue_degrees"] for item in skin])),2)},"method":"OpenCV YuNet face detection + YCrCb skin mask","confidence":.72}
    description = []
    description.append("high contrast" if contrast > .62 else "controlled contrast")
    description.append("desaturated" if sat.mean() < .25 else "saturated" if sat.mean() > .5 else "natural saturation")
    description.append("warm bias" if temp > 4 else "cool bias" if temp < -4 else "neutral white balance")
    return {"interpretation":{"label":", ".join(description),"confidence":.45,"status":"global_frame_sample_interpretation"}, "measurements": {"scope":"whole_frame_samples_with_center_edge_and_skin_subregions","region_aware":"partial", "luminance": {"mean":round(float(l.mean()),3),"p05":round(q(shadows,5),3),"p50":round(q(l,50),3),"p95":round(q(highlights,95),3),"sample_variation":round(float(l.std()),3)},"regions":{"center_luminance_mean":round(float(np.mean([f['center_luminance'] for f in frames])),3),"edge_luminance_mean":round(float(np.mean([f['edge_luminance'] for f in frames])),3),"center_edge_delta":round(float(np.mean([f['center_luminance']-f['edge_luminance'] for f in frames])),3),"subject_background_segmentation":{"status":"deferred"},"graphics_caption_exclusion":{"status":"OCR boxes available separately; exclusion pass deferred"}}, "contrast_proxy": round(float(contrast),3), "local_contrast_proxy":round(float(np.mean([f['edge_proxy'] for f in frames])),4), "saturation_mean":round(float(sat.mean()),3), "white_balance": {"red_blue_bias":round(float(temp),2),"calibration":{"cool_below":-4,"neutral_range":[-4,4],"warm_above":4},"interpretation": "warm" if temp>4 else "cool" if temp<-4 else "neutral"}, "rgb_channel_means": {"red":round(float(rgb[:,0].mean()),3),"green":round(float(rgb[:,1].mean()),3),"blue":round(float(rgb[:,2].mean()),3)}, "dark_pixel_fraction":{"value":round(float(np.mean([f['black_clip'] for f in frames])),4),"threshold":.015,"not_equivalent_to_crushed_blacks":True},"bright_pixel_fraction":{"value":round(float(np.mean([f['white_clip'] for f in frames])),4),"threshold":.985,"not_equivalent_to_highlight_clipping":True}, "dominant_hues":dominant, "harmony":harmony, "skin_tone_behavior":skin_report, "optical_effects":{"status":"measured_proxies_and_deferred_classifiers","grain_high_frequency_proxy":round(float(np.mean([f['grain_proxy'] for f in frames])),5),"vignette_center_edge_luminance_delta":round(float(np.mean([f['center_luminance']-f['edge_luminance'] for f in frames])),4),"sharpness_edge_proxy":round(float(np.mean([f['edge_proxy'] for f in frames])),5),"bloom":None,"halation":None,"motion_blur":None,"chromatic_aberration":None}}, "frame_samples": [{"timestamp":f["t"],"luminance":round(f["lum"],3),"shadow":round(f["p05"],3),"highlight":round(f["p95"],3),"saturation":round(f["sat"],3),"skin_faces":len(f.get("skin",[]))} for f in frames]}


def _per_shot_color(samples,shots):
    output=[]
    for shot in shots:
        selected=[item for item in samples if shot["start"]<=item["timestamp"]<=shot["end"]]
        if not selected:continue
        output.append({"shot_id":shot["shot_id"],"start":shot["start"],"end":shot["end"],"sample_count":len(selected),"observed":{"luminance_mean":round(float(np.mean([item["luminance"] for item in selected])),3),"shadow_mean":round(float(np.mean([item["shadow"] for item in selected])),3),"highlight_mean":round(float(np.mean([item["highlight"] for item in selected])),3),"saturation_mean":round(float(np.mean([item["saturation"] for item in selected])),3),"face_skin_samples":sum(item["skin_faces"] for item in selected)},"confidence":round(min(.9,.45+.08*len(selected)),2)})
    return output


def _audio(path, audio):
    if not audio: return {"present": False, "status": "no_audio_stream"}
    con=av.open(str(path)); st=next(s for s in con.streams if s.type=="audio"); rate=48000
    resampler=av.AudioResampler(format="fltp",layout="stereo",rate=rate); chunks=[]
    for frame in con.decode(st):
        for item in resampler.resample(frame):chunks.append(item.to_ndarray().astype(np.float32))
    con.close()
    if not chunks:return {"present":True,"status":"empty_audio"}
    stereo=np.concatenate(chunks,axis=1); x=stereo.mean(axis=0)
    _AUDIO_CACHE[str(Path(path).resolve())]=x[::3].copy()  # 16 kHz fan-out for ASR/emphasis.
    rms=float(np.sqrt(np.mean(x*x)+1e-12)); peak=float(np.max(abs(stereo))); sample_peak_db=20*math.log10(peak+1e-12)
    half=max(1,rate//2); blocks=np.array([np.sqrt(np.mean(x[i:i+half]**2)+1e-12) for i in range(0,len(x),half)])
    momentary=_window_loudness(x,rate,.4); short_term=_window_loudness(x,rate,3.0)
    integrated_proxy=20*math.log10(rms+1e-12)-.691; integrated=integrated_proxy; loudness_method="rms_proxy_not_bs1770"
    try:
        import pyloudnorm as pyln
        integrated=float(pyln.Meter(rate).integrated_loudness(x)); loudness_method="ITU-R_BS.1770_via_pyloudnorm"
    except (ImportError,ValueError):
        pass
    fft_size=min(len(x),rate*30); windowed=x[:fft_size]*np.hanning(fft_size); power=np.abs(np.fft.rfft(windowed))**2; freqs=np.fft.rfftfreq(fft_size,1/rate); total=float(power.sum()+1e-12)
    def band(lo,hi):return float(power[(freqs>=lo)&(freqs<hi)].sum()/total)
    centroid=float((freqs*power).sum()/total); cumulative=np.cumsum(power); rolloff=float(freqs[min(len(freqs)-1,int(np.searchsorted(cumulative,.85*cumulative[-1])))])
    envelope=np.array([np.sqrt(np.mean(x[i:i+rate//20]**2)+1e-12) for i in range(0,len(x),rate//20)])
    hits=np.where((envelope[1:-1]>envelope[:-2])&(envelope[1:-1]>envelope[2:])&(envelope[1:-1]>np.percentile(envelope,80)))[0]+1 if len(envelope)>2 else []
    median_env=float(np.median(envelope)+1e-12)
    transient_events=[{"timestamp":round(float(i)/20,3),"class":"unknown_transient","confidence":.25,"strength_db_above_median":round(20*math.log10((float(envelope[i])+1e-12)/median_env),2),"verification_status":"unclassified"} for i in hits[:160]]
    bpm,beat_grid,beat_confidence=_beat_grid(envelope)
    mid=(stereo[0]+stereo[1])/2; side=(stereo[0]-stereo[1])/2; stereo_width=float(np.sqrt(np.mean(side*side)+1e-12)/np.sqrt(np.mean(mid*mid)+1e-12))
    dynamic_range=20*math.log10((q(blocks,95)+1e-12)/(q(blocks,10)+1e-12)); crest=20*math.log10((peak+1e-12)/(rms+1e-12))
    short_values=[item["lufs_proxy"] for item in short_term]
    return {"present":True,"observed":{"loudness":{"integrated_lufs":round(integrated,2),"method":loudness_method,"momentary_lufs_curve":momentary,"short_term_lufs_curve":short_term,"loudness_range_proxy_lu":round(q(short_values,95)-q(short_values,10),2) if short_values else None,"decoded_sample_peak_dbfs":round(sample_peak_db,2),"decoded_float_peak_can_exceed_zero":bool(peak>1),"true_peak_dbtp":{"status":"not_measured_without_oversampling"}},"dynamics":{"crest_factor_db":round(crest,2),"dynamic_range_db":round(dynamic_range,2),"clipping_ratio":round(float((abs(stereo)>=1).mean()),6),"limiting_candidate":bool(crest<6 and sample_peak_db>-.5),"compression_strength_proxy":round(clamp(1-dynamic_range/24),3)},"spectrum":{"low_20_250_share":round(band(20,250),4),"mid_250_4k_share":round(band(250,4000),4),"high_4k_20k_share":round(band(4000,20000),4),"spectral_centroid_hz":round(centroid,1),"spectral_rolloff_85_hz":round(rolloff,1)},"mix":{"stereo_side_to_mid_ratio":round(stereo_width,3),"speech_music_sfx_ratio":{"status":"requires_stems"},"ducking":{"status":"requires_stems"}},"speech":{"status":"awaiting_transcript_alignment"},"music":{"bpm_candidate":bpm,"beat_grid_candidate":beat_grid,"beat_confidence":beat_confidence,"verification_status":"unverified_without_music_stem"},"sfx":{"status":"transients_unclassified"}},"events":{"transients":transient_events,"silence_ranges":_silences(blocks),"beat_grid":[],"beat_status":"not_verified"},"interpretation":{"warmth":"warm_candidate" if band(20,250)>.18 else "not_warm","brightness":"bright_candidate" if centroid>3000 else "controlled","confidence":.4}}


def _beat_grid(envelope):
    """Lightweight onset-envelope autocorrelation. Deep lane should use madmom/DBN."""
    if len(envelope) < 40: return None, [], 0.0
    centered = envelope - envelope.mean()
    corr = np.correlate(centered, centered, mode="full")[len(centered)-1:]
    lo, hi = 8, min(len(corr), 41)  # 300–60 BPM at 20 Hz envelope
    if hi <= lo: return None, [], 0.0
    lag = int(np.argmax(corr[lo:hi]) + lo)
    bpm = round(1200 / lag, 1)
    start = int(np.argmax(envelope[:min(len(envelope), lag*2)]))
    confidence=clamp(float(corr[lag]/(corr[0]+1e-12)))
    return bpm, [round((start + i * lag) / 20, 3) for i in range((len(envelope)-start)//lag)], round(confidence,3)


def _window_loudness(x,rate,seconds):
    size=max(1,int(rate*seconds)); step=max(1,size//4); output=[]
    for start in range(0,max(1,len(x)-size+1),step):
        value=20*math.log10(float(np.sqrt(np.mean(x[start:start+size]**2)+1e-12))+1e-12)-.691
        output.append({"time":round((start+size/2)/rate,3),"lufs_proxy":round(value,2)})
    return output


def _silences(blocks):
    quiet=blocks < max(np.percentile(blocks,15), .003); out=[]; start=None
    for i,v in enumerate(quiet):
        if v and start is None:start=i
        if not v and start is not None:
            if i-start>=2:out.append({"start":round(start*.5,2),"end":round(i*.5,2)})
            start=None
    return out


def enrich_transcript(path, model_name=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc: raise RuntimeError("Install faster-whisper to enable transcript enrichment.") from exc
    model_name=model_name or os.getenv("VIRALYST_WHISPER_MODEL", "base.en")
    if model_name not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_name]=WhisperModel(model_name, device="auto", compute_type="int8", cpu_threads=max(1, min(8, os.cpu_count() or 4)))
    model=_WHISPER_MODELS[model_name]
    cache_key=str(Path(path).resolve()); cached_audio=_AUDIO_CACHE.get(cache_key)
    transcript_source=cached_audio if cached_audio is not None else str(path)
    raw_segments, info=model.transcribe(transcript_source, word_timestamps=True, vad_filter=True, beam_size=1, best_of=1, condition_on_previous_text=False)
    words=[]; segments=[]
    for index, seg in enumerate(raw_segments):
        segment_words=[]
        for raw in seg.words or []:
            token=raw.word.strip(); punctuation="".join(re.findall(r"[^\w\s']",token))
            clean=token.rstrip(".,!?;:\"”’") or token
            raw_start,raw_end=round(raw.start,3),round(raw.end,3)
            item={"word":clean,"display":token,"raw_start":raw_start,"raw_end":raw_end,"aligned_start":raw_start,"aligned_end":max(raw_end,round(raw_start+.02,3)),"start":raw_start,"end":max(raw_end,round(raw_start+.02,3)),"confidence":round(raw.probability,3),"punctuation":punctuation,"segment":index,"timing_status":"aligned" if raw_end>raw_start else "repaired_minimum_duration"}
            words.append(item); segment_words.append(item)
        duration=max(float(seg.end-seg.start),.001)
        segments.append({"index":index,"start":round(seg.start,3),"end":round(seg.end,3),"text":seg.text.strip(),"words_per_minute":round(len(segment_words)*60/duration,1) if duration>=2 else None,"delivery_rate_status":"measured" if duration>=2 else "insufficient_window","avg_log_probability":round(float(seg.avg_logprob),3),"no_speech_probability":round(float(seg.no_speech_prob),3)})
    pauses=[]
    for before, after in zip(words,words[1:]):
        gap=round(after["start"]-before["end"],3)
        if gap>=.2:
            pauses.append({"start":before["end"],"end":after["start"],"duration":gap,"after_word":before["display"],"type":"long" if gap>=1 else "short" if gap>=.45 else "micro"})
            after["pause_before_seconds"]=gap
    emphasized=_emphasis(path,words,cached_audio,16000)
    duration=max((words[-1]["end"]-words[0]["start"]) if words else 0,.001)
    punctuation_events=[{"time":w["end"],"mark":w["punctuation"],"word":w["word"]} for w in words if w["punctuation"]]
    sentences=[]; sentence_words=[]
    for word in words:
        sentence_words.append(word)
        if any(mark in word["punctuation"] for mark in ".!?"):
            sentences.append({"sentence_id":len(sentences),"start":sentence_words[0]["start"],"end":word["end"],"text":" ".join(item["display"] for item in sentence_words),"confidence":round(float(np.mean([item["confidence"] for item in sentence_words])),3)})
            sentence_words=[]
    if sentence_words:sentences.append({"sentence_id":len(sentences),"start":sentence_words[0]["start"],"end":sentence_words[-1]["end"],"text":" ".join(item["display"] for item in sentence_words),"confidence":round(float(np.mean([item["confidence"] for item in sentence_words])),3)})
    _AUDIO_CACHE.pop(cache_key,None)
    rolling=[]
    for start in np.arange(words[0]["start"] if words else 0,(words[-1]["end"] if words else 0),2.0):
        window=[w for w in words if start<=w["start"]<start+4]
        if len(window)>=3: rolling.append({"start":round(float(start),3),"end":round(float(start+4),3),"words_per_minute":round(len(window)*15,1),"status":"measured"})
    return {"status":"complete","engine":f"faster-whisper/{model_name}","language":info.language,"language_probability":round(info.language_probability,3),"full_text":" ".join(w["display"] for w in words),"words":words,"sentences":sentences,"segments":segments,"delivery":{"overall_words_per_minute":round(len(words)*60/duration,1),"word_count":len(words),"speaking_span_seconds":round(duration,3),"rolling_windows":rolling},"prosody":{"status":"energy_relative_to_neighboring_words","emphasis_candidates":emphasized,"limitations":["Pitch/F0 and syllable stress require a dedicated prosody model."]},"pauses":pauses,"emphasized_words":emphasized,"punctuation_events":punctuation_events,"speaker_changes":{"status":"requires_diarization","track":[]}}


def _emphasis(path, words, cached_audio=None, cached_rate=16000):
    """Align word spans with source energy; punctuation also supplies emphasis evidence."""
    if not words:return []
    if cached_audio is not None:
        audio=cached_audio; rate=cached_rate
    else:
        con=av.open(str(path)); stream=next((s for s in con.streams if s.type=="audio"),None)
        if stream is None:return []
        rate=8000; resampler=av.AudioResampler(format="fltp",layout="mono",rate=rate); chunks=[]
        for frame in con.decode(stream):
            for item in resampler.resample(frame):chunks.append(item.to_ndarray().reshape(-1))
        con.close()
        if not chunks:return []
        audio=np.concatenate(chunks)
    levels=[]
    for word in words:
        start=max(0,int(word["start"]*rate)); end=min(len(audio),max(start+1,int(word["end"]*rate)))
        levels.append(20*math.log10(float(np.sqrt(np.mean(audio[start:end]**2)+1e-12))+1e-12))
    median=float(np.median(levels)); output=[]
    for word,level in zip(words,levels):
        left=max(0,len(levels)-1); local=np.median(levels[max(0,words.index(word)-2):min(len(levels),words.index(word)+3)])
        delta=level-local
        word["energy_dbfs"]=round(level,2); word["prosodic_emphasis_candidate"]=bool(delta>=3)
        if word["prosodic_emphasis_candidate"]:output.append({"word":word["display"],"start":word["start"],"end":word["end"],"energy_above_local_words_db":round(float(delta),2),"confidence":round(min(.75,.45+max(0,delta)/12),3),"evidence":["local_word_energy"],"verification_status":"candidate"})
    return output
