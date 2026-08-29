from __future__ import annotations

import re
from pathlib import Path

import av
import numpy as np

_OCR_ENGINE=None


def extract_text_overlay(path: Path,duration: float,max_frames: int=8) -> dict:
    """Separate scene-text track using bounded keyframe OCR and temporal grouping."""
    try:
        from rapidocr import RapidOCR
    except ImportError:
        return {"status":"unavailable","engine":"rapidocr","track":[],"reason":"Install rapidocr and onnxruntime"}
    global _OCR_ENGINE
    if _OCR_ENGINE is None:_OCR_ENGINE=RapidOCR()
    targets=np.linspace(0,max(0,duration-.01),max(2,min(max_frames,int(duration*2)+1)))
    con=av.open(str(path)); stream=next(s for s in con.streams if s.type=="video"); stream.thread_type="AUTO"; samples=[]; target_index=0
    for frame in con.decode(stream):
        timestamp=float(frame.time or 0)
        if target_index>=len(targets):break
        if timestamp+1e-6<targets[target_index]:continue
        width,height=frame.width,frame.height
        if height>=width:new_h=540; new_w=max(64,int(width*new_h/height))
        else:new_w=540; new_h=max(64,int(height*new_w/width))
        image=frame.reformat(width=new_w,height=new_h,format="bgr24").to_ndarray()
        result=_OCR_ENGINE(image)
        detections=[]; boxes=result.boxes if result.boxes is not None else []; texts=result.txts if result.txts is not None else []; scores=result.scores if result.scores is not None else []
        for box,text,score in zip(boxes,texts,scores):
            points=np.asarray(box,dtype=float); x1,y1=points.min(axis=0); x2,y2=points.max(axis=0); center_y=(y1+y2)/2/new_h
            role="caption" if center_y>.52 else "title" if center_y<.28 else "label_or_graphic_text"
            detections.append({"text":str(text),"confidence":round(float(score),3),"polygon_normalized":[[round(float(x/new_w),4),round(float(y/new_h),4)] for x,y in points],"position":{"center_x":round(float((x1+x2)/2/new_w),3),"center_y":round(float(center_y),3),"width":round(float((x2-x1)/new_w),3),"height":round(float((y2-y1)/new_h),3)},"role_candidate":role,"typography":{"case":"uppercase" if str(text).isupper() else "lowercase" if str(text).islower() else "mixed","relative_text_height":round(float((y2-y1)/new_h),3),"font_family":{"status":"not_identified"},"stroke_shadow_color":{"status":"not_measured"}}})
        samples.append({"timestamp":round(timestamp,3),"detections":detections}); target_index+=1
    con.close()
    track=_group_track(samples,duration)
    return {"status":"complete_bounded_sampling","engine":"RapidOCR/ONNX","spoken_transcript_kept_separate":True,"sample_count":len(samples),"track":track,"caption_analysis":{"event_count":sum(item["role_candidate"]=="caption" for item in track),"animation":{"status":"requires_dense_caption_motion_pass"},"word_highlighting":{"status":"requires_dense_caption_motion_pass"}},"limitations":["Start/end times are bounded by OCR sampling cadence.","Typography family, stroke, shadow and animation require a denser visual pass."]}


def _group_track(samples,duration):
    active={}; output=[]
    for sample_index,sample in enumerate(samples):
        seen=set()
        for detection in sample["detections"]:
            key=_normalize(detection["text"])+f"@{round(detection['position']['center_y'],1)}"
            seen.add(key)
            if key in active:
                active[key]["end"]=sample["timestamp"]; active[key]["observations"]+=1; active[key]["confidence"]=round(max(active[key]["confidence"],detection["confidence"]),3)
            else:
                event={"text":detection["text"],"start":sample["timestamp"],"end":sample["timestamp"],"confidence":detection["confidence"],"detection_method":"bounded_keyframe_ocr","verification_status":"observed","observations":1,**{key:value for key,value in detection.items() if key not in {"text","confidence"}}}
                active[key]=event; output.append(event)
        for key in list(active):
            if key not in seen and active[key]["end"]<sample["timestamp"]:active.pop(key)
    cadence=(samples[1]["timestamp"]-samples[0]["timestamp"]) if len(samples)>1 else 0
    for event in output:event["end"]=round(min(duration,event["end"]+cadence),3); event["duration"]=round(event["end"]-event["start"],3); event["words_visible"]=len(event["text"].split())
    return output


def _normalize(text):
    return re.sub(r"\W+","",text).lower()
