from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class DenseFrame:
    time: float
    rgb: np.ndarray
    luminance: float
    edge: float


def sparse_candidate_regions(frame_samples: list[dict], duration: float) -> list[dict]:
    """Locate broad regions worth scanning; never call these edits or cuts."""
    regions=[]
    for before,after in zip(frame_samples,frame_samples[1:]):
        hist_a=np.asarray(before["hist"],dtype=np.float32); hist_b=np.asarray(after["hist"],dtype=np.float32)
        hist_delta=float(np.abs(hist_a/hist_a.sum()-hist_b/hist_b.sum()).sum()/2) if hist_a.sum() and hist_b.sum() else 0
        luminance_delta=abs(float(after["lum"])-float(before["lum"]))
        candidate_confidence=min(.74,.2+.55*hist_delta+.35*luminance_delta)
        if hist_delta>=.10 or luminance_delta>=.10:
            regions.append({"start":round(float(before["t"]),3),"end":round(float(after["t"]),3),"candidate_confidence":round(candidate_confidence,3),"detection_method":"sparse_frame_change","evidence":{"histogram_distance":round(hist_delta,4),"luminance_delta":round(luminance_delta,4)},"verification_status":"unverified"})
    return _merge_regions(regions,duration)


def verify_candidate_regions(path: Path, regions: list[dict], duration: float, fps: float) -> tuple[list[dict],list[dict]]:
    """Dense adjacent-frame scan inside sparse candidate regions."""
    if not regions:return [],build_shots([],duration)
    raw=[]
    for region in regions:
        frames=_decode_region(path,max(0,region["start"]-.25),min(duration,region["end"]+.25))
        raw.extend(_boundaries_in_region(frames,region,fps))
    verified=_deduplicate(raw,max(.08,1/max(fps,1)*2))
    return verified,build_shots(verified,duration)


def dense_verify_full_video(path: Path, duration: float, fps: float) -> tuple[list[dict],list[dict]]:
    """STANDARD/FORENSIC path: verify every adjacent decoded frame."""
    region={"start":0.0,"end":duration,"candidate_confidence":.72,"detection_method":"full_dense_scan","evidence":{},"verification_status":"pending"}
    events=_boundaries_in_region(_decode_region(path,0,duration),region,fps)
    verified=_deduplicate(events,max(.08,1/max(fps,1)*2))
    return verified,build_shots(verified,duration)


def build_shots(events: list[dict], duration: float) -> list[dict]:
    verified=[e for e in events if e.get("verification_status")=="verified"]
    boundaries=[float(e["timestamp"]) for e in verified]
    points=[0.0]+sorted(t for t in boundaries if 0<t<duration)+[duration]
    return [{"shot_id":index,"start":round(start,3),"end":round(end,3),"duration":round(end-start,3),"representative_frame":round((start+end)/2,3),"boundary_in":None if index==0 else verified[index-1].get("type")} for index,(start,end) in enumerate(zip(points,points[1:])) if end-start>.001]


def verified_edit_summary(events: list[dict],shots: list[dict],duration: float) -> dict:
    verified=[e for e in events if e.get("verification_status")=="verified"]
    cut_types={"hard_cut","jump_cut"}; cuts=[e for e in verified if e.get("type") in cut_types]
    durations=[shot["duration"] for shot in shots]
    return {"verified_boundary_count":len(verified),"cut_count":len(cuts),"cuts_per_minute":round(len(cuts)*60/max(duration,1),2),"average_shot_length":round(float(np.mean(durations)),3) if durations else None,"median_shot_length":round(float(np.median(durations)),3) if durations else None,"pacing_source":"verified_boundaries_only","reliable":True}


def _decode_region(path: Path,start: float,end: float) -> list[DenseFrame]:
    con=av.open(str(path)); stream=next(s for s in con.streams if s.type=="video"); stream.thread_type="AUTO"
    con.seek(int(max(0,start-.12)*av.time_base),backward=True,any_frame=False)
    output=[]
    for frame in con.decode(stream):
        timestamp=float(frame.time or 0)
        if timestamp<start-.10:continue
        if timestamp>end+.10:break
        if frame.height>=frame.width:new_h=160; new_w=max(48,int(frame.width*new_h/frame.height))
        else:new_w=160; new_h=max(48,int(frame.height*new_w/frame.width))
        image=frame.reformat(width=new_w,height=new_h,format="rgb24").to_ndarray().astype(np.float32)/255
        luminance=.2126*image[...,0]+.7152*image[...,1]+.0722*image[...,2]
        edge=float((np.abs(np.diff(luminance,axis=0)).mean()+np.abs(np.diff(luminance,axis=1)).mean())/2)
        output.append(DenseFrame(timestamp,image,float(luminance.mean()),edge))
    con.close(); return output


def _boundaries_in_region(frames: list[DenseFrame],region: dict,fps: float) -> list[dict]:
    if len(frames)<3:return []
    scores=[]; components=[]
    for before,after in zip(frames,frames[1:]):
        pixel=float(np.mean(np.abs(after.rgb-before.rgb)))
        hist=0.0
        for channel in range(3):
            a=np.histogram(before.rgb[...,channel],bins=16,range=(0,1))[0].astype(float); b=np.histogram(after.rgb[...,channel],bins=16,range=(0,1))[0].astype(float)
            hist+=float(np.abs(a/a.sum()-b/b.sum()).sum()/2)/3
        lum=abs(after.luminance-before.luminance)
        score=.62*pixel+.30*hist+.08*lum
        scores.append(score); components.append((pixel,hist,lum))
    values=np.asarray(scores); median=float(np.median(values)); mad=float(np.median(np.abs(values-median)))
    threshold=max(.105,median+max(.055,7*mad))
    events=[]
    for index,score in enumerate(scores):
        if score<threshold:continue
        left=scores[index-1] if index else 0; right=scores[index+1] if index+1<len(scores) else 0
        if score<left or score<right:continue
        before=frames[index]; after=frames[index+1]
        event_type,classification_evidence=_classify(frames,scores,index,threshold)
        isolation=score/max((left+right)/2,.001)
        verification=min(.985,.58+.20*min(1,(score-threshold)/max(.22-threshold,.05))+.20*min(1,(isolation-1)/3))
        candidate=float(region.get("candidate_confidence",.72))
        final=min(.98,.25*candidate+.75*verification)
        subtype=[]
        if event_type=="hard_cut" and components[index][1]<.35 and components[index][2]<.05:
            subtype=[{"type":"jump_cut","confidence":.58,"verification_status":"requires_subject_continuity"}]
        transform=_estimate_transform(before.rgb,after.rgb,before.time,after.time)
        events.append({"timestamp":round((before.time+after.time)/2,4),"start":round(before.time,4),"end":round(after.time,4),"type":event_type,"subtype_candidates":subtype,"transform_evidence":transform,"candidate_confidence":round(candidate,3),"verification_confidence":round(verification,3),"final_confidence":round(final,3),"verification_status":"verified","training_eligible":bool(final>=.8 and event_type!="uncertain_change"),"detection_method":"dense_adjacent_frame_verification","evidence":{"before_frame_time":round(before.time,4),"after_frame_time":round(after.time,4),"adjacent_change_score":round(score,4),"adaptive_threshold":round(threshold,4),"local_median":round(median,4),"pixel_mad":round(components[index][0],4),"rgb_histogram_distance":round(components[index][1],4),"luminance_delta":round(components[index][2],4),"temporal_isolation":round(isolation,3),**classification_evidence},"observed":True,"interpretation":event_type.replace("_"," ")})
    return events


def _classify(frames,scores,index,threshold):
    before=frames[index]; after=frames[index+1]
    pre=frames[max(0,index-1)]; post=frames[min(len(frames)-1,index+2)]
    return_similarity=float(np.mean(np.abs(pre.rgb-post.rgb)))
    peak=max(before.luminance,after.luminance)
    neighbor_scores=scores[max(0,index-3):min(len(scores),index+4)]
    moderate=sum(value>max(.045,threshold*.38) for value in neighbor_scores)
    edge_drop=min(before.edge,after.edge)/max(pre.edge,post.edge,.001)
    if peak>.88 and return_similarity<.075:
        return "flash",{"return_frame_difference":round(return_similarity,4),"peak_luminance":round(peak,4)}
    if min(before.luminance,after.luminance)<.035 and abs(before.luminance-after.luminance)>.16:
        return "fade",{"minimum_luminance":round(min(before.luminance,after.luminance),4)}
    if moderate>=3 and max(neighbor_scores)<.30:
        return "dissolve",{"moderate_change_frames":moderate}
    if edge_drop<.58 and moderate>=2:
        return "whip",{"edge_retention":round(edge_drop,4),"moderate_change_frames":moderate}
    local_baseline=max((sum(neighbor_scores)-scores[index])/max(1,len(neighbor_scores)-1),.001)
    if scores[index]>=threshold*1.15 and scores[index]/local_baseline>=5:
        return "hard_cut",{}
    return "uncertain_change",{"reason":"verified discontinuity but transition family is ambiguous"}


def _deduplicate(events: list[dict],distance: float) -> list[dict]:
    output=[]
    for event in sorted(events,key=lambda item:item["timestamp"]):
        if output and event["timestamp"]-output[-1]["timestamp"]<distance:
            if event["final_confidence"]>output[-1]["final_confidence"]:output[-1]=event
        else:output.append(event)
    return output


def _estimate_transform(before,after,start,end):
    try:
        import cv2
        a=(before*255).astype(np.uint8); b=(after*255).astype(np.uint8); gray_a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); gray_b=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
        orb=cv2.ORB_create(nfeatures=500); key_a,des_a=orb.detectAndCompute(gray_a,None); key_b,des_b=orb.detectAndCompute(gray_b,None)
        if des_a is None or des_b is None:return {"status":"insufficient_features"}
        matches=sorted(cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(des_a,des_b),key=lambda item:item.distance)[:100]
        if len(matches)<8:return {"status":"insufficient_matches","matches":len(matches)}
        points_a=np.float32([key_a[item.queryIdx].pt for item in matches]); points_b=np.float32([key_b[item.trainIdx].pt for item in matches]); matrix,inliers=cv2.estimateAffinePartial2D(points_a,points_b,method=cv2.RANSAC,ransacReprojThreshold=3)
        if matrix is None:return {"status":"affine_fit_failed"}
        scale=float(np.sqrt(matrix[0,0]**2+matrix[0,1]**2)); rotation=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))); dx=float(matrix[0,2]/before.shape[1]); dy=float(matrix[1,2]/before.shape[0]); ratio=float(inliers.mean()) if inliers is not None else 0
        kind="punch_in" if scale>1.04 else "punch_out" if scale<.96 else "reframe" if abs(dx)>.03 or abs(dy)>.03 else "no_strong_transform"
        confidence=round(min(.9,.35+.55*ratio),3)
        return {"status":"measured","type_candidate":kind,"confidence":confidence,"parameters":{"scale_from":1.0,"scale_to":round(scale,4),"translation_x_frame":round(dx,4),"translation_y_frame":round(dy,4),"rotation_degrees":round(rotation,3),"duration_ms":round((end-start)*1000,1)},"evidence":{"feature_matches":len(matches),"ransac_inlier_ratio":round(ratio,3)},"verification_status":"supported" if confidence>=.7 and kind!="no_strong_transform" else "low_confidence_or_absent"}
    except ImportError:return {"status":"opencv_unavailable"}


def _merge_regions(regions: list[dict],duration: float) -> list[dict]:
    if not regions:return []
    output=[]
    for region in regions:
        if output and region["start"]<=output[-1]["end"]+.05:
            output[-1]["end"]=region["end"]
            output[-1]["candidate_confidence"]=max(output[-1]["candidate_confidence"],region["candidate_confidence"])
        else:output.append(dict(region))
    for region in output:region["end"]=min(duration,region["end"])
    return output
