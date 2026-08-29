from __future__ import annotations


def align_modalities(report: dict) -> dict:
    """Build synchronized evidence without promoting weak correlations to intent."""
    transcript=report.get("transcript") or {}; words=transcript.get("words",[]); sentences=transcript.get("sentences",[]); pauses=transcript.get("pauses",[]); emphasis=transcript.get("emphasized_words",[])
    audio=report.get("audio") or {}; transients=(audio.get("events") or {}).get("transients",[])
    edits=(report.get("editing") or {}).get("verified_events",[])
    overlay=(report.get("text_overlay") or {}).get("track",[])

    # Classify only the class that can be supported without stems: speech attacks.
    for transient in transients:
        timestamp=transient["timestamp"]
        overlapping=[word for word in words if word["start"]-.035<=timestamp<=word["end"]+.035]
        if overlapping:
            transient.update({"class":"speech_attack","confidence":round(max(.55,float(overlapping[0]["confidence"])*.8),3),"verification_status":"aligned_to_spoken_word","aligned_word":overlapping[0]["display"]})

    cross_modal=[]; intent_events=[]
    for edit in edits:
        timestamp=edit["timestamp"]
        nearby_words=[word for word in words if word["start"]-.25<=timestamp<=word["end"]+.25]
        nearby_sentences=[sentence for sentence in sentences if abs(sentence["end"]-timestamp)<=.35 or abs(sentence["start"]-timestamp)<=.35]
        nearby_pauses=[pause for pause in pauses if pause["start"]-.25<=timestamp<=pause["end"]+.25]
        nearby_emphasis=[item for item in emphasis if item["start"]-.2<=timestamp<=item["end"]+.2]
        nearby_transients=[item for item in transients if abs(item["timestamp"]-timestamp)<=.12]
        beat_grid=(audio.get("events") or {}).get("beat_grid",[]) if (audio.get("events") or {}).get("beat_status")=="verified" else []
        nearby_beats=[beat for beat in beat_grid if abs(beat-timestamp)<=.08]
        nearby_ocr=[item for item in overlay if item["start"]-.25<=timestamp<=item["end"]+.25]
        observed={"edit":{"type":edit["type"],"timestamp":timestamp,"confidence":edit["final_confidence"]},"speech_words":nearby_words,"sentence_boundaries":nearby_sentences,"pauses":nearby_pauses,"emphasis":nearby_emphasis,"audio_transients":nearby_transients,"verified_beats":nearby_beats,"ocr":nearby_ocr,"visual_subject_change":{"status":"not_measured"}}
        cross_modal.append({"timeline_id":f"edit-{len(cross_modal):04d}","start":edit["start"],"end":edit["end"],"observed":observed,"interpretations":[]})
        candidates=[]
        if nearby_emphasis:candidates.append({"intent":"spoken_keyword_emphasis","confidence":_score(edit,.18,bool(nearby_transients)),"evidence":["verified_edit","emphasized_word"]+( ["aligned_transient"] if nearby_transients else [])})
        if nearby_sentences or nearby_pauses:candidates.append({"intent":"sentence_or_section_boundary","confidence":_score(edit,.16,bool(nearby_sentences and nearby_pauses)),"evidence":["verified_edit"]+(["sentence_boundary"] if nearby_sentences else [])+(["speech_pause"] if nearby_pauses else [])})
        if nearby_transients:candidates.append({"intent":"audio_transient_aligned_edit","confidence":_score(edit,.10,False),"evidence":["verified_edit","audio_transient"],"note":"Not called beat-driven because no verified music beat exists."})
        if nearby_beats:candidates.append({"intent":"beat_alignment","confidence":_score(edit,.20,True),"evidence":["verified_edit","verified_music_beat"]})
        if nearby_ocr and nearby_emphasis:candidates.append({"intent":"spoken_and_caption_emphasis","confidence":_score(edit,.20,True),"evidence":["verified_edit","on_screen_text","emphasized_word"]})
        candidates=sorted(candidates,key=lambda item:item["confidence"],reverse=True)
        status="supported" if candidates and len(candidates[0]["evidence"])>=3 else "low_confidence_candidates" if candidates else "insufficient_evidence"
        intent_events.append({"timestamp":timestamp,"edit_type":edit["type"],"status":status,"intent_candidates":candidates,"no_single_intent_asserted":True})
    report["cross_modal_events"]=cross_modal
    report["edit_intent"]={"status":"evidence_gated","events":intent_events,"policy":"Intent is multi-label and follows observed measurement, verified detection, and temporal alignment."}
    report["master_timeline"]=_master_timeline(report)
    return report


def _score(edit: dict,addition: float,agreement: bool):
    base=float(edit.get("final_confidence",0))*.55
    return round(min(.92,base+addition+(.12 if agreement else 0)),3)


def _master_timeline(report: dict):
    entries=[]
    for edit in (report.get("editing") or {}).get("verified_events",[]):entries.append({"time":edit["timestamp"],"modality":"editing","event":"verified_boundary","ref":edit["type"],"confidence":edit["final_confidence"]})
    for sentence in (report.get("transcript") or {}).get("sentences",[]):entries.append({"time":sentence["start"],"end":sentence["end"],"modality":"speech","event":"sentence","ref":sentence["sentence_id"],"confidence":sentence["confidence"]})
    for pause in (report.get("transcript") or {}).get("pauses",[]):entries.append({"time":pause["start"],"end":pause["end"],"modality":"speech","event":"pause","ref":pause["type"],"confidence":pause.get("confidence",.9),"detection_method":"gap_between_timestamped_words"})
    for transient in (report.get("audio") or {}).get("events",{}).get("transients",[]):entries.append({"time":transient["timestamp"],"modality":"audio","event":transient["class"],"confidence":transient["confidence"]})
    for text in (report.get("text_overlay") or {}).get("track",[]):entries.append({"time":text["start"],"end":text["end"],"modality":"text_overlay","event":text["role_candidate"],"ref":text["text"],"confidence":text["confidence"]})
    return sorted(entries,key=lambda item:item["time"])
