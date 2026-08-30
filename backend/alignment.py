from __future__ import annotations

import re


def align_modalities(report: dict) -> dict:
    """Build synchronized evidence, transcript-caption alignment, SFX classification, and semantic intelligence."""
    transcript = report.get("transcript") or {}
    words = transcript.get("words", [])
    sentences = transcript.get("sentences", [])
    pauses = transcript.get("pauses", [])
    emphasis = transcript.get("emphasized_words", [])

    audio = report.get("audio") or {}
    transients = (audio.get("events") or {}).get("transients", [])
    edits = (report.get("editing") or {}).get("verified_events", [])
    overlay = (report.get("text_overlay") or {}).get("track", [])
    duration = float(report.get("source", {}).get("duration_seconds", 0))

    # 1. SFX Classification & Audio Transient Enrichment
    _classify_sfx_transients(transients, words, edits, audio)

    # 2. Spoken Transcript <-> Caption Alignment
    alignment_summary = _align_transcript_with_captions(words, overlay)
    report["transcript_caption_alignment"] = alignment_summary

    # 3. Semantic Sections
    semantic_sections = _extract_semantic_sections(duration, sentences, edits, overlay)
    report["semantic"] = {
        "status": "extracted_from_multimodal_evidence" if semantic_sections else "deferred",
        "sections": semantic_sections,
        "method": "multimodal_linguistic_and_pacing_heuristics",
    }

    # 4. Cross-Modal Events & Rich Interpretations
    cross_modal = []
    intent_events = []

    for edit in edits:
        timestamp = edit["timestamp"]
        nearby_words = [w for w in words if w["start"] - 0.25 <= timestamp <= w["end"] + 0.25]
        nearby_sentences = [s for s in sentences if abs(s["end"] - timestamp) <= 0.35 or abs(s["start"] - timestamp) <= 0.35]
        nearby_pauses = [p for p in pauses if p["start"] - 0.25 <= timestamp <= p["end"] + 0.25]
        nearby_emphasis = [item for item in emphasis if item["start"] - 0.2 <= timestamp <= item["end"] + 0.2]
        nearby_transients = [item for item in transients if abs(item["timestamp"] - timestamp) <= 0.12]
        beat_grid = (audio.get("events") or {}).get("beat_grid", []) if (audio.get("events") or {}).get("beat_status") == "verified" else []
        nearby_beats = [b for b in beat_grid if abs(b - timestamp) <= 0.08]
        nearby_ocr = [item for item in overlay if item["start"] - 0.25 <= timestamp <= item["end"] + 0.25]
        current_section = next((sec for sec in semantic_sections if sec["start"] <= timestamp <= sec["end"]), None)

        observed = {
            "edit": {"type": edit["type"], "timestamp": timestamp, "confidence": edit["final_confidence"]},
            "speech_words": nearby_words,
            "sentence_boundaries": nearby_sentences,
            "pauses": nearby_pauses,
            "emphasis": nearby_emphasis,
            "audio_transients": nearby_transients,
            "verified_beats": nearby_beats,
            "ocr": nearby_ocr,
            "semantic_section": current_section["type"] if current_section else "body",
        }

        # Derive rich cross-modal interpretations
        interpretations = []
        if timestamp <= min(3.5, duration * 0.25) or (current_section and current_section["type"] == "hook"):
            interpretations.append("hook_emphasis")
        if nearby_sentences and (nearby_pauses or abs(nearby_sentences[0]["end"] - timestamp) <= 0.2):
            interpretations.append("pacing_reset")
        if nearby_emphasis or (nearby_ocr and any(c.get("animation", {}).get("scale_pop") for c in nearby_ocr)):
            interpretations.append("keyword_emphasis")
        if nearby_beats:
            interpretations.append("beat_sync")
        if current_section and current_section["type"] == "cta":
            interpretations.append("cta_emphasis")
        if current_section and current_section["type"] == "punchline":
            interpretations.append("punchline_cut")
        if any("?" in s.get("text", "") for s in nearby_sentences):
            interpretations.append("question_prompt_cut")
        if edit.get("type") in {"scene_change", "hard_cut"} and not nearby_words:
            interpretations.append("visual_proof")

        cross_modal.append({
            "timeline_id": f"edit-{len(cross_modal):04d}",
            "start": edit["start"],
            "end": edit["end"],
            "observed": observed,
            "interpretations": sorted(set(interpretations)),
        })

        candidates = []
        if nearby_emphasis:
            candidates.append({"intent": "spoken_keyword_emphasis", "confidence": _score(edit, 0.18, bool(nearby_transients)), "evidence": ["verified_edit", "emphasized_word"] + (["aligned_transient"] if nearby_transients else [])})
        if nearby_sentences or nearby_pauses:
            candidates.append({"intent": "sentence_or_section_boundary", "confidence": _score(edit, 0.16, bool(nearby_sentences and nearby_pauses)), "evidence": ["verified_edit"] + (["sentence_boundary"] if nearby_sentences else []) + (["speech_pause"] if nearby_pauses else [])})
        if nearby_transients:
            candidates.append({"intent": "audio_transient_aligned_edit", "confidence": _score(edit, 0.10, False), "evidence": ["verified_edit", "audio_transient"], "note": "Aligned with transient attack."})
        if nearby_beats:
            candidates.append({"intent": "beat_alignment", "confidence": _score(edit, 0.20, True), "evidence": ["verified_edit", "verified_music_beat"]})
        if nearby_ocr and nearby_emphasis:
            candidates.append({"intent": "spoken_and_caption_emphasis", "confidence": _score(edit, 0.20, True), "evidence": ["verified_edit", "on_screen_text", "emphasized_word"]})
        if interpretations:
            for interp in interpretations:
                if interp not in [c["intent"] for c in candidates]:
                    candidates.append({"intent": interp, "confidence": _score(edit, 0.14, True), "evidence": ["multimodal_synchrony", interp]})

        candidates = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        status = "supported" if candidates and len(candidates[0]["evidence"]) >= 3 else "low_confidence_candidates" if candidates else "insufficient_evidence"
        intent_events.append({"timestamp": timestamp, "edit_type": edit["type"], "status": status, "intent_candidates": candidates, "interpretations": interpretations, "no_single_intent_asserted": True})

    report["cross_modal_events"] = cross_modal
    report["edit_intent"] = {
        "status": "evidence_gated",
        "events": intent_events,
        "policy": "Intent is multi-label and follows observed measurement, verified detection, and temporal alignment.",
    }
    report["master_timeline"] = _master_timeline(report)
    return report


def _score(edit: dict, addition: float, agreement: bool) -> float:
    base = float(edit.get("final_confidence", 0)) * 0.55
    return round(min(0.92, base + addition + (0.12 if agreement else 0)), 3)


def _classify_sfx_transients(transients: list[dict], words: list[dict], edits: list[dict], audio: dict) -> None:
    for transient in transients:
        ts = transient["timestamp"]
        # Check alignment with speech attacks
        overlapping_words = [w for w in words if w["start"] - 0.04 <= ts <= w["end"] + 0.04]
        if overlapping_words:
            transient.update({
                "class": "speech_attack",
                "confidence": round(max(0.55, float(overlapping_words[0].get("confidence", 0.8)) * 0.85), 3),
                "verification_status": "aligned_to_spoken_word",
                "aligned_word": overlapping_words[0]["display"],
            })
            continue

        # Check alignment with visual edits
        nearby_edits = [e for e in edits if abs(e["timestamp"] - ts) <= 0.06]
        if nearby_edits:
            transient.update({
                "class": "transition_sfx",
                "confidence": 0.72,
                "verification_status": "aligned_to_visual_edit",
                "aligned_edit_type": nearby_edits[0]["type"],
            })
            continue

        # Classify by spectral characteristics & strength
        db_above = transient.get("strength_db_above_median", 0)
        if db_above >= 10.0:
            transient.update({
                "class": "impact",
                "confidence": 0.65,
                "verification_status": "high_dynamic_transient",
            })
        elif db_above >= 6.0:
            transient.update({
                "class": "click_or_tick",
                "confidence": 0.50,
                "verification_status": "moderate_transient",
            })
        else:
            transient.update({
                "class": "unknown_transient",
                "confidence": 0.30,
                "verification_status": "unclassified",
            })


def _align_transcript_with_captions(words: list[dict], captions: list[dict]) -> dict:
    """Local monotonic sequence alignment between on-screen OCR captions and spoken transcript words.

    Enforces:
    - Strictly bounded match_score in [0.0, 1.0]
    - Monotonic index progression within and across sequential captions
    - Local temporal window gating
    - One-to-one word ownership with reused word tracking
    - Clean separation of displayed_words, highlighted_words, and emphasized_displayed_words
    - Alignment quality gating and training eligibility
    """
    matched_pairs = []
    total_caption_words = 0
    matched_caption_words = 0
    cursor_idx = 0  # Advancing monotonic spoken word pointer
    globally_assigned_indices = set()

    for cap in captions:
        displayed_words = cap.get("text", "").split()
        cap_tokens = _alignment_tokens(cap.get("text", ""))
        if not cap_tokens:
            continue
        total_caption_words += len(cap_tokens)
        cap_start = cap["start"]
        cap_end = cap["end"]

        # Search in a local forward temporal window around the caption
        search_start = max(0, cursor_idx - 1)
        candidate_words = [
            (idx, w) for idx, w in enumerate(words[search_start:], start=search_start)
            if cap_start - 1.0 <= w["start"] <= cap_end + 1.0
        ]

        spoken_refs = []
        unmatched_caption_tokens = []
        lead_lag = None
        highest_matched_idx = cursor_idx
        reused_word_count = 0

        # Monotonic greedy token alignment
        temp_word_idx = search_start
        for c_token in cap_tokens:
            token_matched = False
            for idx, word in candidate_words:
                if idx < temp_word_idx:
                    continue
                w_tokens = _alignment_tokens(word.get("word", ""))
                w_clean = w_tokens[0] if w_tokens else ""
                if w_clean == c_token:
                    is_reused = idx in globally_assigned_indices
                    if is_reused:
                        reused_word_count += 1

                    spoken_refs.append({
                        "index": idx,
                        "word": word["word"],
                        "display": word["display"],
                        "spoken_start": word["start"],
                        "spoken_end": word["end"],
                        "word_reused": is_reused,
                    })
                    globally_assigned_indices.add(idx)

                    if lead_lag is None:
                        lead_lag = round(cap_start - word["start"], 3)
                    temp_word_idx = idx + 1
                    highest_matched_idx = max(highest_matched_idx, idx)
                    token_matched = True
                    break

            if not token_matched:
                unmatched_caption_tokens.append(c_token)

        if spoken_refs:
            cursor_idx = highest_matched_idx + 1

        matched_count = len(spoken_refs)
        # Strictly bounded similarity in [0.0, 1.0]
        lexical_coverage = round(matched_count / max(len(cap_tokens), 1), 3)
        match_score = lexical_coverage
        matched_caption_words += matched_count

        # Identify omitted spoken words in this temporal span
        omitted = []
        if candidate_words:
            matched_indices = {r["index"] for r in spoken_refs}
            for idx, word in candidate_words:
                if idx not in matched_indices and word["start"] >= cap_start and word["end"] <= cap_end:
                    omitted.append(word["display"])

        # Check monotonic index ordering
        monotonicity_passed = all(
            spoken_refs[i]["index"] < spoken_refs[i + 1]["index"]
            for i in range(len(spoken_refs) - 1)
        )

        temporal_error = abs(lead_lag) if lead_lag is not None else 0.0

        # Highlighting & emphasis semantics:
        # displayed_words: physically visible words
        # highlighted_words: words with distinct visual styling
        # emphasized_displayed_words: non-empty ONLY when highlighted words exist
        highlighted_list = cap.get("word_highlighting", {}).get("highlighted_words", [])
        emphasized_displayed = [w for w in highlighted_list if w in cap.get("text", "")]

        # Quality gating & verification
        if match_score >= 0.70 and monotonicity_passed and reused_word_count == 0 and temporal_error <= .5:
            verification_status = "verified"
            training_eligible = True
        elif match_score >= 0.40 and monotonicity_passed:
            verification_status = "candidate"
            training_eligible = False
        else:
            verification_status = "rejected"
            training_eligible = False

        alignment_entry = {
            "caption_text": cap.get("text"),
            "caption_start": cap_start,
            "caption_end": cap_end,
            "match_score": match_score,
            "temporal_error_seconds": round(temporal_error, 3),
            "lead_lag_seconds": lead_lag if lead_lag is not None else 0.0,
            "lexical_coverage": lexical_coverage,
            "monotonicity_passed": monotonicity_passed,
            "reused_word_count": reused_word_count,
            "unmatched_caption_words": unmatched_caption_tokens,
            "omitted_spoken_words": omitted,
            "displayed_words": displayed_words,
            "highlighted_words": highlighted_list,
            "emphasized_displayed_words": emphasized_displayed,
            "spoken_word_refs": spoken_refs,
            "verification_status": verification_status,
            "training_eligible": training_eligible,
            "status": "aligned" if match_score >= 0.50 else "partial_or_graphic_text",
        }
        cap["transcript_alignment"] = alignment_entry
        cap["alignment_confidence"] = match_score
        matched_pairs.append(alignment_entry)

    overall_coverage = round(matched_caption_words / max(total_caption_words, 1), 3) if total_caption_words else 0.0
    return {
        "status": "complete",
        "alignment_method": "local_monotonic_sequence_matching",
        "total_caption_events": len(captions),
        "caption_word_coverage": overall_coverage,
        "verified_aligned_count": sum(1 for a in matched_pairs if a["training_eligible"]),
        "alignments": matched_pairs,
    }


def _alignment_tokens(text: str) -> list[str]:
    """Normalize case/punctuation but preserve contractions as one lexical token."""
    return [token.replace("’", "'").lower() for token in re.findall(r"[\w]+(?:['’][\w]+)?", text)]


def _extract_semantic_sections(duration: float, sentences: list[dict], edits: list[dict], overlay: list[dict]) -> list[dict]:
    """Extract structural narrative hypotheses.

    Note: These are rule-based lexical/structural hypotheses, NOT core ground truth training labels.
    """
    if not sentences and duration <= 0:
        return []

    sections = []
    hook_end = min(3.5, duration * 0.25) if duration > 0 else 3.5

    # 1. Hook (opening 0-3.5s)
    hook_sentences = [s for s in sentences if s["start"] < hook_end]
    hook_text = " ".join(s["text"] for s in hook_sentences) if hook_sentences else "Opening hook"
    sections.append({
        "section_id": 0,
        "type": "hook",
        "start": 0.0,
        "end": round(hook_sentences[-1]["end"] if hook_sentences else hook_end, 3),
        "text": hook_text,
        "confidence": 0.65,
        "evidence": ["opening_interval_heuristic", "pacing_hook"],
        "verification_status": "structural_hypothesis_unverified",
    })

    # 2. Process remaining sentences
    for idx, s in enumerate(sentences):
        if s["end"] <= sections[0]["end"]:
            continue
        text = s.get("text", "")
        text_lower = text.lower()
        start = s["start"]
        end = s["end"]

        cta_keywords = ["subscribe", "follow", "comment", "link in bio", "share", "check out", "let me know", "like the video", "save this"]
        if any(kw in text_lower for kw in cta_keywords):
            sec_type = "cta"
            conf = 0.70
        elif "?" in text or any(text_lower.startswith(w) for w in ["what", "why", "how", "do you", "is there", "can you", "should "]):
            sec_type = "question"
            conf = 0.65
        elif idx == len(sentences) - 1 and (end >= duration - 3.0 or duration < 10):
            sec_type = "conclusion"
            conf = 0.60
        elif any(kw in text_lower for kw in ["because", "reason", "in my opinion", "think", "helps for", "that is why"]):
            sec_type = "explanation"
            conf = 0.60
        elif any(kw in text_lower for kw in ["for example", "look at", "see this", "shows that", "proof"]):
            sec_type = "proof"
            conf = 0.60
        else:
            sec_type = "setup" if start < duration * 0.4 else "payoff"
            conf = 0.55

        sections.append({
            "section_id": len(sections),
            "type": sec_type,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "confidence": conf,
            "evidence": ["sentence_semantics", "lexical_structure_heuristic"],
            "verification_status": "structural_hypothesis_unverified",
        })

    return sections


def _master_timeline(report: dict) -> list[dict]:
    entries = []
    # Editing boundaries
    for edit in (report.get("editing") or {}).get("verified_events", []):
        tier = "training_eligible" if edit.get("training_eligible") else "observed"
        entries.append({
            "time": edit["timestamp"],
            "modality": "editing",
            "event": "verified_boundary",
            "ref": edit["type"],
            "confidence": edit["final_confidence"],
            "tier": tier,
        })

    # Speech sentences
    for sentence in (report.get("transcript") or {}).get("sentences", []):
        entries.append({
            "time": sentence["start"],
            "end": sentence["end"],
            "modality": "speech",
            "event": "sentence",
            "ref": sentence["sentence_id"],
            "confidence": sentence["confidence"],
            "tier": "training_eligible" if sentence["confidence"] >= 0.7 else "observed",
        })

    # Pauses
    for pause in (report.get("transcript") or {}).get("pauses", []):
        entries.append({
            "time": pause["start"],
            "end": pause["end"],
            "modality": "speech",
            "event": "pause",
            "ref": pause["type"],
            "confidence": pause.get("confidence", 0.9),
            "tier": "observed",
        })

    # Audio transients
    for transient in (report.get("audio") or {}).get("events", {}).get("transients", []):
        tier = "training_eligible" if transient.get("verification_status") in {"aligned_to_spoken_word", "aligned_to_visual_edit"} else "observed"
        entries.append({
            "time": transient["timestamp"],
            "modality": "audio",
            "event": transient["class"],
            "confidence": transient["confidence"],
            "tier": tier,
        })

    # Captions / Text overlay
    for text in (report.get("text_overlay") or {}).get("track", []):
        entries.append({
            "time": text["start"],
            "end": text["end"],
            "modality": "text_overlay",
            "event": text["role_candidate"],
            "ref": text["text"],
            "confidence": text["confidence"],
            "tier": "training_eligible" if text["confidence"] >= 0.8 else "observed",
        })

    return sorted(entries, key=lambda item: item["time"])
