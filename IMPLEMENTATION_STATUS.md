# EXTRACTOR implementation status

The report contract follows one direction only:

`measurement -> detection -> alignment -> interpretation`

Observed measurements remain separate from semantic interpretations. Every training-eligible edit has detector provenance, verification status, and a bounded confidence value.

## Implemented now

- Dense adjacent-frame cut verification after a sparse candidate-region scan.
- Separate candidate, verification, and final confidence values; unverified candidates never become cuts or pacing statistics.
- Real `shots` from verified boundaries and separately named `frame_samples`.
- Word- and sentence-timed spoken transcript, punctuation, confidence, language, pauses, delivery rate, and emphasis candidates.
- Separate bounded OCR/text-overlay track.
- A shared master timeline and cross-modal evidence records.
- Conservative multi-label intent candidates. Beat intent is impossible without a verified beat.
- Structured transient events; speech-aligned attacks are classified while unsupported events stay `unknown_transient`.
- Integrated LUFS, loudness curves, LRA proxy, decoded sample peak, dynamics, spectral balance, and stereo-width measurements.
- Region-aware center/edge color measurements, per-shot color, face/skin measurements, and basic grain/vignette/sharpness proxies.
- Basic affine transform candidates with measured scale, translation, rotation, and duration.
- Explicit TURBO, STANDARD, and FORENSIC contracts. STANDARD is the default watcher mode.

## Partially implemented

- Transition classification is signal-based; difficult match cuts, masking transitions, and stylized effects need learned verification.
- BPM and beat candidates are measured but deliberately remain unverified without reliable music isolation.
- OCR tracks text regions and timing; typography and per-word animation are still deferred.
- Skin analysis uses face detection plus a color mask; segmentation-quality subject/background mattes remain deferred.
- Transform detection covers basic affine camera/edit transforms, not full optical-flow effect reconstruction.

## Explicitly deferred (never fabricated)

- Speaker diarization.
- Voice/music/SFX stem separation and stem-aware mix ratios or ducking.
- Learned SFX classes beyond speech alignment.
- Deep video semantics such as reliable B-roll, gameplay, proof reveal, hook, payoff, and CTA classification.
- Full caption typography and animation reconstruction.
- Reliable speed ramps, reverse, freeze frames, green screen, masks, compositing, and advanced VFX parameters.
- True-peak dBTP (requires oversampled true-peak measurement).
- High-confidence editing intent when synchronized evidence is insufficient.

These deferred capabilities belong in later STANDARD/FORENSIC model stages. They are excluded from core-brain training features until verified.
