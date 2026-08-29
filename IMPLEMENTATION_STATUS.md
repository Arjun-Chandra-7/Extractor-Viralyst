# EXTRACTOR Implementation Status

The report contract follows one direction only:

`measurement -> detection -> alignment -> interpretation`

Observed measurements remain strictly separated from semantic interpretations. Every training-eligible edit has detector provenance, verification status, and bounded calibrated confidence values.

## Implemented Subsystems

### 1. Hardware & Acceleration Runtime
- **CUDA Detection & Runtime Profiling**: Queries PyTorch CUDA, CTranslate2 CUDA device count, ONNX Runtime providers, and `nvidia-smi` GPU driver telemetry.
- **GPU Faster-Whisper**: Automatic CUDA execution (`device="cuda"`, `compute_type="float16"`) with CPU int8 fallback and thread-safe process caching.
- **Contract Verification**: Reports feature `internal_verification_passed: true` (with backwards-compatible `reliable: true` alias) and empirical calibration basis for `minimum_training_confidence: 0.8`.

### 2. Spoken Transcript & Prosody
- **Adaptive Multi-Pass ASR**: Fast GPU Whisper base pass + selective beam search re-transcription on suspicious or low-confidence segments.
- **Monotonic Non-Overlapping Timestamps**: Strictly prevents word overlap collisions (`aligned_end <= next_word.aligned_start`) with minimum 0.04s duration guarantee.
- **Multi-Factor Prosodic Emphasis**: Combines F0 pitch autocorrelation, duration expansion factor, local RMS energy delta (dB), emphatic punctuation, and semantic stopword filtering.

### 3. Dense OCR & Caption Typography
- **False-Positive Filtering**: Rejects oversized full-frame bounding boxes with single-character noise (e.g. the `"9"` artifact) and low-confidence non-alphanumeric text.
- **Dense ROI Tracking**: Tracks active text bounding boxes across time (~3-4 fps) measuring start, end, duration, position trajectories, and motion drift.
- **Typography & Formatting**: Estimates font class (`sans_serif`, `serif`, `display_bold`), fill color (RGB & Hex), stroke/outline detection, shadow, background box, and contrast.
- **Animation & Word Highlighting**: Detects entry animations (`pop_in`, `slide_up`, `instant`) and highlighted uppercase/styled words.
- **Transcript ↔ Caption Alignment**: Computes token matching, `lead_lag_seconds`, omitted spoken words, and emphasized displayed words.

### 4. Visual Analysis & Editing Verification
- **Subject Tracking**: OpenCV YuNet face and subject detection with continuous spatial coordinate tracking (`visual.subjects`).
- **Motion Dynamics**: Optical flow / motion vector proxies measuring motion velocity, pan/tilt proxies, camera dynamics, and jitter (`visual.motion`).
- **Boundary Discrimination**: Distinguishes `jump_cut` (subject continuity preserved with pose jump), `scene_change` (high optical delta + subject change), `hard_cut`, `flash`, `fade`, `dissolve`, `whip`.
- **Transform Fitting Constraint**: Strictly rejects transform fitting across unrelated scene cuts without subject continuity. Detects `punch_in`, `punch_out`, `digital_zoom`, `pan`, `tilt`, `reframe`.
- **Speed Effects & Shot Content**: Detects `freeze_frame` and classifies shots as `talking_head`, `interview_or_context`, `screen_recording_or_graphic`, `b_roll`.

### 5. Color Science & Caption Exclusion
- **Caption Mask Exclusion Pass**: Masks out active OCR bounding boxes before whole-frame color calculations to prevent white captions from skewing scene luminance and white balance.
- **Formal Red-Blue Bias Metric**: Formula `((R_mean - B_mean) / ((R_mean + G_mean + B_mean)/3 + 1e-6)) * 100`, units `%`, neutral range `[-4.0, +4.0]%` calibrated against D65 neutral balance.
- **Subject vs Background Separation**: Luminance and contrast delta between tracked subject and background.
- **Optical Proxies**: High-frequency grain proxy, observational vignette proxy (`center_edge_luminance_delta`), sharpness edge proxy, and bloom/chroma aberration proxies.

### 6. Speech Audio Grading & DSP
- **4x Oversampled True Peak**: Implements polyphase sinc oversampling (`true_peak_dbtp`, ITU-R BS.1770-4).
- **Speech-Specific Grading**: Speech LUFS, speech clarity score (500Hz-3.5kHz), SNR estimate, sibilance ratio (5k-8.5k), de-essing recommendation flag, proximity effect boost, and composite intelligibility index.
- **Stem Separation Band Proxy**: 4-band spectral decomposition (sub-bass, speech core, music mid, high transients), speech-to-music ratio (dB), and music ducking detection.
- **SFX Transient Classification**: Classifies transient hits into `speech_attack`, `transition_sfx`, `impact`, `click_or_tick`, and `unknown_transient`.
- **Compact Momentary Loudness**: Downsamples momentary loudness curves for storage efficiency.

### 7. Multimodal Semantics & 50K Corpus Batching
- **Semantic Narrative Sections**: Identifies `hook`, `question`, `setup`, `explanation`, `proof`, `payoff`, `cta`, and `conclusion`.
- **Cross-Modal Interpretations**: Populates rich semantic descriptors (`hook_emphasis`, `pacing_reset`, `keyword_emphasis`, `beat_sync`, `cta_emphasis`, `punchline_cut`, `visual_proof`).
- **Tiered Master Timeline**: Separates events into `training_eligible`, `observed`, and `low_confidence`.
- **Batching & Throughput Benchmarking**: Supports multi-process cross-video batch extraction, compact JSON, `.json.gz` compression, and automated throughput benchmarks (`python -m backend.batch benchmark`).

## Explicitly Deferred

- Multi-speaker diarization (pyannote DNN).
- Heavyweight source stem separation (Demucs / MDX-Net 4-stem model).
- Deep cross-attention video-language foundation models (X-CLIP, Video-LLaVA).

