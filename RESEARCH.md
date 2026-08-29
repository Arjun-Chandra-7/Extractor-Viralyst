# Research-backed extraction stack

The product should never collapse analysis into a single “AI score”. It needs two lanes:

| Lane | Budget | Purpose | Engines |
| --- | --- | --- | --- |
| Fast pass | 10–15 seconds for short-form sources on a GPU workstation | report immediately; create candidates and measurements | PyAV/FFmpeg, linear-light colour metrics, spectral audio, TransNetV2/Adaptive scene detection, VAD, OCR on keyframes |
| Deep enrichment | asynchronous / opt-in | validate semantic claims and stem/speaker context | Whisper large-v3-turbo, pyannote, Demucs, visual-language evidence pass |

## 50K/hour execution architecture

At this scale the unit of work is a shard, not an HTTP request. Downloads, sparse decode, GPU inference, report writing and training-shard writing run as overlapping bounded queues with backpressure. For 50,000 videos/hour the sustained completion rate is 13.89 videos/second. At 12 MB/video the payload is 600 GB, requiring about 1.52 Gbit/s after a modest 12% transport/retry allowance.

Use NVDEC through NVIDIA DALI on production GPU workers, with multiple decode sessions and micro-batched 8-frame clips. X-CLIP is the default transferable visual embedding because its published checkpoint consumes eight 224×224 frames and supports video/text retrieval and classification. Store embeddings and measured features once; train small task-specific heads from those artifacts rather than decoding all videos again each epoch. Reports remain individual JSON documents, while the training index should be sharded Parquet/Arrow for parallel scanning.

## Chosen components

- **Shot boundaries:** [TransNetV2](https://github.com/soCzech/TransNetV2) should replace simple histogram-only cuts in the deep pass. It distinguishes abrupt and gradual transitions and reports strong benchmark results. Keep PySceneDetect adaptive/content detection as the no-model fallback.
- **Text:** [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) is the scene-text engine, invoked only on selected keyframes and tracked across adjacent frames. Persist text, polygon, confidence, source frame, first/last seen times and caption animation features separately from ASR.
- **ASR:** use [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) on GPU for the quality lane; its reduced decoder is deliberately faster than large-v3. Faster-Whisper `small`/`base` remains the instant preview setting. Word times must be aligned/word-level, never guessed from segment text.
- **Diarization:** use [pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1) after ASR and intersect turns with ASR word spans. It requires an accepted model license/token, so it cannot be an invisible hard dependency.
- **Stems:** use [Demucs](https://github.com/facebookresearch/demucs) only as a deep job. It produces vocals/drums/bass/other stems, but full-fidelity separation is not a 10-second guarantee. Score speech/music balance from stem energy after separation, never from the mixed waveform alone.
- **Colour:** use [Colour](https://github.com/colour-science/colour) where colour-space accuracy matters. Decode respecting transfer metadata, transform into linear-light/CIELAB, and report quantiles and deltas; `RGB mean` is not a grade.

## Editing event policy

Every event must have `start`, `end`, `type`, `confidence`, `evidence`, and `intent_candidates`. An edit is only promoted from *candidate* when at least two compatible signals exist—for example a shot boundary plus scale change (punch-in), or a cut plus beat onset (beat cut). “Match cut”, “B-roll”, “green screen”, “speed ramp”, and “J/L cut” require visual/audio temporal evidence and should remain `unverified` when that evidence is unavailable.

## High-fidelity report contract

- Audio loudness uses EBU R128 / ITU-R BS.1770 integrated, short-term and momentary values in the deep lane. The fast report labels its estimate as a proxy.
- Video grading metrics are calculated per shot in linear light / Lab: exposure quantiles, clipping, contrast, chroma, hue distribution, white-point bias, RGB relationship, split-tone delta and local-contrast proxy.
- Intent is an inference, not a detection. Reports must expose the specific timing evidence (spoken keyword, transient, cut, OCR change) and retain alternate explanations.

## Installation profile

`requirements.txt` deliberately remains lightweight. Use `requirements-deep.txt` only on a worker with appropriate GPU/RAM and pre-download model weights during deployment, not on the first user upload.
