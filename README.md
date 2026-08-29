# VIRALYST Extractor

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the evidence contract and an honest implemented/partial/deferred capability matrix.

Fast, per-video intelligence reports for short-form video. The local STANDARD analyser combines PyAV/FFmpeg decoding, dense edit verification, Faster-Whisper, RapidOCR, loudness analysis, and measured color features without a cloud roundtrip.

## Run

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8080
```

Open `http://localhost:8080`. Upload a video and download the individual JSON report from the report view.

## Easiest workflow: watched folder

Run the watcher and leave its terminal open:

```bash
./watch-videos.sh
```

Then copy videos into `drop-videos-here/`. For each completed copy, the watcher creates an individual JSON report in `watched-reports/`, validates that report, and deletes the source video. If extraction fails, the video is moved to `failed-videos/` and is **not** deleted. Watched reports include Faster-Whisper transcript intelligence by default: per-word time/confidence/punctuation, pauses, emphasized-word evidence, segment delivery speed and overall WPM.

The default `tiny.en` model is optimized for a few-second CPU pass. For higher accuracy at lower speed, use:

```bash
./watch-videos.sh --transcript-model base.en
```

STANDARD is the watched-folder default. It performs dense adjacent-frame boundary verification, real shot segmentation, ITU-R BS.1770 loudness, timed transcript, bounded OCR, cross-modal alignment, partial region/skin color analysis and evidence-gated intent candidates.

```bash
./watch-videos.sh --mode standard
```

Explicit processing tiers:

- `--mode turbo`: bounded 16-frame corpus triage. It emits change regions only—never cut count, pacing, shots or intent.
- `--mode standard`: accurate local extraction for training/reporting. This is the default.
- `--mode forensic`: runs the STANDARD base and marks unavailable heavyweight forensic engines explicitly; it never fabricates their output.

## Design notes

- `POST /api/analyse` produces one independent report per uploaded video.
- STANDARD densely scans adjacent low-resolution frames for verified boundaries while retaining at most 96 representative color samples. Audio is decoded once at 48 kHz stereo and fanned out to the analyzers.
- It measures first, then derives semantic labels. Confidence and evidence are carried on every inferred edit event.
- Transcript and bounded OCR tracks are produced in STANDARD by default. `POST /api/transcribe/{report_id}` remains available as a retry/re-enrichment endpoint.

## Deep analysis workers

Read [RESEARCH.md](RESEARCH.md) before installing the deep profile. `requirements-deep.txt` contains research-grade engines for TransNetV2 shot transitions, PaddleOCR scene text, pyannote diarization, Demucs stems and standards-aware colour/loudness processing. They belong on a pre-warmed GPU worker: model downloads and full stem separation should never happen inside the request path of an “instant” report.

## 50K corpus mode

Corpus mode intentionally runs a bounded 16-frame sparse pass and an 8 kHz mono audio triage pass. It emits one compact report per video plus an append-only training manifest.

```bash
# Optional: download a JSONL URL manifest concurrently.
python -m backend.batch download videos.jsonl --destination corpus-videos --workers 64

# Analyse all local media with independent processes.
python -m backend.batch analyse corpus-videos --output corpus-reports --workers 24

# Train a small supervised classifier when labels exist, or an autoencoder otherwise.
python -m backend.train corpus-reports/training-manifest.jsonl --output models/corpus.pt

# On a GPU worker, create one reusable X-CLIP video embedding per source.
python -m backend.embed corpus-reports/training-manifest.jsonl --batch-size 32
```

Before promising a deadline, query `/api/capacity` with the average file size, sustained link speed, worker count, and benchmarked per-video time. For example, 50,000 × 12 MB is 600 GB; transferring that in one hour requires roughly 1.52 Gbit/s after a modest protocol/retry allowance.
