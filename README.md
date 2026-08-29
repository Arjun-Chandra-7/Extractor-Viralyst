# VIRALYST Extractor

Fast, per-video intelligence reports for short-form video. The core analyser uses PyAV/FFmpeg decoding and numpy feature extraction, so a report is useful immediately without a cloud roundtrip. Transcript and OCR are opt-in enrichments: install a local Whisper model and/or wire an OCR provider for them.

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

Then copy videos into `drop-videos-here/`. For each completed copy, the watcher creates an individual JSON report in `watched-reports/`, validates that report, and deletes the source video. If extraction fails, the video is moved to `failed-videos/` and is **not** deleted.

Turbo mode is the default. For the slower detailed CPU report:

```bash
./watch-videos.sh --mode detailed
```

## Design notes

- `POST /api/analyse` produces one independent report per uploaded video.
- The fast pass samples video at a bounded cadence (maximum 96 frames) and audio at a 22.05 kHz mono analysis rate. It is designed to make a first report in seconds rather than decode every frame of a long source.
- It measures first, then derives semantic labels. Confidence and evidence are carried on every inferred edit event.
- `POST /api/transcribe/{report_id}` adds exact Whisper word timestamps when a local Faster-Whisper model is configured. This is deliberately separate from the fast pass.

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
