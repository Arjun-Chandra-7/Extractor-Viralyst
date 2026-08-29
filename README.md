# VIRALYST Extractor

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the evidence contract and comprehensive implemented capability matrix.

Reports are schema-validated and atomically published. `processing.runtime` records the active decode/CUDA path and device details.

Fast, per-video feature extraction and multimodal intelligence for short-form video datasets (1 to 50,000+ videos). The local STANDARD analyser combines PyAV decoding, GPU Faster-Whisper, RapidOCR with dense tracking, OpenCV YuNet face/subject tracking, BS.1770 / 4x True-Peak audio grading, caption-excluded color science, and evidence-gated editing verification without cloud roundtrips.

## Requirements & GPU Acceleration

- **OS**: Linux / macOS / Windows
- **Python**: 3.10 - 3.13
- **Hardware**: NVIDIA GPU with CUDA recommended for fast ASR (Faster-Whisper CUDA float16) and OpenCV DNN. Runs seamlessly on CPU when CUDA is absent.

```bash
# Install dependencies
python -m pip install -r requirements.txt
```

## Running the Web Interface

```bash
uvicorn backend.main:app --reload --port 8080
```

Open `http://localhost:8080` in your browser. Upload any video to inspect real-time extraction results and download the report JSON.

## Watched Folder Workflow

Run the watcher and leave its terminal open:

```bash
./watch-videos.sh
```

Copy videos into `drop-videos-here/`. The watcher processes each video, creates a validated report in `watched-reports/`, and cleans up source files.

Watched reports include:
- **GPU Faster-Whisper transcript**: non-overlapping monotonic word timestamps, punctuation, language detection, pauses, and multi-factor prosodic emphasis (pitch/F0, duration, energy, stopword filtering).
- **Dense ROI OCR**: false-positive filtering, typography estimation, animation detection, word highlighting, and transcript ↔ caption alignment.
- **Visual & Editing**: subject tracking, optical flow dynamics, hard cut vs jump cut vs scene change discrimination, transform estimation (with subject continuity gating), and freeze frames.
- **Color Science**: caption-excluded luminance/saturation, formal `red_blue_bias` %, skin tone analysis, and vignette/optical proxies.
- **Audio DSP**: 4x oversampled True Peak (`true_peak_dbtp`), speech LUFS, clarity, SNR, sibilance, ducking detection, and transient classification.

## 50K Corpus Batch Processing & Benchmarking

For large-scale dataset training across thousands of videos:

```bash
# Optional: Download video manifest concurrently
python -m backend.batch download videos.jsonl --destination corpus-videos --workers 32

# Batch analyse corpus with multi-process workers (compact / gzip storage options)
python -m backend.batch analyse corpus-videos --output corpus-reports --mode standard --workers 8 --compact --gzip

# Benchmark throughput on your hardware (steady-state videos/minute)
python -m backend.batch benchmark corpus-videos --counts 1,10,100 --workers 4
```

## Running Tests

```bash
python -m unittest discover tests
```

