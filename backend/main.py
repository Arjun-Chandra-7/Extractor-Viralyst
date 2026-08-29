from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import analyse, enrich_transcript
from .capacity import capacity_plan
from .turbo import analyse_turbo

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
UPLOADS = ROOT / "uploads"
REPORTS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="VIRALYST Extractor", version="0.1.0")
app.mount("/app", StaticFiles(directory=ROOT / "frontend", html=True), name="app")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.post("/api/analyse")
async def analyse_video(video: UploadFile = File(...), mode: str = "turbo"):
    if not video.filename:
        raise HTTPException(400, "A video file is required")
    suffix = Path(video.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise HTTPException(415, "Unsupported video format")
    report_id = uuid.uuid4().hex[:12]
    video_path = UPLOADS / f"{report_id}{suffix}"
    with video_path.open("wb") as target:
        shutil.copyfileobj(video.file, target)
    started = time.perf_counter()
    try:
        engine = analyse if mode == "detailed" else analyse_turbo
        report = await asyncio.to_thread(engine, video_path, report_id, video.filename)
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not analyse this media: {exc}") from exc
    report["processing"]["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (REPORTS / f"{report_id}.json").write_text(json.dumps(report, indent=2))
    return report


@app.get("/api/capacity")
def estimate_capacity(videos: int = 50_000, hours: float = 1, average_megabytes: float = 12, download_gbps: float = 1, workers: int = 16, seconds_per_video_per_worker: float = 4):
    return capacity_plan(videos, hours, average_megabytes, download_gbps, workers, seconds_per_video_per_worker)


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    path = REPORTS / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return json.loads(path.read_text())


@app.get("/api/reports/{report_id}/download")
def download_report(report_id: str):
    path = REPORTS / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="application/json", filename=f"viralyst-{report_id}.json")


@app.post("/api/transcribe/{report_id}")
async def transcribe(report_id: str):
    report_path = REPORTS / f"{report_id}.json"
    candidates = list(UPLOADS.glob(f"{report_id}.*"))
    if not report_path.exists() or not candidates:
        raise HTTPException(404, "Video or report not found")
    report = json.loads(report_path.read_text())
    try:
        report["transcript"] = await asyncio.to_thread(enrich_transcript, candidates[0])
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    report_path.write_text(json.dumps(report, indent=2))
    return report
