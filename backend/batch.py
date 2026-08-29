from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from .turbo import analyse_turbo
from .pipeline import analyse, enrich_transcript
from .ocr import extract_text_overlay
from .alignment import align_modalities
from .contract import atomic_json_write, reconcile_states


def _analyse_one(item: tuple) -> dict:
    path_str, out_dir_str, label, mode, transcript, transcript_model, compact, compress_gzip = item
    path = Path(path_str)
    rid = hashlib.blake2s(str(path).encode(), digest_size=8).hexdigest()

    started = time.perf_counter()
    if mode == "turbo":
        report = analyse_turbo(path, rid, path.name)
    else:
        text_overlay = extract_text_overlay(path, 60.0, target_fps=3.0)
        report = analyse(path, rid, path.name, caption_boxes=text_overlay.get("caption_boxes_for_exclusion"))
        report["text_overlay"] = text_overlay
        if transcript:
            report["transcript"] = enrich_transcript(path, transcript_model)
        report = align_modalities(report)
        report = reconcile_states(report)

    if label is not None:
        report["training_label"] = label

    elapsed = round(time.perf_counter() - started, 3)
    report["processing"]["elapsed_seconds"] = elapsed

    ext = ".json.gz" if compress_gzip else ".json"
    target = Path(out_dir_str) / f"{rid}{ext}"
    atomic_json_write(target, report, compact=compact, compress_gzip=compress_gzip)

    features = report.get("training_features", {})
    return {
        "report_id": rid,
        "path": str(path),
        "report": str(target),
        "elapsed": elapsed,
        "features": features.get("values", features),
        "label": label,
    }


def analyse_directory(
    source: Path,
    output: Path,
    workers: int = 4,
    mode: str = "turbo",
    transcript: bool = True,
    transcript_model: str = "base.en",
    compact: bool = True,
    compress_gzip: bool = False,
) -> tuple[int, Path]:
    output.mkdir(parents=True, exist_ok=True)
    extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    files = [p for p in source.rglob("*") if p.suffix.lower() in extensions]
    manifest = output / "training-manifest.jsonl"

    tasks = [
        (str(p), str(output), None, mode, transcript, transcript_model, compact, compress_gzip)
        for p in files
    ]

    with manifest.open("a", buffering=1, encoding="utf-8") as sink, cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_analyse_one, tasks, chunksize=max(1, len(tasks) // (workers * 2) or 1)):
            sink.write(json.dumps(result, separators=(",", ":")) + "\n")

    return len(files), manifest


def benchmark_throughput(source: Path, counts: list[int] = [1, 10], mode: str = "standard", workers: int = 2) -> dict:
    """Benchmark extraction throughput on 1, 10, 100 video workloads."""
    extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    candidate_files = [p for p in source.rglob("*") if p.suffix.lower() in extensions]
    if not candidate_files:
        raise ValueError(f"No video files found in {source}")

    sample_file = candidate_files[0]
    results = {}

    import tempfile
    for count in counts:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            tasks = [
                (str(sample_file), str(tmp_out), None, mode, True, "base.en", True, False)
                for _ in range(count)
            ]
            t0 = time.perf_counter()
            with cf.ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_analyse_one, tasks, chunksize=1))
            total_time = round(time.perf_counter() - t0, 3)
            sec_per_video = round(total_time / count, 3)
            vids_per_min = round(count * 60 / max(total_time, 0.001), 2)
            results[f"{count}_videos"] = {
                "total_seconds": total_time,
                "seconds_per_video": sec_per_video,
                "steady_state_videos_per_minute": vids_per_min,
            }

    return {
        "benchmark_mode": mode,
        "workers": workers,
        "sample_video": sample_file.name,
        "results": results,
    }


def download_manifest(manifest: Path, destination: Path, workers: int = 16) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    def fetch(row):
        key = row.get("id") or hashlib.blake2s(row["url"].encode(), digest_size=8).hexdigest()
        suffix = Path(row["url"].split("?", 1)[0]).suffix or ".mp4"
        target = destination / f"{key}{suffix}"
        if not target.exists():
            request = urllib.request.Request(row["url"], headers={"User-Agent": "VIRALYST/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response, target.open("wb") as sink:
                while chunk := response.read(1024 * 1024):
                    sink.write(chunk)
        return target

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, rows))


def main():
    parser = argparse.ArgumentParser(description="VIRALYST high-throughput corpus runner")
    sub = parser.add_subparsers(dest="command", required=True)

    analyse_cmd = sub.add_parser("analyse")
    analyse_cmd.add_argument("source", type=Path)
    analyse_cmd.add_argument("--output", type=Path, default=Path("corpus-reports"))
    analyse_cmd.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    analyse_cmd.add_argument("--mode", choices=("turbo", "standard"), default="turbo")
    analyse_cmd.add_argument("--transcript", action=argparse.BooleanOptionalAction, default=True)
    analyse_cmd.add_argument("--transcript-model", default="base.en")
    analyse_cmd.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    analyse_cmd.add_argument("--gzip", action=argparse.BooleanOptionalAction, default=False)

    bench_cmd = sub.add_parser("benchmark")
    bench_cmd.add_argument("source", type=Path)
    bench_cmd.add_argument("--mode", choices=("turbo", "standard"), default="standard")
    bench_cmd.add_argument("--counts", default="1,5")
    bench_cmd.add_argument("--workers", type=int, default=2)

    download_cmd = sub.add_parser("download")
    download_cmd.add_argument("manifest", type=Path)
    download_cmd.add_argument("--destination", type=Path, default=Path("corpus-videos"))
    download_cmd.add_argument("--workers", type=int, default=32)

    args = parser.parse_args()
    if args.command == "analyse":
        count, manifest = analyse_directory(
            args.source,
            args.output,
            args.workers,
            args.mode,
            args.transcript,
            args.transcript_model,
            args.compact,
            args.gzip,
        )
        print(json.dumps({"processed": count, "training_manifest": str(manifest)}))
    elif args.command == "benchmark":
        counts = [int(c.strip()) for c in args.counts.split(",") if c.strip()]
        res = benchmark_throughput(args.source, counts, args.mode, args.workers)
        print(json.dumps(res, indent=2))
    else:
        paths = download_manifest(args.manifest, args.destination, args.workers)
        print(json.dumps({"downloaded": len(paths), "destination": str(args.destination)}))


if __name__ == "__main__":
    main()

