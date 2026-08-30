#!/usr/bin/env python3
"""VIRALYST High-Throughput Corpus Training Pipeline CLI."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from backend.corpus import CorpusRunner
from backend.watch import VIDEO_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description="VIRALYST High-Throughput Corpus Training Pipeline (<= 4h / 7,500 videos)")
    parser.add_argument("--input", "-i", type=Path, default=Path("drop-videos-here"), help="Directory containing input videos")
    parser.add_argument("--output", "-o", type=Path, default=Path("corpus-reports"), help="Directory to store extracted JSON reports")
    parser.add_argument("--jsonl", type=Path, default=None, help="Path for compact streaming corpus.jsonl dataset file")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of concurrent worker threads (default: 4)")
    parser.add_argument("--transcript-model", default="base.en", help="Faster-Whisper model (default: base.en)")
    parser.add_argument("--delete-source", action="store_true", help="Delete source video after successful report commit")
    parser.add_argument("--benchmark-dir", type=Path, default=None, help="Benchmark distinct real videos from this directory; duplicates are rejected")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if args.benchmark_dir:
        bench_dir=args.benchmark_dir.resolve(); video_paths=[p for p in bench_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
        if len(video_paths)<2: print("Benchmark requires at least two distinct real videos."); sys.exit(2)
        print(f"=== COLD CORPUS THROUGHPUT BENCHMARK: {len(video_paths)} DISTINCT VIDEOS ===")
        runner = CorpusRunner(bench_dir, output_dir, jsonl_path=args.jsonl, max_workers=args.workers, transcript_model=args.transcript_model)
        t0 = time.perf_counter()
        stats = runner.run_batch(video_paths)
        wall_time = time.perf_counter() - t0

        print("\n=== BENCHMARK REPORT ===")
        print(f"Requested/full/duplicates: {stats['requested']}/{stats['full_extractions']}/{stats['duplicates']}")
        print(f"Total wall time:           {stats['total_wall_seconds']:.2f} s")
        print(f"Throughput:                {stats['videos_per_minute']:.2f} videos/minute")
        print(f"Projected 7,500 runtime:   {stats['projected_7500_hours']:.2f} hours")
        print(f"4-HOUR SLA STATUS:         {'PASS' if stats['sla_passed'] else 'FAIL'}")
        return

    # Normal directory processing
    video_files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Starting CORPUS_TRAIN pipeline on {len(video_files)} videos with {args.workers} workers...")
    runner = CorpusRunner(input_dir, output_dir, jsonl_path=args.jsonl, max_workers=args.workers, transcript_model=args.transcript_model, delete_source_after_commit=args.delete_source)
    stats = runner.run_batch(video_files)

    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total processed:           {stats['total_processed']}")
    print(f"Total wall time:           {stats['total_wall_seconds']:.2f} s")
    print(f"Throughput:                {stats['videos_per_minute']:.2f} videos/minute")
    print(f"Projected 7,500 runtime:   {stats['projected_7500_hours']:.2f} hours")
    print(f"SLA Target Passed:         {stats['sla_passed']}")


if __name__ == "__main__":
    main()
