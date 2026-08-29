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
    parser.add_argument("--benchmark-samples", type=int, default=0, help="Run synthetic batch benchmark on N cloned samples")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if args.benchmark_samples > 0:
        print(f"=== RUNNING CORPUS BENCHMARK ON {args.benchmark_samples} SAMPLES ===")
        ref_video = Path("uploads/2061004d6841.mp4")
        if not ref_video.exists():
            print(f"Error: reference video {ref_video} not found for benchmark.")
            sys.exit(1)

        import shutil
        bench_dir = output_dir / "bench_tmp"
        bench_dir.mkdir(parents=True, exist_ok=True)
        video_paths = []
        for idx in range(args.benchmark_samples):
            v_path = bench_dir / f"bench_sample_{idx:04d}.mp4"
            if not v_path.exists():
                shutil.copy(str(ref_video), str(v_path))
            video_paths.append(v_path)

        runner = CorpusRunner(bench_dir, output_dir, jsonl_path=args.jsonl, max_workers=args.workers, transcript_model=args.transcript_model)
        t0 = time.perf_counter()
        stats = runner.run_batch(video_paths)
        wall_time = time.perf_counter() - t0

        shutil.rmtree(bench_dir, ignore_errors=True)

        print("\n=== BENCHMARK REPORT ===")
        print(f"Videos processed:          {stats['total_processed']}")
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
