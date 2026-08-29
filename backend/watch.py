from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

from .pipeline import analyse
from .turbo import analyse_turbo

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class FolderWatcher:
    def __init__(self, inbox: Path, reports: Path, failed: Path, mode: str, poll_seconds: float):
        self.inbox = inbox.resolve()
        self.reports = reports.resolve()
        self.failed = failed.resolve()
        self.mode = mode
        self.poll_seconds = poll_seconds
        self.running = True
        self.observed: dict[Path, tuple[int, int]] = {}
        for folder in (self.inbox, self.reports, self.failed):
            folder.mkdir(parents=True, exist_ok=True)

    def stop(self, *_):
        self.running = False

    def ready_files(self):
        """Yield files only after size is unchanged for five consecutive polls."""
        current = set()
        for path in self.inbox.iterdir():
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            current.add(path)
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            previous_size, stable_polls = self.observed.get(path, (-1, 0))
            stable_polls = stable_polls + 1 if size == previous_size and size > 0 else 0
            self.observed[path] = (size, stable_polls)
            if stable_polls >= 5:
                yield path
        for missing in set(self.observed) - current:
            self.observed.pop(missing, None)

    def process(self, source: Path):
        # Content-derived IDs make retries idempotent and avoid duplicate reports.
        digest = hashlib.blake2s(digest_size=8)
        with source.open("rb") as media:
            while chunk := media.read(1024 * 1024):
                digest.update(chunk)
        report_id = digest.hexdigest()
        destination = self.reports / f"{report_id}-{_safe_stem(source.stem)}.json"
        temporary = destination.with_suffix(".json.partial")
        engine = analyse if self.mode == "detailed" else analyse_turbo
        print(f"[extracting] {source.name} ({self.mode})", flush=True)
        try:
            report = engine(source, report_id, source.name)
            report["watch_folder"] = {
                "source_deleted_after_report": True,
                "report_path": str(destination),
            }
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            # Validate before publishing the report or deleting the source.
            validated = json.loads(temporary.read_text(encoding="utf-8"))
            if validated.get("report_id") != report_id:
                raise ValueError("Report validation failed")
            temporary.replace(destination)
            source.unlink()
            self.observed.pop(source, None)
            print(f"[complete] report={destination.name}; deleted={source.name}", flush=True)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failed_target = _available_name(self.failed / source.name)
            if source.exists():
                shutil.move(str(source), str(failed_target))
            error_report = self.reports / f"FAILED-{report_id}-{_safe_stem(source.stem)}.json"
            error_report.write_text(json.dumps({
                "status": "failed",
                "source": source.name,
                "source_preserved_at": str(failed_target),
                "error": str(exc),
            }, indent=2), encoding="utf-8")
            self.observed.pop(source, None)
            print(f"[failed] {source.name}: {exc}; preserved={failed_target}", flush=True)

    def run(self):
        print(f"VIRALYST watcher ready\n  drop:    {self.inbox}\n  reports: {self.reports}\n  failed:  {self.failed}\n  mode:    {self.mode}", flush=True)
        while self.running:
            for path in list(self.ready_files()):
                if not self.running:
                    break
                self.process(path)
            time.sleep(self.poll_seconds)
        print("VIRALYST watcher stopped", flush=True)


def _safe_stem(value: str):
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return cleaned.strip("-")[:80] or "video"


def _available_name(path: Path):
    if not path.exists():
        return path
    for number in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate failure filename for {path.name}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Watch a folder, extract each video, then delete successful sources")
    parser.add_argument("--inbox", type=Path, default=root / "drop-videos-here")
    parser.add_argument("--reports", type=Path, default=root / "watched-reports")
    parser.add_argument("--failed", type=Path, default=root / "failed-videos")
    parser.add_argument("--mode", choices=("turbo", "detailed"), default="turbo")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    watcher = FolderWatcher(args.inbox, args.reports, args.failed, args.mode, max(.25, args.poll_seconds))
    signal.signal(signal.SIGINT, watcher.stop)
    signal.signal(signal.SIGTERM, watcher.stop)
    watcher.run()


if __name__ == "__main__":
    main()
