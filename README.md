# VIRALYST EXTRACTOR

Drop a video into [drop-videos-here](drop-videos-here/). The always-on service processes it, validates an individual JSON report, saves that report in [watched-reports](watched-reports/), and only then deletes the source. Failed inputs are preserved in [failed-videos](failed-videos/).

## Normal use

You do not need to start anything manually. The service is installed as `viralyst-extractor.service` and runs in STANDARD mode.

```bash
systemctl --user status viralyst-extractor.service
journalctl --user -u viralyst-extractor.service -f
```

If it is stopped:

```bash
systemctl --user restart viralyst-extractor.service
```

For persistence after logout/reboot on Linux:

```bash
loginctl enable-linger xor_sensei
```

## What each report contains

Each report is independent and evidence-first. It includes source metadata, verified edit/shot structure, timed spoken transcript, separate OCR/caption tracking, sparse color measurements, audio grading, cross-modal timeline events, confidence/provenance, and training eligibility. It does not promote unverified guesses into training data.

## Corpus processing

For large Core Brain ingestion, use CORPUS_TRAIN rather than the watched folder:

```bash
python extract-corpus.py --input corpus-videos --output corpus-reports --workers 4 --jsonl corpus-reports/corpus.jsonl
```

Benchmark only with a folder of distinct real videos:

```bash
python extract-corpus.py --benchmark-dir benchmark-videos --output corpus-benchmark --workers 4
```

Full extraction throughput excludes duplicates, cached reports, skipped files, and failures.

## For an AI agent

Read [SKILL.md](SKILL.md). It is the authoritative operating procedure and safety contract for managing this repository.
