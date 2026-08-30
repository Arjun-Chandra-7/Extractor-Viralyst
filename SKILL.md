# VIRALYST EXTRACTOR — Agent Operating Skill

Use this repository as a production watched-folder video extraction system. Your job is to keep it running safely, process incoming videos, inspect reports/errors, and make only requested, tested changes.

## Primary workflow

1. Check the service:

   ```bash
   systemctl --user is-active viralyst-extractor.service
   ```

2. If inactive, inspect logs and restart it:

   ```bash
   journalctl --user -u viralyst-extractor.service -n 100 --no-pager
   systemctl --user restart viralyst-extractor.service
   ```

3. Incoming media belongs in `drop-videos-here/`.

4. Successful reports are written to `watched-reports/`. A source video is deleted only after its report has been atomically written and schema-validated.

5. Failed media is moved to `failed-videos/`; do not delete or overwrite it. Read the matching `FAILED-*.json` report before retrying.

## Modes

- `STANDARD` is the watched-folder default: detailed, trustworthy per-video report.
- `CORPUS_TRAIN` is for bulk dataset extraction through `extract-corpus.py`.
- `TURBO` is bounded triage only. Never treat its candidates as verified cuts, pacing, or training labels.
- `FORENSIC` is an escalation mode; do not claim unavailable forensic engines ran.

## Truth and training safety

Preserve this order:

`measurement -> detection -> verification -> temporal alignment -> interpretation -> training eligibility`

- Keep observed values separate from interpretations.
- Never turn sparse frame changes into cuts without dense verification.
- Never call an edit beat-driven without a verified beat grid.
- Caption text must identify whether it is `observed_ocr` or `transcript_assisted`.
- Caption alignments with temporal error above 0.5 seconds are not training-eligible.
- Unknown/deferred is valid. Do not invent values, confidence, GPU use, model output, benchmark performance, or forensic results.

## Validation and changes

- Before changing code, inspect the existing implementation and preserve working behavior.
- Use `apply_patch` for repository edits.
- Run the relevant tests, at minimum:

  ```bash
  python3 -m unittest discover -s tests -v
  ```

- Check `git diff --check` before committing.
- Do not delete user videos, reports, models, corpus data, or Git history unless explicitly instructed.
- Do not count duplicates, cache hits, skips, or failures as full-extraction benchmark throughput.
- Never claim RTX/CUDA/NVDEC acceleration unless `processing.runtime` proves it on the active host.

## Corpus commands

```bash
python extract-corpus.py --input corpus-videos --output corpus-reports --workers 4 --jsonl corpus-reports/corpus.jsonl
python extract-corpus.py --benchmark-dir benchmark-videos --output corpus-benchmark --workers 4
```

Benchmark directories must contain distinct real videos. Report full extractions, duplicates, cache hits, failures, actual wall time, videos/minute, and projected runtime separately.

## Service installation

The installed service definition is `deploy/viralyst-extractor.service`. To reinstall after changing it:

```bash
./install-watcher-service.sh
```

It must remain active after installation. Verify with `systemctl --user is-active viralyst-extractor.service`.
