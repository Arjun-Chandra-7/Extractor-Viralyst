from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from .turbo import analyse_turbo


def _analyse_one(item):
    path, out_dir, label = item
    path=Path(path); rid=hashlib.blake2s(str(path).encode(),digest_size=8).hexdigest()
    report=analyse_turbo(path,rid,path.name)
    if label is not None:report["training_label"]=label
    target=Path(out_dir)/f"{rid}.json"
    target.write_text(json.dumps(report,separators=(",",":")))
    return {"report_id":rid,"path":str(path),"report":str(target),"elapsed":report["processing"]["elapsed_seconds"],"features":report["training_features"],"label":label}


def analyse_directory(source: Path, output: Path, workers: int):
    output.mkdir(parents=True,exist_ok=True)
    extensions={".mp4",".mov",".mkv",".webm",".avi",".m4v"}
    files=[p for p in source.rglob("*") if p.suffix.lower() in extensions]
    manifest=output/"training-manifest.jsonl"
    with manifest.open("a",buffering=1) as sink, cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_analyse_one,((str(p),str(output),None) for p in files),chunksize=4):
            sink.write(json.dumps(result,separators=(",",":"))+"\n")
    return len(files), manifest


def download_manifest(manifest: Path, destination: Path, workers: int):
    destination.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    def fetch(row):
        key=row.get("id") or hashlib.blake2s(row["url"].encode(),digest_size=8).hexdigest()
        suffix=Path(row["url"].split("?",1)[0]).suffix or ".mp4"; target=destination/f"{key}{suffix}"
        if not target.exists():
            request=urllib.request.Request(row["url"],headers={"User-Agent":"VIRALYST/1.0"})
            with urllib.request.urlopen(request,timeout=45) as response, target.open("wb") as sink:
                while chunk:=response.read(1024*1024):sink.write(chunk)
        return target
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch,rows))


def main():
    parser=argparse.ArgumentParser(description="VIRALYST high-throughput corpus runner")
    sub=parser.add_subparsers(dest="command",required=True)
    analyse_cmd=sub.add_parser("analyse"); analyse_cmd.add_argument("source",type=Path); analyse_cmd.add_argument("--output",type=Path,default=Path("corpus-reports")); analyse_cmd.add_argument("--workers",type=int,default=max(1,(os.cpu_count() or 4)-1))
    download_cmd=sub.add_parser("download"); download_cmd.add_argument("manifest",type=Path); download_cmd.add_argument("--destination",type=Path,default=Path("corpus-videos")); download_cmd.add_argument("--workers",type=int,default=32)
    args=parser.parse_args()
    if args.command=="analyse":
        count,manifest=analyse_directory(args.source,args.output,args.workers); print(json.dumps({"processed":count,"training_manifest":str(manifest)}))
    else:
        paths=download_manifest(args.manifest,args.destination,args.workers); print(json.dumps({"downloaded":len(paths),"destination":str(args.destination)}))


if __name__=="__main__":main()
