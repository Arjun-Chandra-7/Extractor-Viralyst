from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch


def sample_eight(path: Path):
    with av.open(str(path)) as con:
        stream=next(s for s in con.streams if s.type=="video")
        duration=float(con.duration/av.time_base) if con.duration else 0
        frames=[]
        for target in np.linspace(0,max(0,duration-.01),8):
            con.seek(int(target*av.time_base),backward=True,any_frame=False)
            selected=None
            for frame in con.decode(stream):
                selected=frame
                if float(frame.time or 0)>=target:break
            if selected is not None:frames.append(selected.to_ndarray(format="rgb24"))
    if not frames:raise ValueError(f"No frames in {path}")
    while len(frames)<8:frames.append(frames[-1])
    return frames


def embed(manifest: Path, output: Path, model_name: str, batch_size: int):
    """Create transferable video embeddings in GPU micro-batches.

    X-CLIP consumes exactly eight temporally sampled frames. These embeddings can
    feed a task head without repeatedly decoding the original 5K sources.
    """
    from transformers import AutoProcessor, XCLIPModel
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.float16 if device=="cuda" else torch.float32
    processor=AutoProcessor.from_pretrained(model_name)
    model=XCLIPModel.from_pretrained(model_name,torch_dtype=dtype).to(device).eval()
    identifiers=[]; vectors=[]
    with torch.inference_mode():
        for offset in range(0,len(rows),batch_size):
            batch=rows[offset:offset+batch_size]
            videos=[sample_eight(Path(row["path"])) for row in batch]
            inputs=processor(videos=videos,return_tensors="pt")
            pixels=inputs["pixel_values"].to(device=device,dtype=dtype)
            features=model.get_video_features(pixel_values=pixels)
            features=torch.nn.functional.normalize(features,dim=-1).float().cpu().numpy()
            identifiers.extend(row["report_id"] for row in batch); vectors.append(features)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,ids=np.asarray(identifiers),embeddings=np.concatenate(vectors),model=np.asarray(model_name))
    return {"output":str(output),"videos":len(identifiers),"dimensions":int(vectors[0].shape[1]),"device":device,"model":model_name}


def main():
    parser=argparse.ArgumentParser(description="GPU-batched VIRALYST video embeddings")
    parser.add_argument("manifest",type=Path); parser.add_argument("--output",type=Path,default=Path("corpus-reports/xclip-embeddings.npz")); parser.add_argument("--model",default="microsoft/xclip-base-patch32"); parser.add_argument("--batch-size",type=int,default=32)
    args=parser.parse_args(); print(json.dumps(embed(args.manifest,args.output,args.model,args.batch_size)))


if __name__=="__main__":main()
