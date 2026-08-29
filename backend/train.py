from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_rows(path: Path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:raise ValueError("Training manifest is empty")
    keys=sorted(rows[0]["features"])
    x=np.asarray([[float(row["features"].get(k,0)) for k in keys] for row in rows],dtype=np.float32)
    labels=[row.get("label") for row in rows]
    return rows,keys,x,labels


class Autoencoder(nn.Module):
    def __init__(self,width,latent):
        super().__init__(); hidden=max(latent*2,16)
        self.encoder=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,hidden),nn.GELU(),nn.Linear(hidden,width))
    def forward(self,x):return self.decoder(self.encoder(x))


class Classifier(nn.Module):
    def __init__(self,width,classes):
        super().__init__(); self.network=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,classes))
    def forward(self,x):return self.network(x)


def train(manifest: Path, output: Path, epochs: int, latent: int):
    rows,keys,x,labels=load_rows(manifest)
    mean=x.mean(0); std=np.maximum(x.std(0),1e-6); tensor=torch.from_numpy((x-mean)/std)
    supervised=all(label is not None for label in labels)
    if supervised:
        names=sorted(set(str(label) for label in labels)); mapping={name:i for i,name in enumerate(names)}
        target=torch.tensor([mapping[str(label)] for label in labels]); model=Classifier(x.shape[1],len(names)); loss_fn=nn.CrossEntropyLoss()
    else:
        names=[]; target=tensor; model=Autoencoder(x.shape[1],min(latent,max(2,x.shape[1]//2))); loss_fn=nn.MSELoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    model.train()
    batch_size=min(1024,max(16,len(rows)))
    generator=torch.Generator().manual_seed(7)
    for _ in range(epochs):
        for indices in torch.randperm(len(tensor),generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True); prediction=model(tensor[indices]); loss=loss_fn(prediction,target[indices]); loss.backward(); optimizer.step()
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"feature_keys":keys,"mean":mean,"std":std,"classes":names,"mode":"supervised_classifier" if supervised else "self_supervised_autoencoder","samples":len(rows),"final_loss":float(loss.detach())},output)
    return {"model":str(output),"mode":"supervised_classifier" if supervised else "self_supervised_autoencoder","samples":len(rows),"features":len(keys),"final_loss":round(float(loss.detach()),6)}


def main():
    parser=argparse.ArgumentParser(description="Train on VIRALYST corpus features")
    parser.add_argument("manifest",type=Path); parser.add_argument("--output",type=Path,default=Path("models/viralyst-corpus.pt")); parser.add_argument("--epochs",type=int,default=30); parser.add_argument("--latent",type=int,default=8)
    args=parser.parse_args(); print(json.dumps(train(args.manifest,args.output,args.epochs,args.latent)))


if __name__=="__main__":main()
