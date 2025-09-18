import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm
from glycowork.glycan_data.loader import lib as libr

from experiments.protein_encoding import ESM
from ext_baselines.sweettalk import RNN
from gifflar.data.modules import LGI_GDM
from gifflar.model.utils import LectinStorage, get_prediction_head


def train_model(model, optimizer, scheduler, datamodule, num_epochs: int = 25, padding: bool = False):
    """training loop for language models, keeps track of a few metrics"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_perplexity = 100.0
    train_losses, val_losses = [], []
    train_list, val_list = [], []
    lectin_embeddings = LectinStorage(
        encoder=ESM(33), 
        lectin_encoder="ESM", 
        le_layer_num=33, 
        # path="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data_new_256",
        path="/scratch/data_new_256"
    )
    head, criterion, metrics = get_prediction_head(1280 + 256, 1, "regression", size="large")
    train_metrics, val_metrics = metrics["train"], metrics["val"]
    print("LGISweetTalk has", sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters() if p.requires_grad), "trainable parameters")
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        model.train()
        running_loss = []
        for batch in tqdm(datamodule.train_dataloader()):
            gly_embed = torch.stack([model(glycan) for glycan in batch["IUPAC"]])
            lec_embed = lectin_embeddings.batch_query(batch["aa_seq"])
            preds = head(torch.cat([gly_embed, lec_embed], dim=-1)).reshape(-1)

            labels = batch.y.squeeze().cuda()
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            train_metrics.update(preds.detach().cpu(), labels.detach().cpu().long())
        train_losses.append(np.mean(running_loss))
        train_list.append({k: v.item() for k, v in train_metrics.compute().items()})
        train_list[-1]["train/loss"] = train_losses[-1]
        train_metrics.reset()
        print('Train Loss: {:.4f}'.format(train_losses[-1]))
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(datamodule.val_dataloader()):
                gly_embed = torch.stack([model(glycan) for glycan in batch["IUPAC"]])
                lec_embed = lectin_embeddings.batch_query(batch["aa_seq"])
                preds = head(torch.cat([gly_embed, lec_embed], dim=-1)).reshape(-1)

                labels = batch.y.squeeze().cuda()
                loss = criterion(preds, labels)

                val_losses.append(loss.item())
                val_metrics.update(preds.detach().cpu(), labels.detach().cpu().long())
        val_losses.append(np.mean(val_losses))
        val_list.append({k: v.item() for k, v in val_metrics.compute().items()})
        val_list[-1]["val/loss"] = val_losses[-1]
        val_metrics.reset()
        print('Validation Loss: {:.4f}'.format(val_losses[-1]))

        scheduler.step()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Perplexity: {:4f}'.format(best_perplexity))
    model.load_state_dict(best_model_wts)

    return model, pd.DataFrame(train_list), pd.DataFrame(val_list)


def main(base: Path, filename: str):
    vs = [int(x.stem.split("_")[-1]) for x in filter(lambda x: "version_" in str(x), base.iterdir())]
    if vs is not None and len(vs) > 0:
        version = base / f"version_{max(vs) + 1}"
    else:
        version = base / "version_0"
    version.mkdir(parents=True, exist_ok=True)

    datamodule = LGI_GDM(
        root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data_new_256", 
        filename=f"/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data_new_256/{filename}",
        # root="/scratch/data_new_256", 
        # filename=f"/scratch/data_new_256/{filename}",
        hash_code="3d2b1204",
        batch_size=64, 
        transform=None,
        pre_transform=None, 
        force_reload=False,
    )
    with open(version / "config.json", "w") as f:
        json.dump({
            "lectin_enconder": "ESM",
            "lectin_layer_num": 33,
            "filename": filename
        }, f, indent=4)
    
    model = RNN(input_size=len(libr) + 1, hidden_size=256, num_classes=256)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model, train_metrics, val_metrics = train_model(
        model, 
        optimizer, 
        scheduler, 
        datamodule,
        num_epochs=1, 
        padding=False
    )
    torch.save(model.state_dict(), version / "model.pth")
    pd.concat([pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)], axis=1).to_csv(version / "metrics.csv", index=False)

if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
