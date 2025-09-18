import json
from pathlib import Path
import time, copy, argparse


import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from glycowork.glycan_data.loader import lib as libr
from tqdm import tqdm
from gifflar.benchmarks import get_dataset
from gifflar.data.modules import DownstreamGDM
from gifflar.utils import get_metrics

def motif_find(s):
    """converts a IUPACcondensed-ish glycan into a list of overlapping, asterisk-separated glycowords"""
    b = s.split('(')
    b = [k.split(')') for k in b]
    b = [item for sublist in b for item in sublist]
    b = [k.strip('[') for k in b]
    b = [k.strip(']') for k in b]
    b = [k.replace('[', '') for k in b]
    b = [k.replace(']', '') for k in b]
    b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
    return b

def process_glycans(glycan_list):
    """converts list of glycans into a list of lists of glycowords"""
    glycan_motifs = [motif_find(k) for k in glycan_list]
    glycan_motifs = [[i.split('*') for i in k] for k in glycan_motifs]
    return glycan_motifs

def character_to_label(character):
    """tokenizes character by indexing passed library"""
    character_label = libr.get(character, 0)
    return character_label

def pad_sequence(seq, max_length, pad_label: int = len(libr)):
  """adds padding as a new label"""
  seq = seq[:max_length]
  seq += [pad_label] *(max_length - len(seq))
  return seq

def string_to_labels(character_string):
    """tokenizes word by indexing characters in passed library"""
    return list(map(character_to_label, [y for sublist in process_glycans([character_string]) for x in sublist for y in x]))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers: int = 2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.encoder = nn.Embedding(input_size, hidden_size, padding_idx=self.num_classes-1)
        self.decoder = nn.Linear(hidden_size, num_classes)
        self.gru = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.logits_fc = nn.Linear(2*hidden_size, num_classes)
    
    def forward(self, glycan):
        input_seq = torch.tensor(string_to_labels(glycan), dtype=torch.long).cuda()
        if len(input_seq) == 0:
            return torch.zeros((self.num_classes, )).cuda()
        embedded = self.bn1(self.encoder(input_seq))
        
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, torch.full((batch_size, ), input_seq.shape[1], dtype=torch.long))
        # print(packed.data.shape, packed.batch_sizes.shape)
        outputs, (h_n, c_n) = self.gru(embedded)
        # outputs, outputs_len = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        logits = self.logits_fc(outputs.mean(dim=0))
        # logits = logits.transpose(0,1).contiguous()
        # logits_flatten = logits.view(-1, self.num_classes)
        
        return logits

def train_model(model, criterion, optimizer, scheduler, metrics, datamodule, num_epochs: int = 25, padding: bool = False):
    """training loop for language models, keeps track of a few metrics"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_perplexity = 100.0
    train_losses, val_losses = [], []
    train_metrics, val_metrics = metrics["train"], metrics["val"]
    train_list, val_list = [], []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        model.train()
        running_loss = []
        for batch in tqdm(datamodule.train_dataloader()):
            preds = torch.stack([model(glycan) for glycan in batch["IUPAC"]])
            labels = batch.y.squeeze().cuda() if hasattr(batch, "y") else batch.y_oh.cuda()
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                preds = torch.softmax(preds, dim=1)
            else:
                labels = labels.float()
            if not isinstance(criterion, nn.CosineEmbeddingLoss):
                loss = criterion(preds, labels)
            else:
                target = torch.ones(preds.shape[0]).cuda()
                loss = criterion(preds, labels, target)
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
                preds = torch.stack([model(glycan) for glycan in batch["IUPAC"]])
                labels = batch.y.squeeze().cuda() if hasattr(batch, "y") else batch.y_oh.cuda()
                
                if not isinstance(criterion, nn.CosineEmbeddingLoss):
                    loss = criterion(preds, labels)
                else:
                    target = torch.ones(preds.shape[0]).cuda()
                    loss = criterion(preds, labels, target)
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

    ## plot loss & perplexity over the course of training 
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.show()
    return model, pd.DataFrame(train_list), pd.DataFrame(val_list)


def main(base: Path, task: str):
    vs = [int(x.stem.split("_")[-1]) for x in filter(lambda x: "version_" in str(x), base.iterdir())]
    if vs is not None and len(vs) > 0:
        version = base / f"version_{max(vs) + 1}"
    else:
        version = base / "version_0"
    version.mkdir(parents=True, exist_ok=True)

    if task == "glycosylation":
        config = {"name": "Glycosylation", "task": "classification", "num_classes": 5}
        metrics = get_metrics("classification", n_outputs=5)
    elif task == "tissue":
        config = {"name": "Tissue", "task": "classification", "num_classes": 20}
        metrics = get_metrics("multilabel", n_outputs=20)
    elif task == "Taxonomy_Kingdom":
        config = {"name": "Taxonomy_Kingdom", "task": "classification", "num_classes": 13}
        metrics = get_metrics("multilabel", n_outputs=13)
    elif task == "spectrum":
        config = {"name": "Spectrum", "task": "spectrum", "num_classes": 2048}
        metrics = get_metrics("spectrum", n_outputs=2048)
    else:
        raise ValueError(f"Unknown task {task}")

    data_config = get_dataset(config, "/scratch/chair_kalinina/s8rojoer/GIFFLAR/data_new_256")
    # data_config = get_dataset(config, "/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data_new_256")
    datamodule = DownstreamGDM(
        root="/scratch/chair_kalinina/s8rojoer/GIFFLAR/data_new_256",
        # root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data_new_256", 
        filename=data_config["filepath"], 
        hash_code="e2301aa9",
        batch_size=64, 
        transform=None,
        pre_transform=None, 
        force_reload=False,
        **data_config,
    )
    with open(version / "config.json", "w") as f:
        data_config["filepath"] = str(data_config["filepath"])
        json.dump(data_config, f, indent=4)
    
    model = RNN(input_size=len(libr) + 1, hidden_size=256, num_classes=data_config["num_classes"])
    print("SweetTalk has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters")
    model.cuda()
    if task == "glycosylation":
        criterion = nn.CrossEntropyLoss()
    elif task == "spectrum":
        criterion = nn.CosineEmbeddingLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model, train_metrics, val_metrics = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        metrics,
        datamodule,
        num_epochs=50,
        padding=False,
    )
    torch.save(model.state_dict(), version / "model.pth")
    pd.concat([pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)], axis=1).to_csv(version / "metrics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["glycosylation", "tissue", "Taxonomy_Kingdom", "spectrum"])
    args = parser.parse_args()
    main(args.base, args.task)
