from tqdm import tqdm
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from glycowork.ml.model_training import train_model, SAM
from glycowork.ml.models import prep_model

from gifflar.data.modules import LGI_GDM
from gifflar.data.datasets import GlycanOnDiskDataset
from experiments.protein_encoding import ENCODER_MAP
from gifflar.model.utils import LectinStorage


le = LectinStorage(ENCODER_MAP["ESM"](33), "ESM", 33)

class LGI_OnDiskDataset(GlycanOnDiskDataset):
    @property
    def processed_file_names(self):
        """Return the list of processed file names."""
        return [split + ".db" for split in ["train", "val", "test"]]


def get_ds(dl, split_idx: int):
    ds = LGI_OnDiskDataset(root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/glycowork_full", path_idx=split_idx)
    data = []
    for x in tqdm(dl):
        data.append(Data(
            labels=x["sweetnet_x"],
            y=x["y"],
            edge_index=x["sweetnet_edge_index"],
            aa_seq=x["aa_seq"][0],
        ))
        if len(data) == 100:
            ds.extend(data)
            del data
            data = []
    if len(data) != 0:
        ds.extend(data)
        del data


def collate_lgi(data):
    print(len(data))
    offset = 0
    labels, edges, y, train_idx, batch = [], [], [], [], []
    for i, x in enumerate(data):
        labels.append(x["sweetnet_x"])
        edge_index = x["sweetnet_edge_index"]
        edges.append(torch.stack([
            edge_index[0] + offset,
            edge_index[1] + offset,
        ]))
        offset += len(x["sweetnet_x"])
        y.append(x["y"])
        train_idx.append(a := le.query(x["aa_seq"]))
        if a is None:
            print(x["aa_seq"])
        batch += [i for _ in range(len(x["labels"]))]

    labels = torch.cat(labels, dim=0)
    edges = torch.cat(edges, dim=1)
    y = torch.stack(y)
    train_idx = torch.stack(train_idx)
    batch = torch.tensor(batch)

    return Batch(
        labels=labels.to("cuda"),
        edge_index=edges.to("cuda"),
        y=y.to("cuda"),
        train_idx=train_idx.to("cuda"),
        batch=batch.to("cuda"),
    )


def train():
    datamodule = LGI_GDM(
        root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/lgi_data", filename="/home/rjo21/Desktop/GIFFLAR/lgi_data_full.pkl", hash_code="8b34af2a",
        batch_size=1, transform=None, pre_transform={"GIFFLARTransform": "", "SweetNetTransform": ""}, force_reload=True,
    )
    
    model = prep_model("LectinOracle", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    m, met = train_model(
        model=model,
        dataloaders={"train": torch.utils.data.DataLoader(datamodule.train, batch_size=128, collate_fn=collate_lgi), 
                     "val": torch.utils.data.DataLoader(datamodule.val, batch_size=128, collate_fn=collate_lgi)},
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        return_metrics=True,
        mode="regression",
        num_epochs=100,
        patience=100,
    )
    
    import pickle
    
    with open("lectinoracle_full_model_2.pkl", "wb") as f:
        pickle.dump(m, f)
    with open("lectinoracle_full_metrics_2.pkl", "wb") as f:
        pickle.dump(met, f)
    

def test():
    datamodule = LGI_GDM(
        root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/lgi_data", filename="/home/rjo21/Desktop/GIFFLAR/experiments/results/unilectin.tsv",
        hash_code="8b34af2a", batch_size=1, transform=None, pre_transform={"GIFFLARTransform": "", "SweetNetTransform": ""}#, force_reload=True,
    )
    
    get_ds(datamodule.train_dataloader(), 0)
    get_ds(datamodule.val_dataloader(), 1)
    get_ds(datamodule.test_dataloader(), 2)

    train_set = LGI_OnDiskDataset("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/glycowork_full", path_idx=0)
    val_set = LGI_OnDiskDataset("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/glycowork_full", path_idx=1)


if __name__ == "__main__":
    train()
