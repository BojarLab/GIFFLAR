import sys
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from gifflar.pretransforms import get_pretransforms
from gifflar.data.modules import LGI_GDM

def main(filename: str):
    datamodule = LGI_GDM(
        root="/scratch/data_new_256", 
        filename="/scratch/data_new_256/" + filename, 
        hash_code="3d2b1204",
        batch_size=1, 
        transform=None, 
        num_workers=0,
        pre_transform=get_pretransforms("", **{"GIFFLARTransform": "", "GNNGLYTransform": "", "ECFPTransform": "", "SweetNetTransform": ""})
    )

    model = defaultdict(list)

    for batch in tqdm(datamodule.train_dataloader()):
        y = batch.y.item()
        model[batch.IUPAC[0]].append(y)
        model[batch.aa_seq[0]].append(y)
        model["general"].append(y)

    labels, preds = [], []

    for batch in tqdm(datamodule.val_dataloader()):
        iupac = model.get(batch.IUPAC[0], model["general"])
        aa_seq = model.get(batch.aa_seq[0], model["general"])
        labels.append(batch.y.item())
        preds.append(np.mean(iupac + aa_seq))

    print(np.sqrt(np.mean((np.array(labels) - np.array(preds))**2)))


if __name__ == "__main__":
    main(sys.argv[1])
