from pathlib import Path
import pickle
import random
import pandas as pd

from tqdm import tqdm
import numpy as np
from glycowork.glycan_data.loader import glycan_binding as lgi


def classical_data():
    # Use stack to convert to a Series with a MultiIndex
    lgi.index = lgi["target"]
    lgi.drop(columns=["target", "protein"], inplace=True)
    s = lgi.stack()

    glycans = {f"Gly{i:04d}": iupac for i, iupac in enumerate(lgi.columns)}
    glycans.update({iupac: f"Gly{i:04d}" for i, iupac in enumerate(lgi.columns)})

    lectins = {f"Lec{i:04d}": aa_seq for i, aa_seq in enumerate(lgi.index)}
    lectins.update({aa_seq: f"Lec{i:04d}" for i, aa_seq in enumerate(lgi.index)})

    # Convert the Series to a list of triplets (row, col, val)
    data = []
    splits = np.random.choice(["train", "val", "test"], len(s), p=[0.7, 0.2, 0.1])
    for i, ((aa_seq, iupac), val) in tqdm(enumerate(s.items())):
        data.append((lectins[aa_seq], glycans[iupac], val, splits[i]))

    data = random.sample(data, int(len(data) * 0.20))

    with open("lgi_data_20.pkl", "wb") as f:
        pickle.dump((data, lectins, glycans), f)


def contrastive_data(decoy_threshold: float = 0, max_num_decoys: int = 4, norm_percentile: float = 90):
    DECOY = 0
    LIGAND = 1

    lgi.index = lgi["target"]
    lgi.drop(columns=["target", "protein"], inplace=True)
    s = lgi.stack()

    data = {}
    for ((aa_seq, iupac), val) in s.items():
        if aa_seq not in data:
            data[aa_seq] = [{},{}]
        index = int(val > decoy_threshold)
        data[aa_seq][index][iupac] = val

    triplets = []
    for i, (aa_seq, mols) in tqdm(enumerate(data.items())):
        if i == 1000:
            break
        if len(mols[DECOY]) == 0 or len(mols[LIGAND]) == 0:
            continue
        for ligand, zRFU in mols[LIGAND].items():
            decoys = list(mols[DECOY].keys())
            random.shuffle(decoys)
            for decoy in decoys[:max_num_decoys]:
                splits = np.random.choice(["train", "val", "test"], 1, p=[0.7, 0.2, 0.1])
                triplets.append([aa_seq, ligand, zRFU, decoy, mols[DECOY][decoy], str(splits[0])])
    
    # Normalize the zRFU values to [0,1]
    scale_factor = np.percentile([t[2] for t in triplets], norm_percentile) - decoy_threshold
    for t in triplets:
        t[2] = min((t[2] - decoy_threshold) / scale_factor, 1)

    with open("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/contrastive_data_small.pkl", "wb") as f:
        pickle.dump(triplets, f)


def clean_unilectin():
    BASE = Path("results")
    full = pd.read_csv(BASE / "unilectin.tsv", sep="\t")
    clean = pd.read_csv(BASE / "unilectin_glycans.csv")
    mapping = dict(full[["PDB_ID", "seq"]].values)
    clean["seq"] = clean["pdb_id"].map(mapping)
    clean = clean[["supposed_iupac", "seq", "pdb_id"]]
    clean.dropna(inplace=True)
    clean.rename(columns={"supposed_iupac": "IUPAC", "pdb_id": "PDB_ID"}, inplace=True)
    clean["split"] = np.random.choice(["val", "test"], len(clean), p=[2/3, 1/3])
    clean["y"] = 1
    clean.to_csv(BASE / "unilectin_clean.tsv", sep="\t", index=False)


if __name__ == '__main__':
    # classical_data()
    # contrastive_data()
    clean_unilectin()
