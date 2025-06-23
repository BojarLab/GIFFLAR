import sys

from pathlib import Path

from glycowork.glycan_data.loader import df_species
from glycowork.motif.graph import glycan_to_nxGraph
import pandas as pd
import torch

from gifflar.data.utils import GlycanStorage

BONDS = {
    "alpha_bond": "C[C@H](OC)CC",
    "beta_bond": "C[C@@H](OC)CC",
    "nostereo_bond": "CC(OC)CC"
}

gs = GlycanStorage("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data_tissue")

def parse_mono(filepath, iupac):
    mono = dict()
    bonds = set()

    monos_text = ""
    bonds_text = ""

    g = glycan_to_nxGraph(iupac)

    for n in g.nodes:
        node = g.nodes[n]
        if n % 2 == 0:  # monosaccharide
            r = gs.query(node["string_labels"])
            if r is None:  # return from processing function
                return None
            if "," in node["string_labels"]:
                m = f"\"{node['string_labels']}\""
            else:
                m = node["string_labels"]
            mono[m] = r["smiles"]
            monos_text += f"\n{n // 2 + 1} {m}"
        else:
            if "a" in node["string_labels"]:  # alpha_bond
                bond_type = "alpha_bond"
            elif "b" in node["string_labels"]:
                bond_type = "beta_bond"
            else:
                bond_type = "nostereo_bond"
            bonds.add(bond_type)
            N = list(g.neighbors(n))
            bonds_text += f"\n{min(N) // 2 + 1} {max(N) // 2 + 1} {bond_type}"

    with open(filepath, "w") as f:
        print("SMILES", file=f)
        for iupac, smiles in mono.items():
            print(iupac, smiles, file=f)
        for bond in bonds:
            print(bond, BONDS[bond], file=f)
        print("\nMONOMERS", end="", file=f)
        print(monos_text, file=f)
        print("\nBONDS", end="", file=f)
        print(bonds_text, file=f)
    return mono


def parse_level():
    root = Path(f"tissue")
    root.mkdir(exist_ok=True)
    graphs = root / "graphs"
    graphs.mkdir(exist_ok=True)

    df = pd.read_csv("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/data/tissue.tsv", sep="\t")
    df["ID"] = list(range(len(df)))
    monos = dict()
    mask = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\rParsing {i}", end="")
        output = parse_mono(graphs / f"{row['ID']}_graph.txt", row["IUPAC"])
        if output is None:
            mask.append(False)
            continue
        monos.update(output)
        mask.append(True)

    df = df[mask]
    labels = [x for x in df.columns if x not in ["ID", "IUPAC", "split"]]
    df["label"] = df[labels].apply(lambda row: ','.join(row.index[row == 1]), axis=1)
    df[["ID", "IUPAC", "label", "split"]].to_csv(root / "multilabel.txt", index=False)

    with open(root / "bonds.txt", "w") as f:
        print("Molecule,SMILES", file=f)
        for bond, smiles in BONDS.items():
            print(bond, smiles, file=f, sep=",")

    with open(root / "monos.txt", "w") as f:
        print("Molecule,SMILES", file=f)
        for mono, smiles in monos.items():
            print(mono, smiles, file=f, sep=",")


if __name__ == '__main__':
    parse_level()
