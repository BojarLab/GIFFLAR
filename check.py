from pathlib import Path

import yaml
import pandas as pd


SEEDS = [42, 1234, 1337]
MODELS = ["rf", "svm", "xgb", "mlp", "gnngly", "sweetnet", "rgcn", "gifflar"]
DATASETS = ["Glycosylation", "Tissue", "Taxonomy_Kingdom", "Spectrum"]
BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "Gothenburg" / "GIFFLAR" / "logs_final"

hits = {}
for seed in SEEDS:
    hits[seed] = pd.DataFrame(index=DATASETS, columns=MODELS, dtype=int)

for model in MODELS:
    for version in sorted([p for p in (BASE / model).iterdir() if p.is_dir()], key=lambda x: int(x.name.split("_")[1]), reverse=True):
        if not (version / "hparams.yaml").exists() or not (version / "metrics.csv").exists():
            continue
        with open(version / "hparams.yaml", "r") as file:
            hparams = yaml.load(file, Loader=yaml.FullLoader)
        seed = hparams["seed"]
        dataset = hparams["dataset"]["name"]
        metrics = pd.read_csv(version / "metrics.csv")
        if hits[seed].at[dataset, model] == 1:
            continue
        hits[seed].at[dataset, model] = int(len(metrics) == 3 or metrics["epoch"].max() >= 99)
        if "epoch" in metrics.columns and metrics["epoch"].max() < 99:
            print(seed, dataset, model, metrics["epoch"].max())

for seed in SEEDS:
    print(seed)
    print(hits[seed].to_markdown())
