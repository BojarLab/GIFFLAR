from pathlib import Path
import sys

import pandas as pd
import numpy as np
import yaml
from openpyxl.styles import Font


model_order = {m: i for i, m in enumerate(["rf", "svm", "xgb", "mlp", "gnngly", "sweetnet", "rgcn", "gifflar"])}
appendix_order = {"lp": 1, "rw": 2, "both": 3}
with pd.ExcelWriter(sys.argv[2] + '.xlsx', engine='xlsxwriter') as writer:
    for metric in ["Accuracy", "MatthewsCorrCoef", "AUROC", "Sensitivity"]:
        root = Path(sys.argv[1])
        models = sorted([str(x).split("/")[-1] for x in root.iterdir() if x.is_dir()], key=lambda x: (model_order.get(x.split("_")[0], 100), appendix_order.get(x.split("_")[-1], 0)))
        datasets = ["Immunogenicity", "Glycosylation", "Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
        df = pd.DataFrame(index=datasets, columns=models, dtype=float)
        df.fillna(-1.0, inplace=True)
        for model in models:
            base = root / model
            for v in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda x: int(x.name.split("_")[1]), reverse=True):
                if not (v / "hparams.yaml").exists() or not (v / "metrics.csv").exists():
                    continue
                with open(v / "hparams.yaml", "r") as file:
                    name = yaml.load(file, Loader=yaml.FullLoader)["dataset"]["name"]
                    if "_" in name:
                        name = name.split("_")[1]
                if df.at[name, model] == -1:
                    res = pd.read_csv(v / "metrics.csv", index_col=0)
                    col = list(filter(lambda x: "val/" in x and metric in x, res.columns))
                    if col != []:
                        df.at[name, model] = float(np.nanmax(res[col].values))
        df.to_excel(writer, sheet_name=metric)

        workbook = writer.book
        worksheet = writer.sheets[metric]
        bold_format = workbook.add_format({"bold": True})
        for r, (_, row) in enumerate(df.iterrows()):
            max_val = row.values.max()
            for c, val in enumerate(row.values):
                if val == max_val:
                    worksheet.write(r + 1, c + 1, val, bold_format)

