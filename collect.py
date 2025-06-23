from pathlib import Path
import sys

import pandas as pd
import yaml
from openpyxl.styles import Font


model_order = {m: i for i, m in enumerate(["rf", "svm", "xgb", "mlp", "gnngly", "sweetnet", "rgcn", "gifflar"])}
appendix_order = {"lp": 1, "rw": 2, "both": 3}


def order_models(name: str) -> tuple:
    parts = name.split("_")
    mo = model_order.get(parts[0], 100)
    if len(parts) == 1:
        return mo,
    parts[1] = appendix_order.get(parts[1], parts[1])
    tmp = [mo]
    for x in parts[1:]:
        try:
            tmp.append(int(x))
        except:
            tmp.append(x)
    # return tuple(tmp)
    # print(name, res)
    return tuple(tmp)


if len(sys.argv) > 3:
    selection = set(sys.argv[3:])
else:
    selection = None

with pd.ExcelWriter(sys.argv[2] + '.xlsx', engine='xlsxwriter') as writer:
    for metric in ["Accuracy", "MatthewsCorrCoef", "AUROC", "Sensitivity"]:
        root = Path(sys.argv[1])
        models = sorted(filter(lambda x: selection is None or x in selection, [str(x).split("/")[-1] for x in root.iterdir() if x.is_dir()]), key=lambda x: order_models(x))
        # print(models)
        datasets = ["Immunogenicity", "Glycosylation", "Tissue", "Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
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
                    if len(col) != 0:
                        if "val/loss" in res.columns:
                            res.dropna(axis="index", subset=["val/loss"], inplace=True)
                            row = res["val/loss"].idxmin()
                            df.at[name, model] = float(res.iloc[row][col].iloc[0])
                        else:
                            df.at[name, model] = float(res[col].dropna().max().iloc[0])
        df.to_excel(writer, sheet_name=metric)

        workbook = writer.book
        worksheet = writer.sheets[metric]
        bold_format = workbook.add_format({"bold": True})
        for r, (_, row) in enumerate(df.iterrows()):
            try:
                max_val = row.values.max()
                for c, val in enumerate(row.values):
                    if val == max_val:
                        worksheet.write(r + 1, c + 1, val, bold_format)
            except:
                pass
