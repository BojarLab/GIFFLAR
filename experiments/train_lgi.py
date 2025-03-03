import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from argparse import ArgumentParser
import time
import copy
from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric import seed_everything

from gifflar.data.modules import LGI_GDM, ConstrastiveGDM
from gifflar.model.baselines.sweetnet import SweetNetLightning
from gifflar.model.downstream import DownstreamGGIN
from gifflar.pretransforms import get_pretransforms
from gifflar.utils import read_yaml_config, hash_dict
from experiments.lgi_model import LGI_Model
from experiments.contrastive_model import ContrastLGIModel

GLYCAN_ENCODERS = {
    "gifflar": DownstreamGGIN,
    "sweetnet": SweetNetLightning,
}


def collect_metrics(path: Path):
    this = pd.read_csv(path / "metrics.csv")
    if (path / "resuming.txt").exists():
        with open(path / "resuming.txt") as f:
            parent = f.readlines()[0].strip().split(" ")[-1]
        df = collect_metrics(Path(parent))
        return pd.concat([df, this], ignore_index=True)
    return this


def unfold_config(config: dict):
    if isinstance(config["model"]["glycan_encoder"], dict):
        ges = [config["model"]["glycan_encoder"]]
    else:
        ges = config["model"]["glycan_encoder"]
    del config["model"]["glycan_encoder"]

    if isinstance(config["model"]["lectin_encoder"], dict):
        les = [config["model"]["lectin_encoder"]]
    else:
        les = config["model"]["lectin_encoder"]
    del config["model"]["lectin_encoder"]

    for le in les:
        for ge in ges:
            tmp_config = copy.deepcopy(config)
            tmp_config["model"]["lectin_encoder"] = le
            tmp_config["model"]["glycan_encoder"] = ge
            yield tmp_config


def train(contrastive: bool = False, ckpt_file: Path | None = None, **kwargs):
    kwargs["pre-transforms"] = {"GIFFLARTransform": "", "SweetNetTransform": ""}
    kwargs["hash"] = hash_dict(kwargs["pre-transforms"])
    seed_everything(kwargs["seed"])

    if contrastive:
        GDM = ConstrastiveGDM
        MODEL = ContrastLGIModel
        LOG_SUFFIX = "CLGI_"
    else:
        GDM = LGI_GDM
        MODEL = LGI_Model
        LOG_SUFFIX = "LGI_"

    datamodule = GDM(
        root=kwargs["root_dir"], filename=kwargs["origin"], hash_code=kwargs["hash"],
        batch_size=kwargs["model"].get("batch_size", 1), transform=None, num_workers=12,
        pre_transform=get_pretransforms("", **(kwargs["pre-transforms"] or {})),
    )
    add_validation = []
    add_tasks = []
    for entry in kwargs["add_valid"]:
        add_validation.append(LGI_GDM(
            root=kwargs["root_dir"], filename=entry["path"], hash_code=kwargs["hash"],
            batch_size=kwargs["model"].get("batch_size", 1), transform=None, num_workers=12,
            pre_transform=get_pretransforms("", **(kwargs["pre-transforms"] or {})), 
        ))
        add_tasks.append((entry["name"], entry["task"]))

    # set up the logger
    glycan_model_name = kwargs["model"]["glycan_encoder"]["name"] + (
            kwargs["model"]["glycan_encoder"].get("suffix", None) or "")
    lectin_model_name = kwargs["model"]["lectin_encoder"]["name"] + (
            kwargs["model"]["lectin_encoder"].get("suffix", None) or "")
    logger = CSVLogger(kwargs["logs_dir"], name=LOG_SUFFIX + glycan_model_name + lectin_model_name)
    logger.log_hyperparams(kwargs)

    glycan_encoder = GLYCAN_ENCODERS[kwargs["model"]["glycan_encoder"]["name"]](**kwargs["model"]["glycan_encoder"])
    model = MODEL(
        glycan_encoder,
        kwargs["model"]["lectin_encoder"]["name"],
        kwargs["model"]["lectin_encoder"]["layer_num"],
        add_tasks = add_tasks,
        **kwargs,
    )
    model.to("cuda")

    if ckpt_file is not None:
        with open(Path(logger.log_dir) / "resuming.txt", "w") as f:
            print(f"Resuming from {ckpt_file.parent.parent}", file=f)

    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath=Path(logger.log_dir) / "weights", 
                monitor="val/loss", 
                mode="min", 
                save_last=True, 
                save_top_k=1, 
                save_weights_only=False,
            ),
            RichProgressBar(), 
            RichModelSummary(),
        ],
        logger=logger,
        max_epochs=kwargs["model"]["epochs"],
        accelerator="gpu",
        # limit_train_batches=10,
        # limit_val_batches=10,
    )
    start = time.time()
    trainer.fit(
        model, 
        train_dataloaders=datamodule.train_dataloader(), 
        val_dataloaders=[datamodule.val_dataloader()] + [add_val.val_dataloader() for add_val in add_validation],
        ckpt_path=ckpt_file,
    )
    print("Training took", time.time() - start, "s")
    collect_metrics(Path(logger.log_dir)).to_csv(Path(logger.log_dir) / "comb_metrics.csv", index=False)


def train_contrastive(**kwargs):
    kwargs["pre-transforms"] = {"GIFFLARTransform": "", "SweetNetTransform": ""}
    kwargs["hash"] = hash_dict(kwargs["pre-transforms"])
    seed_everything(kwargs["seed"])

    datamodule = ConstrastiveGDM(
        root=kwargs["root_dir"], filename=kwargs["origin"], hash_code=kwargs["hash"],
        batch_size=kwargs["model"].get("batch_size", 1), transform=None,
        pre_transform=get_pretransforms("", **(kwargs["pre-transforms"] or {})),
    )

    # set up the logger
    glycan_model_name = kwargs["model"]["glycan_encoder"]["name"] + (
            kwargs["model"]["glycan_encoder"].get("suffix", None) or "")
    lectin_model_name = kwargs["model"]["lectin_encoder"]["name"] + (
            kwargs["model"]["lectin_encoder"].get("suffix", None) or "")
    logger = CSVLogger(kwargs["logs_dir"], name="LGI_" + glycan_model_name + lectin_model_name)
    logger.log_hyperparams(kwargs)

    glycan_encoder = GLYCAN_ENCODERS[kwargs["model"]["glycan_encoder"]["name"]](**kwargs["model"]["glycan_encoder"])
    model = ContrastLGIModel(
        glycan_encoder,
        kwargs["model"]["lectin_encoder"]["name"],
        kwargs["model"]["lectin_encoder"]["layer_num"],
        **kwargs,
    )
    model.to("cuda")

    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(dirpath=Path(kwargs["logs_dir"]) / f"LGI_{glycan_model_name}{lectin_model_name}" / "weights", monitor="val/loss"),
            RichProgressBar(), 
            RichModelSummary(),
        ],
        logger=logger,
        max_epochs=kwargs["model"]["epochs"],
        accelerator="gpu",
    )
    start = time.time()
    trainer.fit(model, datamodule)
    print("Training took", time.time() - start, "s")


def main(mode, config):
    if (c := Path(config)).is_file():
        custom_args = read_yaml_config(config)
        for args in unfold_config(custom_args):
            train(contrastive=mode == "contrastive", **args)
    elif c.is_dir():
        if not ((c / "hparams.yaml").exists() and (c / "metrics.csv").exists() and (c / "weights" / "last.ckpt").exists()):
            raise FileNotFoundError("One or multiple of hparams.yaml, metrics.csv, or weights/last.ckpt are missing. No training can be resumed.")
        custom_args = read_yaml_config(c / "hparams.yaml")
        train(contrastive=mode == "contrastive", ckpt_file=c / "weights" / "last.ckpt", **custom_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("mode", type=str, help="Run classical DTI training or contrastive training", choices=["classical", "contrastive"])
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.mode, args.config)
