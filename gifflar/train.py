import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from typing import Any
import time

import torch
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, Logger
from torch_geometric import seed_everything

from gifflar.data.modules import DownstreamGDM, PretrainGDM
from gifflar.model.baselines.gnngly import GNNGLY
from gifflar.model.baselines.mlp import MLP
from gifflar.model.baselines.rgcn import RGCN
from gifflar.model.baselines.sweetnet import SweetNetLightning
from gifflar.benchmarks import get_dataset
from gifflar.model.downstream import DownstreamGGIN
from gifflar.model.glylm import GlycanLM
from gifflar.model.pretrain import PretrainGGIN
from gifflar.pretransforms import get_pretransforms
from gifflar.transforms import get_transforms
from gifflar.utils import get_sl_model, get_metrics, read_yaml_config, hash_dict, unfold_config

torch.multiprocessing.set_sharing_strategy('file_system')

MODELS = {
    "gifflar": DownstreamGGIN,
    "gnngly": GNNGLY,
    "mlp": MLP,
    "rgcn": RGCN,
    "sweetnet": SweetNetLightning,
    "glylm": GlycanLM,
}


def setup(count: int = 4, **kwargs: Any) -> tuple[dict, DownstreamGDM, Logger | None, dict | None]:
    """
    Set up the training environment.

    Params:
        count: The number of outputs needed.
        kwargs: The configuration for the training.

    Returns:
        data_config: The configuration for the data.
        datamodule: The datamodule for the training.
        logger: The logger for the training.
        metrics: The metrics for the training
    """
    seed_everything(kwargs["seed"])

    # set up the data module
    data_config = get_dataset(kwargs["dataset"], kwargs["root_dir"])
    datamodule = DownstreamGDM(
        root=kwargs["root_dir"], filename=data_config["filepath"], hash_code=kwargs["hash"],
        batch_size=kwargs["model"].get("batch_size", 1), transform=None,
        pre_transform=get_pretransforms(data_config["name"], **(kwargs.get("pre-transforms", None) or {})), **data_config,
    )
    data_config["num_classes"] = datamodule.train.dataset_args["num_classes"]
    kwargs["dataset"]["filepath"] = str(data_config["filepath"])

    if count == 2:
        return data_config, datamodule, None, None

    # set up the logger
    logger = CSVLogger(kwargs["logs_dir"], name=kwargs["model"]["name"] + (kwargs["model"].get("suffix", None) or ""))
    logger.log_hyperparams(kwargs)

    if count == 3:
        return data_config, datamodule, logger, None

    # set up the metrics
    metrics = get_metrics(data_config["task"], data_config["num_classes"])

    return data_config, datamodule, logger, metrics


def fit(**kwargs: Any) -> None:
    """
    Fit a statistical learning model.

    Params:
        kwargs: The configuration for the training.
    """
    data_config, datamodule, logger, metrics = setup(**kwargs)

    # initialize the model and extract the data
    model = get_sl_model(kwargs["model"]["name"], data_config["task"], data_config["num_classes"], **kwargs)
    train_X, train_y, train_yoh = datamodule.train.to_statistical_learning()

    # fit the model
    start = time.time()
    model.fit(train_X, train_yoh if data_config["task"] == "multilabel" else train_y)
    print("Training took", time.time() - start, "s")

    # evaluate the model on all splits
    for X, y, yoh, name in [
        (train_X, train_y, train_yoh, "train"),
        (*datamodule.val.to_statistical_learning(), "val"),
        (*datamodule.test.to_statistical_learning(), "test"),
    ]:
        labels = torch.tensor(
            yoh if data_config["task"] == "multilabel" else y,
            dtype=torch.long if data_config["task"] not in {"regression", "spectrum"} else torch.float
        )

        preds = torch.tensor(model.predict_proba(X) if data_config["task"] in {"classification", "multilabel"} else
                             model.predict(X), dtype=torch.float)

        if data_config["task"] == "classification":
            if data_config["num_classes"] > 1:
                labels = labels[:, 0]
                if kwargs["model"]["name"] == "xgb":
                    preds = preds[0]
            else:
                preds = preds[:, 1]
                labels = labels.reshape(-1)
        elif data_config["task"] == "multilabel" and len(preds.shape) == 3:
            preds = preds[:, :, 1].T
        elif data_config["num_classes"] == 1:
            preds = preds.reshape(labels.shape)
        elif data_config["task"] != "spectrum":
            preds = preds.reshape(-1)
            labels = labels.reshape(-1)

        metrics[name].update(preds, labels)
        logger.log_metrics(metrics[name].compute())
    logger.save()


def train(**kwargs: Any) -> None:
    """
    Train a deep learning model.

    Params:
        kwargs: The configuration for the training.
    """
    data_config, datamodule, logger, _ = setup(3, **kwargs)
    model = MODELS[kwargs["model"]["name"]](output_dim=data_config["num_classes"], task=data_config["task"],
                                            pre_transform_args=kwargs.get("pre-transforms", {}), **kwargs["model"])
    trainer = Trainer(
        callbacks=[
            RichModelSummary(),
            RichProgressBar(),
        ],
        max_epochs=kwargs["model"]["epochs"],
        logger=logger,
        limit_train_batches=3,
        limit_val_batches=3,
        # accelerator="cpu",
    )
    start = time.time()
    trainer.fit(model, datamodule)
    print("Training took", time.time() - start, "s")


def pretrain(**kwargs: Any) -> None:
    """
    Pretrain a deep learning model.

    Params:
        kwargs: The configuration for the training.
    """
    transforms, task_list = get_transforms(kwargs.get("transforms", []))
    datamodule = PretrainGDM(
        file_path=kwargs["file_path"], hash_code=kwargs["hash"], batch_size=kwargs["model"].get("batch_size", 1),
        transform=transforms, pre_transform=get_pretransforms(**(kwargs.get("pre-transforms", None) or {})),
    )
    model = PretrainGGIN(tasks=task_list, pre_transform_args=kwargs["pre-transforms"], **kwargs["model"])

    # set up the logger
    logger = CSVLogger(kwargs["logs_dir"],
                       name=kwargs["model"]["name"] + (kwargs["model"].get("suffix", None) or "") + "_pretrain")
    logger.log_hyperparams(kwargs)

    trainer = Trainer(
        devices=[1],
        callbacks=[
            ModelCheckpoint(save_top_k=-1),
            RichModelSummary(),
            RichProgressBar(),
        ],
        max_epochs=kwargs["model"]["epochs"],
        logger=logger,
    )
    trainer.fit(model, datamodule)


def embed(prep_args: dict[str, str], **kwargs: Any) -> None:
    """
    Embed the data using a pretrained model.

    Params:
        prep_args: The configuration for the pretraining.
            model_name: The name of the model.
            hparams_path: The path to the hyperparameters.
            ckpt_path: The path to the checkpoint.
            pkl_dir: The directory to save the embeddings.
        kwargs: The configuration for the training.
    """
    output_name = (Path(prep_args["save_dir"]) /
                   f"{kwargs['dataset']['name']}_{prep_args['name']}_{hash_dict(prep_args, 8)}")
    if output_name.exists():
        return
    else:
        output_name.mkdir(parents=True)

    with open(prep_args["hparams_path"], "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model = PretrainGGIN(**config["model"], tasks=None, pre_transform_args=kwargs.get("pre-transforms", {}), save_dir=output_name)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(prep_args["ckpt_path"])["state_dict"])
    else:
        model.load_state_dict(torch.load(prep_args["ckpt_path"], map_location=torch.device("cpu"))["state_dict"])
    model.eval()

    data_config, data, _, _ = setup(2, **kwargs)
    trainer = Trainer()
    trainer.predict(model, data.predict_dataloader())


def main(config: str | Path) -> None:
    """Main routine starting (pre-)training and embedding data using a pretrained model."""
    custom_args = read_yaml_config(config)
    custom_args["hash"] = hash_dict(custom_args.get("pre-transforms", {}))
    if "root_dir" in custom_args:
        for args in unfold_config(custom_args):
            print(args)
            if "prepare" in args:
                embed(args["prepare"], **args)
            else:
                if args["model"]["name"] in ["rf", "svm", "xgb"]:
                    fit(**args)
                else:
                    train(**args)
                print("Finished training", args["model"]["name"], "on", args["dataset"]["name"])
    else:
        pretrain(**custom_args)
        print("Finished pretraining GIFFLAR on", custom_args["file_path"])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    main(parser.parse_args().config)
