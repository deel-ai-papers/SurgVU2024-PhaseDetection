import argparse
import datetime
import os

import pytorch_lightning
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning import DataModule
from lightning import VideoClassificationModule


torch.set_float32_matmul_precision("medium")

this_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, os.pardir))


def parse_args():
    # === LOAD CONFIG FROM YAML ===
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(parent_directory, "configs/mvit_resampled.yaml"),
        help="Config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        CONFIG = argparse.Namespace(**yaml.safe_load(f))
        print(f"Loaded config from {args.config}")

    # pretty print config
    print("=== CONFIG ===")
    print(yaml.dump(vars(CONFIG), default_flow_style=False))
    return CONFIG


def train(config: argparse.Namespace):
    # === hparams ===
    dataset_name = config.DATA["DATASET_NAME"]
    config_name = config.CONFIG_NAME

    log_name = (
        f"{datetime.datetime.now().strftime('%m%d')}" + f"_{dataset_name}_{config_name}"
    )
    log_path = os.path.join(parent_directory, "logs", dataset_name, log_name)
    print(f"Logging to {log_path}...\n")

    # === TRAIN ===
    # Create the data module and the classification module
    data_module = DataModule(config=config)
    classification_module = VideoClassificationModule(config)

    # Create a ModelCheckpoint callback to save the best model
    monitored_metric = config.TRAIN.get("MONITORED_METRIC", "val_f1")

    checkpoint_callback = ModelCheckpoint(
        filename="best_model",
        monitor=monitored_metric,
        mode="max",
        save_last=True,
        verbose=True,
    )

    # Create a trainer and fit the model
    trainer = pytorch_lightning.Trainer(
        precision="bf16-mixed",
        max_epochs=config.TRAIN["EPOCHS"],
        limit_train_batches=50000,
        val_check_interval=2000,  # don't wait for the end of the epoch to validate
        limit_val_batches=250,  # 250 random validation samples
        accelerator="gpu",
        devices=-1,
        logger=[
            pytorch_lightning.loggers.WandbLogger(
                name=log_name,
                project="surgvu24",
                config=vars(config),
            ),
        ],
        callbacks=[checkpoint_callback],
    )
    trainer.fit(classification_module, data_module)


if __name__ == "__main__":
    config = parse_args()
    train(config)
