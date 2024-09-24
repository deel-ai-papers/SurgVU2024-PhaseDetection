import argparse
import os

import pytorch_lightning
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from pytorchvideo.transforms import create_video_transform

from surgvu24 import SequenceWrapper, SurgVu24
from video_models import get_model

this_directory = os.path.abspath(os.path.dirname(__file__))
repo_directory = os.path.abspath(os.path.join(this_directory, os.pardir, os.pardir))


# === DATA (LIGHTNING DATA MODULE) ===
class DataModule(pytorch_lightning.LightningDataModule):
    _DEFAULT_PREPROCESSING_PARAMS = {
        "num_samples": None,
        "convert_to_float": False,
        "video_mean": (0.41757566, 0.26098573, 0.25888634),
        "video_std": (0.21938758, 0.1983, 0.19342837),
        "crop_size": 224,
        "horizontal_flip_prob": 0.5,
        "aug_type": "randaug",
        "aug_paras": {"magnitude": 6, "num_layers": 2, "prob": 0.5},
        "random_resized_crop_paras": {
            # "scale": (0.08, 1.0),
            "scale": (0.25, 1.0),  # maybe 0.08 is a bit too small
            "aspect_ratio": (3.0 / 4.0, 4.0 / 3.0),
        },
    }

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.config = config

    def _get_dataloader(self, split: str):
        """
        Create a dataloader for the given split.

        Args:
            split (str): The split to load. Can be "train", "val", or "test".
        """
        dataset_class = SurgVu24
        # shuffle only for training and validation as we sample 500 random sequences for testing
        # shuffle = True if split in ["train", "val"] else False
        # ValueError: sampler option is mutually exclusive with shuffle
        shuffle = False
        mode = "train" if split == "train" else "val"  # for video_transform
        train_ratio_to_use = self.config.DATA.get("TRAIN_RATIO_TO_USE", 1.0)

        dataset = dataset_class(
            split=split,
            data_dir=self.config.DATA["DATA_PATH"],
            train_ratio_to_use=train_ratio_to_use,
        )
        processing_params = self._DEFAULT_PREPROCESSING_PARAMS
        processing_params.update(self.config.DATA.get("PREPROCESSING_PARAMS", {}))
        seq_transform = create_video_transform(mode, **processing_params)
        use_sampler = self.config.DATA.get("USE_WEIGHTED_SAMPLER", True)
        if split == "train" and use_sampler:
            sampler = torch.utils.data.WeightedRandomSampler(
                dataset.get_sample_weights(config=self.config),
                len(dataset) // 15,
                replacement=True,
            )
        elif split == "val" and use_sampler:
            sampler = torch.utils.data.WeightedRandomSampler(
                dataset.get_sample_weights(config=self.config),
                len(dataset) // 4,
                replacement=True,
            )
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            SequenceWrapper(
                dataset,
                sequence_length=self.config.DATA["SEQUENCE_LENGTH"],
                seq_transform=seq_transform,
                sequence_stride=self.config.DATA["SEQUENCE_STRIDE"],
                temporal_dim=1,
            ),
            batch_size=self.config.TRAIN["BATCH_SIZE"],
            num_workers=self.config.TRAIN["NUM_WORKERS"],
            shuffle=shuffle,
            sampler=sampler,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )

    def train_dataloader(self):
        """
        Return the training dataloader.
        """
        return self._get_dataloader("train")

    def val_dataloader(self):
        """
        Return the validation dataloader.
        """
        return self._get_dataloader("val")

    def test_dataloader(self):
        """
        Return the test dataloader.
        """
        return self._get_dataloader("test")


# === MODEL (LIGHTNING MODULE) ===
class VideoClassificationModule(pytorch_lightning.LightningModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # === HPARAMS ===
        # data
        self.num_classes = config.DATA["NUM_CLASSES"]
        # model
        self.model_name = config.MODEL["MODEL_NAME"]
        self.pretrained = config.MODEL["PRETRAINED"]
        self.init_checkpoint_file = config.MODEL.get("INIT_CHECKPOINT_FILE", None)
        # training
        self.epochs = config.TRAIN["EPOCHS"]
        self.learning_rate = config.TRAIN["BASE_LR"]
        self.weight_decay = config.TRAIN["WEIGHT_DECAY"]
        self.optimizer_name = config.TRAIN.get("OPTIMIZER", "adam")

        # === MODEL ===
        if self.model_name == "csn_r101" or self.model_name == "mvit_base_32x3":
            self.model = get_model(
                num_classes=self.num_classes,
                pretrained=self.pretrained,
                model_name=self.model_name,
            )
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        # === METRICS ===
        # self.automatic_optimization = False
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )
        self.train_detail_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average=None
        )

        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )
        self.val_detail_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average=None
        )
        self.val_jaccard = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )

        self.preds_buffer = None
        self.step_list = [
            "range_of_motion",
            "rectal_artery_vein",
            "retraction_collision_avoidance",
            "skills_application",
            "suspensory_ligaments",
            "suturing",
            "uterine_horn",
            "other",
        ]

        # === LOAD CHECKPOINT ===
        if self.init_checkpoint_file:
            if self.model_name == "csn_r101":
                self.model = VideoClassificationModule.load_from_checkpoint(
                    os.path.join(repo_directory, self.init_checkpoint_file)
                ).model
                print(f"Loaded model from {self.init_checkpoint_file}")
                # adapt the model to the number of classes
                if self.num_classes != self.model.blocks[-1].proj.out_features:
                    self.model.blocks[-1].proj = torch.nn.Linear(
                        2048, self.num_classes, bias=True
                    )
            elif self.model_name == "mvit_base_32x3":
                self.model = VideoClassificationModule.load_from_checkpoint(
                    os.path.join(repo_directory, self.init_checkpoint_file)
                ).model
                print(f"Loaded model from {self.init_checkpoint_file}")
                # adapt the model to the number of classes
                if self.num_classes != self.model.head.in_features:
                    self.model.head.proj = torch.nn.Linear(
                        self.model.head.proj.in_features,
                        self.num_classes,
                        bias=True,
                    )
            else:
                raise ValueError(f"Unknown model name {self.model_name}")

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs = batch[:-1]
        label = batch[-1]

        # Get the model predictions
        y_hat = self(*inputs)  # Note: inputs of shape (B, C, T, H, W)
        label = label.view(-1, label.size(-1))
        y_hat = y_hat.view(-1, y_hat.size(-1))

        # Compute cross entropy loss, loss.backwards will be called behind the
        # scenes by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, label)
        label = label.argmax(dim=-1)
        self.train_acc(y_hat, label)
        self.train_f1(y_hat, label)
        self.train_detail_f1.update(y_hat, label)
        # Log the train loss to Tensorboard
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        for i, value in enumerate(
            self.train_detail_f1.compute().detach().cpu().numpy()
        ):
            self.log(
                f"train_f1_{self.step_list[i].replace(' ', '_').lower()}",
                value,
                on_epoch=True,
                on_step=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        # Get the inputs and labels from the batch
        inputs = batch[:-1]
        label = batch[-1]

        # Get the model predictions
        with torch.no_grad():
            y_hat = self(*inputs)  # Note: inputs of shape (B, C, T, H, W)
        label = label.view(-1, label.size(-1))
        y_hat = y_hat.view(-1, y_hat.size(-1))

        label = label.view(-1, label.size(-1))
        y_hat = y_hat.view(-1, y_hat.size(-1))

        loss = F.cross_entropy(y_hat, label)
        label = label.argmax(dim=-1)
        self.val_acc(y_hat, label)
        self.val_f1(y_hat, label)
        self.val_detail_f1.update(y_hat, label)
        self.val_jaccard(y_hat, label)

        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, sync_dist=True, prog_bar=True)
        for i, value in enumerate(self.val_detail_f1.compute().detach().cpu().numpy()):
            self.log(
                f"val_f1_{self.step_list[i].replace(' ', '_').lower()}",
                value,
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )
        self.log(
            "val_jaccard",
            self.val_jaccard,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr
        scheduler, which is usually useful for training video models.
        """
        # return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        if self.optimizer_name == "adam":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer_name}")

        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.epochs // 3, gamma=0.2
                ),
                # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                #     optimizer, T_max=self.epochs, eta_min=1e-8
                # ),
                "interval": "epoch",
            },
        }
