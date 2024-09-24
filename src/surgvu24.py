from typing import Union

import numpy as np
import os
from abc import ABC
from abc import abstractmethod
import argparse
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import re
from pathlib import Path
from typing import Callable
from typing import List

from PIL import Image


def get_paths(root: str, filter_func: Callable[[str], bool] = None) -> List[str]:
    """
    List all files in the `root` given path. Hidden files are ignored, in goal
        to handle potential errors with, for example, .DS_Store files
        or other OS-related files.

    Args:

        root (str): The root path from which we want to list the files or dirs present

        filter_func: (Callable[[str], bool]) : Function handler used to filter path
            if given. `Defaullt = None`

    Returns:

        List[str]: Array of founded paths inside `root`
    """
    if not os.path.exists(root):
        raise ValueError(f"Directory {root} not exists.")

    path_expr = f"{root}/[!.]*"
    video_paths = glob.glob(path_expr)

    if filter_func:
        video_paths = list(filter(filter_func, video_paths))

    video_paths.sort(key=lambda path: int(re.sub(r"\D", "", path)))
    return video_paths


# from pytorchvideo.transforms import MixVideo
def pil_loader(path: str) -> Image:
    """
    Handler for loading image with PIL library

    Args:

        path (str): Path to the image to read

    Returns:

        PIL.Image : The loaded PIL Image
    """
    with open(path, "rb") as file:
        with Image.open(file) as img:
            return img.convert("RGB")


class BaseSurgeryDataset(Dataset, ABC):
    """
    Base class for surgery datasets.

    This class is meant to be inherited by specific surgery datasets like Cholec80 and
    Heichole.

    Args:
        split (str): Split of the dataset. Can be one of "train", "val" , "test" or
            "all".
        transform (callable): Transform to apply on the images.
        data_dir (str): Path to the dataset directory.
        source (str): Source of the images. Defaults to "frames".
        train_ratio_to_use (float): Ratio of the training set to use. Useful to
            subsample the training set for ablation study purposes. Defaults to 1.0.
    """

    def __init__(
        self,
        split: str,
        transform: callable,
        data_dir: str,
        source: str = "frames",
        train_ratio_to_use: float = 1.0,
        **kwargs,
    ):
        self.split = split
        self.data_dir = data_dir
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform
        self.source = source
        self.train_ratio_to_use = train_ratio_to_use

        # Common dataset initialization
        self.dataset_infos = self.get_dataset_infos()
        self._split_dataset()

    @property
    @abstractmethod
    def DATASET_NAME(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def FRAME_RATE(self) -> Union[int, float]:
        """Frame rate of the images extracted from the videos. Can be an integer or a
        float.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def LABELS_DOWNSAMPLE_RATE(self) -> int:
        """Downsampling rate to apply on the labels text files to match the frame rate.
        If the labels are already downsampled, set this to 1.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def CLASSES(self):
        raise NotImplementedError

    @abstractmethod
    def _split_dataset(self):
        raise NotImplementedError()

    def get_dataset_infos(self):
        """
        From the dataset directory, get the paths of the images and the corresponding
        labels. The dataset infos are returned in a dataframe, with the following
        columns:
        - `video_idx`: index of the video
        - `img_path`: path to the image
        - `frame_idx`: index of the frame in the video
        - `label`: label of the image
        - `phase`: phase of the image

        Returns:
            df (pd.DataFrame): Dataset infos in a dataframe.
        """
        # === get file paths and labels ===
        frames_dir = os.path.join(self.data_dir, self.source)
        labels_dir = os.path.join(self.data_dir, "annot")
        frames_paths = get_paths(frames_dir)
        phase_files = get_paths(labels_dir)

        img_paths = []
        labels = []
        # === get image paths and labels ===
        for frames_path, phase_path in zip(frames_paths, phase_files):
            with open(phase_path, encoding="utf-8") as file:
                first_line = True
                current_index = 0

                for phase_line in file:
                    if first_line:
                        first_line = False
                        continue
                    elts = phase_line.split()
                    if len(elts) == 1:
                        # print(f"Error in {phase_path}")
                        # print(phase_line)
                        elts.append("Other")
                    frame_number = int(
                        float(elts[0])
                    )  # in case the frame number is float
                    label = " ".join(elts[1:])  # in case the label has spaces

                    if frame_number % self.LABELS_DOWNSAMPLE_RATE == 0:
                        img_paths.append(
                            os.path.join(frames_path, str(current_index) + ".jpg")
                        )
                        labels.append(self.CLASSES.index(label))
                        current_index += 1

        # === compute dataset infos in a dataframe ===
        df = pd.DataFrame({"img_path": img_paths, "label": labels})
        df["phase"] = df["label"].apply(lambda x: self.CLASSES[x])
        df["video_idx"] = df["img_path"].apply(lambda x: int(x.split("/")[-2]))
        df["frame_idx"] = df["img_path"].apply(
            lambda x: int(x.split("/")[-1].split(".")[0])
        )
        # reorder columns
        df = df[["video_idx", "img_path", "frame_idx", "label", "phase"]]

        return df

    def get_sample_weights(self, config: argparse.Namespace = None):
        class_freqs = None if config is None else config.DATA.get("CLASS_FREQS", None)

        if class_freqs is None:
            label_weights = (
                self.dataset_infos["label"].value_counts(normalize=True).sort_index()
            )
        else:
            label_weights = class_freqs

        self.dataset_infos["sample_weight"] = self.dataset_infos["label"].apply(
            lambda x: 1 / label_weights[x]
        )
        return self.dataset_infos["sample_weight"].values

    def __getitem__(self, idx: int):
        # get raw item
        raw_item = self.get_raw_item(idx)
        img, label = raw_item

        if self.transform is not None:
            img = self.transform(img)

        else:
            return img, label

    def get_raw_item(self, idx: int):
        img_path = self.dataset_infos.iloc[idx]["img_path"]
        img = pil_loader(img_path)
        label = self.dataset_infos.iloc[idx]["label"]

        return img, label

    def __len__(self):
        return len(self.dataset_infos)


class SequenceWrapper(Dataset):
    """
    Wrapper class to convert a BaseSurgeryDataset into a sequence-based dataset.

    Args:
        dataset (BaseSurgeryDataset): The surgery dataset to wrap.
        sequence_length (int): Length of the sequences to generate.
        seq_transform (callable): Transform to apply on the sequences. Defaults to None.
        sequence_stride (int): Stride to apply on the frames to downsample the
            sequences. Defaults to 1.
        temporal_dim (int): Dimension to stack the images in the sequence. Defaults to
            0. Alternatively, can be set to 1 to permute the channels dimension to the
            first dimension. Defaults to 0.
    """

    def __init__(
        self,
        dataset: BaseSurgeryDataset,
        sequence_length: int = 10,
        seq_transform: callable = None,
        sequence_stride: int = 1,
        temporal_dim: int = 0,
    ):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.seq_transform = seq_transform
        self.sequence_stride = sequence_stride

        self.video_bounds = self._get_video_bounds()
        self.temporal_dim = temporal_dim

    def _get_sequence_indexes(self, frame_index: int, sequence_length: int) -> list:
        """
        Args:
            frame_index (int): Index of the frame.
            sequence_length (int): Length of the sequence to generate.
        """
        video_idx = self.dataset.dataset_infos.loc[frame_index]["video_idx"]
        video_start, video_end = self.video_bounds[video_idx]

        # === Sequence indexes ===
        start_index = frame_index - (sequence_length - 1) * self.sequence_stride
        end_index = frame_index + 1
        seq_indexes = np.array(list(range(start_index, end_index)))[
            :: self.sequence_stride
        ]

        # === Padding ===
        # repeat first frame if sequence goes before video start
        invalid_start_indexes = seq_indexes < video_start
        if any(invalid_start_indexes):
            first_valid_index = seq_indexes[~invalid_start_indexes][0]
            seq_indexes[invalid_start_indexes] = first_valid_index

        # repeat last frame if sequence goes after video end
        invalid_end_indexes = seq_indexes > video_end
        if any(invalid_end_indexes):
            last_valid_index = seq_indexes[~invalid_end_indexes][-1]
            seq_indexes[invalid_end_indexes] = last_valid_index

        return seq_indexes

    def _get_video_bounds(self) -> List[tuple]:
        """Get the bounds of all videos in the dataset. This is used to pad the
        sequences when they overlap between videos.

        Returns:
            dict[int, tuple]: Dictionary of video bounds, with video index as key and
                tuple of start and end indexes as value.
        """
        video_bounds = {}
        for video_idx in self.dataset.dataset_infos["video_idx"].unique():
            sub_df = self.dataset.dataset_infos[
                self.dataset.dataset_infos["video_idx"] == video_idx
            ]
            video_bounds[video_idx] = (sub_df.index[0], sub_df.index[-1])
        return video_bounds

    def __getitem__(self, idx: int):
        """
        Get a sequence of images and the corresponding label from the dataset.

        Args:
            idx (int): Index of the main frame of the sequence

        Returns:
            tuple: Tuple containing:
                - img_tensor (torch.Tensor): Tensor of shape (sequence_length, C, H, W)
                - label (int): Label of the main frame from of the sequence
        """
        # === Get sequence indexes ===
        seq_indexes = self._get_sequence_indexes(idx, self.sequence_length)
        assert len(seq_indexes) == self.sequence_length, (
            f"Sequence length is {len(seq_indexes)} but should be "
            f"{self.sequence_length}"
        )

        # === Get label of the main frame ===
        main_frame_idx = seq_indexes[-1]

        main_item = self.dataset.get_raw_item(main_frame_idx)
        label = main_item[1]

        # === Get images ===
        img_list = []
        for index in seq_indexes:
            img = self.dataset.get_raw_item(index)[0]
            if self.dataset.transform is not None:
                img = self.dataset.transform(img)

            img_list.append(img)

        seq_tensor = torch.stack(img_list, dim=self.temporal_dim)

        # # === Apply seq transform ===
        if self.seq_transform is not None:
            seq_tensor = self.seq_transform(seq_tensor)

        label = torch.as_tensor(label).view(1, 1)
        label = (
            torch.nn.functional.one_hot(label, len(self.dataset.CLASSES))
            .view(-1)
            .float()
        )

        output = (seq_tensor, label)

        return output

    def __len__(self):
        return len(self.dataset.dataset_infos)


class SurgVu24(BaseSurgeryDataset):
    """
    SurgVu24 dataset, inheriting from BaseSurgeryDataset.

    Args:
        split (str): Split of the dataset. Can be one of "train", "val", "test" or
            "all".
        transform (callable): Transform to apply on the images.
        data_dir (str): Path to the dataset directory.
        source (str): Source of the images. Can be one of "frames", or "frames_518".
            Defaults to "frames".
    """

    @property
    def DATASET_NAME(self) -> str:
        return "surgvu24"

    @property
    def FRAME_RATE(self) -> Union[int, float]:
        return 1  # Annotations and frames are already sampled at 1fps

    @property
    def LABELS_DOWNSAMPLE_RATE(self) -> int:
        return 1  # No downsampling needed as annotations are already at 1fps

    @property
    def CLASSES(self):
        return [
            "Range of motion",
            "Rectal artery/vein",
            "Retraction and collision avoidance",
            "Skills application",
            "Suspensory ligaments",
            "Suturing",
            "Uterine horn",
            "Other",
        ]

    def __init__(
        self,
        split: str = "train",
        transform: callable = None,
        data_dir: str = "./data/dataset/surgvu24",
        **kwargs,
    ):
        super().__init__(
            split,
            transform,
            data_dir,
            **kwargs,
        )

    def _split_dataset(self):
        """
        Split the dataset into train, val and test sets.
        """
        videos_indexes = self.dataset_infos["video_idx"].unique()
        splits_proportions = [0.9, 0.1, 0.0]

        if self.split == "train":
            train_video_indexes = videos_indexes[
                : int(len(videos_indexes) * splits_proportions[0])
            ]
            if self.train_ratio_to_use < 1:
                # Use only a fraction of the training videos (for ablation studies)
                g = np.random.default_rng(seed=0)
                train_video_indexes = g.choice(
                    train_video_indexes,
                    int(len(train_video_indexes) * self.train_ratio_to_use),
                    replace=True,
                )

            self.dataset_infos = self.dataset_infos[
                self.dataset_infos["video_idx"].isin(train_video_indexes)
            ]
            self.dataset_infos.reset_index(drop=True, inplace=True)

        elif (self.split == "val") or (self.split == "test"):
            self.dataset_infos = self.dataset_infos[
                self.dataset_infos["video_idx"].isin(
                    videos_indexes[
                        int(len(videos_indexes) * splits_proportions[0]) : int(
                            len(videos_indexes) * sum(splits_proportions[:2])
                        )
                    ]
                )
            ]
            self.dataset_infos.reset_index(drop=True, inplace=True)
        # elif self.split == "test":
        #     self.dataset_infos = self.dataset_infos[
        #         self.dataset_infos["video_idx"].isin(
        #             videos_indexes[
        #                 int(len(videos_indexes) * sum(splits_proportions[:2])) :
        #             ]
        #         )
        #     ]
        #     self.dataset_infos.reset_index(drop=True, inplace=True)
        elif self.split == "all":
            pass
        else:
            raise ValueError(f"Split {self.split} not recognized.")
        # # drop 80% of the rows labelled as "Other" to balance the dataset
        # self.dataset_infos = self.dataset_infos[
        #     ~(
        #         (self.dataset_infos["label"] == 7)
        #         & (np.random.rand(len(self.dataset_infos)) < 0.9)
        #     )
        # ]
        # self.dataset_infos.reset_index(drop=True, inplace=True)
