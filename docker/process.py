import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from tqdm import tqdm
from typing import Tuple
from evalutils.exceptions import ValidationError
import random
from typing import Dict
import json
from pytorchvideo.models.csn import create_csn
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from scipy.stats import mode

from typing import Any, Callable
from pytorchvideo.transforms import create_video_transform
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)


####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
# Fix fillna
####
execute_in_docker = True
do_majority_vote_smoothing = True


def majority_vote_smoothing(sequence: np.ndarray, window_width: int = 59) -> np.ndarray:
    """
    Apply majority vote smoothing to a sequence of predictions using a sliding window.

    Args:
        sequence (np.ndarray): The input sequence of predictions (1D array).
        window_width (int): The width of the sliding window (must be odd).

    Returns:
        np.ndarray: The postprocessed sequence with majority vote smoothing applied.

    Raises:
        ValueError: If window_width is not odd.
    """
    if window_width % 2 == 0:
        raise ValueError("window_width must be odd to ensure a central frame.")

    smoothed_sequence = np.zeros_like(sequence)

    for i in range(len(sequence)):
        half_window = window_width // 2
        if i - half_window < 0:
            half_window = i
        if i + half_window + 1 > len(sequence):
            half_window = len(sequence) - i - 1
        window = sequence[i - half_window : i + half_window + 1]
        smoothed_sequence[i] = mode(window).mode

    return smoothed_sequence


def mvit_base_32x3(
    num_classes: int = 7,
    pretrained: bool = True,
    progress: bool = True,
    dropout_rate: float = 0.0,
    **kwargs: Any,
) -> torch.nn.Module:
    """
    Build Mixer-ViT base model with 32x3 settings:
    Mixer-ViT: Multiscale Vision Transformers for Temporal Feature Integration.
    Du Tran, Heng Wang, Lorenzo Torresani, Matt Feiszli. ICCV 2021.

    Args:
        num_classes (int): Number of classes for the model. Defaults to 7.
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        dropout_rate (float): unused
        kwargs: use these to modify any of the other model settings.
    """

    mvit_base_32x3_config = {
        "spatial_size": 224,
        "temporal_size": 32,
        "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
        "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
        "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
        "pool_kv_stride_adaptive": [1, 8, 8],
        "pool_kvq_kernel": [3, 3, 3],
    }
    mvit_base_32x3_config.update(kwargs)
    model = create_multiscale_vision_transformers(
        **mvit_base_32x3_config,
    )
    if pretrained:
        path = checkpoint_paths["mvit_base_32x3"]
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    # model = torch.hub.load(
    #     "facebookresearch/pytorchvideo", model="mvit_base_32x3", pretrained=pretrained
    # )

    if num_classes != 400:
        model.head.proj = torch.nn.Linear(768, num_classes, bias=True)

    return model


def crop_frame(frame, resize):
    frame_shape = frame.shape
    h, w = frame_shape[:2]
    crop_w = int(h * 3.75 / 3)
    crop_h = int(0.92 * h)
    frame = frame[:crop_h, (w - crop_w) // 2 : (w + crop_w) // 2]
    frame = cv2.resize(frame, resize)
    # convert to RGB if the frame is in BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


class VideoLoader:
    def load(self, *, fname):
        path = Path(fname)
        print(path)
        if not path.is_file():
            raise IOError(
                f"Could not load {fname} using {self.__class__.__qualname__}."
            )
            # cap = cv2.VideoCapture(str(fname))
        # return [{"video": cap, "path": fname}]
        return [{"path": fname}]

    # only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )


class SurgVU_classify(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            index_key="input_video",
            file_loaders={"input_video": VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=(
                Path("/output/surgical-step-classification.json")
                if execute_in_docker
                else Path("./output/surgical-step-classification.json")
            ),
            validators=dict(
                input_video=(
                    # UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        ###
        ### TODO: adapt the following part for creating your model and loading weights
        ###

        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.model_device)

        self.model = mvit_base_32x3(num_classes=8, pretrained=False, dropout_rate=0.0)
        self.model.to(self.model_device)
        self.model.load_state_dict(
            torch.load("model.pth", map_location=self.model_device)
        )
        self.model.eval()
        self.batch_size = 1
        self.seq_length = 32

        processing_params = {
            "num_samples": None,
            "convert_to_float": False,
            "video_key": None,
            "video_mean": (0.41757566, 0.26098573, 0.25888634),
            "video_std": (0.21938758, 0.1983, 0.19342837),
            "crop_size": 224,
            "horizontal_flip_prob": 0.5,
            "aug_type": "randaug",
            "aug_paras": {"magnitude": 12, "num_layers": 2, "prob": 0.5},
            "random_resized_crop_paras": {
                # "scale": (0.08, 1.0),
                "scale": (0.25, 1.0),  # maybe 0.08 is a bit too small
                "aspect_ratio": (3.0 / 4.0, 4.0 / 3.0),
            },
        }
        self.seq_transform = create_video_transform("val", **processing_params)

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
        # Comment for docker build
        # Comment for docker built

        print(self.step_list)

    def step_predict_json_sample(self):
        single_output_dict = {"frame_nr": 1, "surgical_step": None}
        return single_output_dict

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case  # VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path)  # video file > load evalutils.py

        # return
        # Write resulting candidates to result.json for this case
        return scored_candidates

    def save(self):
        print("Saving prediction results to " + str(self._output_file))
        with open(str(self._output_file), "w") as f:
            print(
                f"Saving {len(self._case_results[0])} predictions to "
                + str(self._output_file)
            )
            json.dump(self._case_results[0], f)
        print(f"predictions: {self._case_results[0]}")

    def predict(self, fname) -> Dict:
        """
        Inputs:
        fname -> video file path

        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation
        """

        print("Video file to be loaded: " + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        ############################################################################
        if num_frames == 0:
            print("No frames in the video")
            return []
        try:
            print("No. of frames: ", num_frames)
            # generate output json
            all_frames_predicted_outputs = []
            resize = (256, 256)
            fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = 1
            ret_0, frame_0 = cap.read()
            frame_0 = crop_frame(frame_0, resize)
            frame_0 = torch.from_numpy(frame_0).permute(2, 0, 1).float() / 255.0
            # reinint cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset video
            frame_buffer = [
                frame_0 for _ in range(self.seq_length + (self.batch_size - 1))
            ]
            i = 0
            k = 0
            with tqdm(total=num_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    k += 1
                    pbar.update(1)
                    if k % int(fps / target_fps) != 0:
                        continue
                    if ret:
                        # crop the borders and the bottom part of the frame
                        # resize the frame
                        frame = crop_frame(frame, resize)

                        ## TODO: check format of frames (RGB, BGR, etc.) and range (0-255, 0-1) and channel order
                        # roll the buffer
                        frame_buffer.pop(0)
                        frame_buffer.append(
                            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        )

                    if (i % self.batch_size == self.batch_size - 1) or not ret:
                        # if not ret, the batch size might be smaller than self.batch_size
                        batch_size_ = min(self.batch_size, (i % self.batch_size) + 1)
                        batch_frame_buffer = [
                            frame_buffer[b : b + self.seq_length]
                            for b in range(batch_size_)
                        ]

                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                batch = torch.stack(
                                    [
                                        self.seq_transform(torch.stack(buffer, dim=1))
                                        for buffer in batch_frame_buffer
                                    ],
                                    dim=0,
                                )
                                batch = batch.bfloat16().to(self.model_device)
                                preds = self.model(batch)
                                preds = torch.argmax(preds, dim=1)
                                preds = preds.cpu().numpy()

                        ############################################################################

                        # step_detection = self.dummy_step_prediction_model()
                        for b in range(batch_size_):
                            frame_nr = i - batch_size_ + b + 1
                            frame_dict = self.step_predict_json_sample()
                            frame_dict["frame_nr"] = frame_nr
                            frame_dict["surgical_step"] = int(preds[b])
                            all_frames_predicted_outputs.append(frame_dict)
                    if not ret:
                        break
                    i += 1

            if do_majority_vote_smoothing:
                # Extract the surgical steps (predictions)
                # sort the frames by frame number
                predicted_steps = [
                    frame["surgical_step"] for frame in all_frames_predicted_outputs
                ]

                # Apply majority vote smoothing on the predicted steps
                smoothed_steps = majority_vote_smoothing(
                    np.array(predicted_steps), window_width=251
                )

                # Update the predictions with the smoothed values
                for idx, step in enumerate(smoothed_steps):
                    all_frames_predicted_outputs[idx]["surgical_step"] = int(step)

            steps = all_frames_predicted_outputs
            print("steps: ", steps)
            # self._case_results[0] = steps
            return steps
        except Exception as e:
            print(e)
            raise e


if __name__ == "__main__":
    SurgVU_classify().process()
