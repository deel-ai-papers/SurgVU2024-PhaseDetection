from typing import Any

import torch
import torch.nn as nn
from pytorchvideo.models.csn import create_csn
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)
from torch.hub import load_state_dict_from_url

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"
checkpoint_paths = {
    "csn_r101": f"{root_dir}/kinetics/CSN_32x2_R101.pyth",
    "mvit_base_32x3": f"{root_dir}/kinetics/MVIT_B_32x3_f294077834.pyth",
}


def get_model(model_name: str, **kwargs):
    if model_name == "csn_r101":
        return csn_r101(**kwargs)
    elif model_name == "mvit_base_32x3":
        return mvit_base_32x3(**kwargs)
    else:
        raise ValueError(f"Unknown model name {model_name}")


def csn_r101(
    num_classes: int = 7,
    pretrained: bool = True,
    progress: bool = True,
    **kwargs: Any,
) -> torch.nn.Module:
    """
    Build Channel-Separated Convolutional Networks (CSN):
    Video classification with channel-separated convolutional networks.
    Du Tran, Heng Wang, Lorenzo Torresani, Matt Feiszli. ICCV 2019.

    Args:
        num_classes (int): Number of classes for the model. Defaults to 7.
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        dropout_rate (float): Dropout rate for the model. Defaults to 0.5.
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings.
    """
    model = create_csn(
        model_depth=101,
        stem_pool=nn.MaxPool3d,
        head_pool_kernel_size=(4, 7, 7),
        dropout_rate=0.0,
        **kwargs,
    )

    if pretrained:
        path = checkpoint_paths["csn_r101"]
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)

    if num_classes != 400:
        model.blocks[-1].proj = torch.nn.Linear(2048, num_classes, bias=True)

    return model


def mvit_base_32x3(
    num_classes: int = 7,
    pretrained: bool = True,
    progress: bool = True,
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
