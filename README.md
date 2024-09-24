# Surgical Video Understanding Challenge 2024 (SurgVU24)

Welcome to the repository for our submission to the **Surgical Video Understanding Challenge 2024 (SurgVU24)**, part of the Endoscopic Vision Challenge at [MICCAI 2024](https://miccai2024.org/) in Marrakesh, Morocco!

This repository contains code and resources to participate in SurgVU24, focusing on developing machine learning models for surgical context detection in endoscopic videos.

## Table of Contents

- [Overview](#overview)
- [Challenge Categories](#challenge-categories)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Machine learning models that can detect and track surgical context from endoscopic video will enable transformational interventions. For example, the ability to automatically categorize surgical progress (i.e., Phase, Step, Task, or Action) and the instruments used will allow for improved assessments of surgical performance, efficiency, and tool choreography, as well as new analyses of operating room resource planning. Indeed, the theoretical and practical applications are broad and far-reaching.

## Challenge Categories

 We took part to the category 2 sub challenge.

### Category 2: Surgical Step Segmentation

In this category, participants will focus on segmenting surgical videos into different steps or phases. Accurate segmentation is crucial for workflow analysis, education, and improving surgical outcomes.

## Repository Structure

```plaintext
.
├── configs # all configuration are here
│   └── mvit_resampled.yaml
├── README.md
├── requirements.txt
└── src # all code is here
    ├── format_surgvu.py
    ├── lightning.py
    ├── surgvu24.py
    ├── train.py
    └── video_models.py
└── docker # file to build the submission container are here
    ├── Dockerfile
    ├── process.py
    ├── model.pth

```

- **configs/**: Configuration files for training models.
- **process.py**: Script for preprocessing data.
- **requirements.txt**: Python dependencies.
- **src/**: Source code directory.
  - **format_surgvu.py**: Data formatting utilities.
  - **lightning.py**: PyTorch Lightning modules.
  - **surgvu24.py**: Dataset definitions.
  - **train.py**: Training script.
  - **video_models.py**: Video model architectures.

## Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- [Other dependencies](#installation)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/surgvu24.git
   cd surgvu24
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

To train a model, use the `train.py` script located in the `src/` directory.

#### Command-Line Arguments

```bash
python src/train.py --help

usage: train.py [-h] [--config CONFIG]

options:
  -h, --help       Show this help message and exit.
  --config CONFIG  Path to the configuration file.
```

#### Example Command

```bash
python src/train.py --config configs/mvit_resampled.yaml
```

This command trains a model using the configuration specified in `configs/mvit_resampled.yaml`.

## Configuration

The configuration files are in YAML format and are located in the `configs/` directory.

### Example Configuration (`configs/mvit_resampled.yaml`)

```yaml
CONFIG_NAME:
  MVIT_PRETRAINED_RESAMPLED
DATA:
  DATASET_NAME: surgvu24
  DATA_PATH: /surgvu24/processed/
  NUM_CLASSES: 8
  SEQUENCE_LENGTH: 32
  SEQUENCE_STRIDE: 1
  USE_WEIGHTED_SAMPLER: True
  CLASS_FREQS:
    [
      0.006464,
      0.044724,
      0.007025,
      0.017800,
      0.028829,
      0.089727,
      0.052081,
      0.753350,
    ]
  PREPROCESSING_PARAMS:
    random_resized_crop_paras:
      scale:
        [0.08, 1.0]
      aspect_ratio:
        [0.75, 1.33]
    aug_paras:
      magnitude: 10
      num_layers: 2
      prob: 0.5
MODEL:
  MODEL_NAME: mvit_base_32x3
  PRETRAINED: True
TRAIN:
  BATCH_SIZE: 5
  EPOCHS: 25
  OPTIMIZER: adam
  BASE_LR: 1.0E-4
  WEIGHT_DECAY: 1.0E-8
  NUM_WORKERS: 8
  MONITORED_METRIC: val_f1
```

#### Configuration Sections

- **CONFIG_NAME**: Name identifier for the configuration.
- **DATA**: Data-related parameters.
  - **DATASET_NAME**: Name of the dataset.
  - **DATA_PATH**: Path to the processed data.
  - **NUM_CLASSES**: Number of classes in the dataset.
  - **SEQUENCE_LENGTH**: Length of input video sequences.
  - **SEQUENCE_STRIDE**: Stride between sequences.
  - **USE_WEIGHTED_SAMPLER**: Whether to use a weighted sampler during training.
  - **CLASS_FREQS**: Class frequencies for weighted sampling.
  - **PREPROCESSING_PARAMS**: Parameters for data augmentation.
- **MODEL**: Model-related parameters.
  - **MODEL_NAME**: Name of the model architecture.
  - **PRETRAINED**: Whether to use a pretrained model.
- **TRAIN**: Training parameters.
  - **BATCH_SIZE**: Training batch size.
  - **EPOCHS**: Number of training epochs.
  - **OPTIMIZER**: Optimization algorithm.
  - **BASE_LR**: Base learning rate.
  - **WEIGHT_DECAY**: Weight decay for regularization.
  - **NUM_WORKERS**: Number of worker threads for data loading.
  - **MONITORED_METRIC**: Metric to monitor during training.

#### Customizing the Configuration

You can create your own configuration file by copying an existing one and modifying the parameters as needed.

```bash
cp configs/mvit_resampled.yaml configs/my_custom_config.yaml
```

Then edit `configs/my_custom_config.yaml` to adjust parameters.

## Contributing

We welcome contributions to improve this repository. Please submit a pull request or open an issue to discuss your proposed changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We extend our gratitude to all participants and organizers of the SurgToolLoc challenges in previous years. Their work has significantly contributed to the advancement of surgical data science.

- [SurgToolLoc 2022](https://surgtoolloc.grand-challenge.org/)
- [SurgToolLoc 2023](https://surgtoolloc23.grand-challenge.org/)
