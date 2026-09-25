## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation Code](#evaluation-code)
- [Result](#result)
- [License](#license)
- [Citation](#citation)
- [Contributors](#contributors)

<h2 id="overview">📖 Overview</h2>

This repository contains the baseline model for automatic evaluation of
text–audio alignment in
[XACLE Challenge 2027](https://xacle.org/2027/). It provides a model
trained to estimate subjective evaluation scores from text–audio pairs.

In this baseline model, M2D-CLAP is used for both the audio encoder and text
encoder. The pretrained encoders are frozen, and score prediction is performed
using a fully connected projector and score predictor applied to the features
extracted from these encoders.

<h2 id="features">✨ Features</h2>

- Automatically evaluates text–audio alignment scores.
- Uses M2D-CLAP for both the audio encoder and text encoder.
- Trains only the fully connected projector and score predictor while keeping
  the pretrained M2D-CLAP encoders frozen.
- Provides a ready-to-use pretrained baseline model through GitHub releases.

<h2 id="requirements">💻 Requirements</h2>

- Python: Tested on 3.12.9
- CUDA: Tested on 13.0
- PyTorch: Tested on 2.13.0
- Python packages (the core dependencies are listed in `requirements.txt`):
  - einops==0.8.2
  - nnAudio==0.3.4
  - numpy==2.5.2
  - pandas==2.3.3
  - scipy==1.18.1
  - SoundFile==0.14.0
  - timm==1.0.28
  - tqdm==4.70.0
  - transformers==4.57.6

<h2 id="installation">⚙️ Installation</h2>

### 1. Clone the repository

```bash
git clone https://github.com/XACLE-Challenge/XACLE_2027_baselinemodel.git
cd XACLE_2027_baselinemodel
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch

Install PyTorch, TorchAudio, and TorchVision according to your environment.
The following versions were used for this baseline:

```bash
pip install torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

For a CPU environment or a different CUDA version, select the corresponding
installation command from the official PyTorch installation guide.

### 4. Download datasets and a pretrained baseline model

- **Datasets**
  - Download the XACLE Challenge 2027 development dataset from the
    [dataset repository](https://github.com/XACLE-Challenge/XACLE_Challenge_2027_development_dataset).
  - The development dataset is inherited from the XACLE Challenge 2026
    dataset.
  - After downloading and extracting the dataset, place the
    `XACLE_Challenge_2027_development_dataset` directory directly under the
    `datasets` directory. The resulting path must be
    `datasets/XACLE_Challenge_2027_development_dataset/`.

- **A pretrained baseline model**
  - Download the pretrained model from the
    [GitHub Releases page](https://github.com/XACLE-Challenge/XACLE_2027_baselinemodel/releases).
  - The released baseline model directory contains:
    - `best_model.pt`: Trained projector and score predictor parameters.
    - `config.json`: Configuration used for training.
  - After downloading the pretrained model, place its directory in `chkpt`.

- **M2D-CLAP checkpoint**
  - From the assets on the official
    [nttcslab/m2d v0.5.0 release](https://github.com/nttcslab/m2d/releases/tag/v0.5.0),
    download the following ZIP file:

    ```text
    m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
    ```

    This baseline uses the standard M2D-CLAP 2025 model. Do not select the
    `m2d_clap_vit_base-80x1001p80x2p16kpBpTI-2025.zip` model with 20 ms
    temporal resolution.
  - Extract the downloaded ZIP file and place the extracted directory under
    `models/m2d/checkpoints/`. The final checkpoint path must be:

    ```text
    models/m2d/checkpoints/
    └── m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/
        └── checkpoint-30.pth
    ```

  - See [`models/m2d/README.md`](models/m2d/README.md) for the verified
    checkpoint SHA-256 value and third-party license information.

- **Regarding the placement of these directories, please refer to the
  [Project Structure](#project-structure).**

<h2 id="project-structure">📂 Project Structure</h2>

```text
XACLE_2027_baselinemodel/
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── inference.py
├── evaluate.py
├── configs/
│   └── train.json
├── chkpt/
│   └── trained_baseline_model/                   # Need to download
├── datasets/
│   ├── xacle_baseline_dataset.py
│   └── XACLE_Challenge_2027_development_dataset/ # Need to download
│       ├── meta_data/
│       │   ├── train_average.csv
│       │   ├── validation_average.csv
│       │   └── test_average.csv
│       └── wav/
│           ├── train/
│           ├── validation/
│           └── test/
├── losses/
│   └── loss_function.py
├── models/
│   ├── xacle_baseline_model.py
│   └── m2d/
│       ├── encoder.py
│       ├── checkpoints/                          # Need to download
│       │   └── m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/
│       │       └── checkpoint-30.pth
│       └── vendor/
└── utils/
```

<h2 id="usage">🚀 Usage</h2>

### For training (when learning from scratch)

```bash
python train.py --config configs/train.json
```

- A directory named `chkpt` is created, and a timestamped subdirectory is
  created for each training run.
- The JSON file containing the resolved training settings is saved as
  `config.json` in the created subdirectory.
- The model with the highest validation SRCC is saved as `best_model.pt`.
- Training logs are displayed in standard output and saved as `log.txt`.
- Because the M2D-CLAP encoders are frozen, `best_model.pt` contains only the
  fully connected projector and score predictor parameters.

### For inference

```bash
python inference.py <checkpoint_directory> <dataset_key>
```

- `<checkpoint_directory>`: Path to the directory containing `best_model.pt`
  and `config.json` (for example, `chkpt/trained_baseline_model`).
- `<dataset_key>`: Specify the dataset used for inference. Enter either
  `validation` or `test`.
- Inference results are saved as
  `inference_result_for_<dataset_key>.csv` in the checkpoint directory.

Examples:

```bash
python inference.py chkpt/trained_baseline_model validation
python inference.py chkpt/trained_baseline_model test
```

<h2 id="evaluation-code">✔ Evaluation Code</h2>

```bash
python evaluate.py \
  <inference_csv_path> \
  <ground_truth_csv_path> \
  <save_results_dir>
```

- `<inference_csv_path>`: Path to the CSV file containing the inference
  results.
- `<ground_truth_csv_path>`: Path to the metadata CSV containing the
  ground-truth `average_score` values.
- `<save_results_dir>`: Directory in which `evaluation_result.csv` is saved.
- The script calculates SRCC, LCC, KTAU, MSE, and the number of evaluated
  samples.

Examples:

```bash
python evaluate.py \
  chkpt/trained_baseline_model/inference_result_for_validation.csv \
  datasets/XACLE_Challenge_2027_development_dataset/meta_data/validation_average.csv \
  chkpt/trained_baseline_model/evaluation/validation

python evaluate.py \
  chkpt/trained_baseline_model/inference_result_for_test.csv \
  datasets/XACLE_Challenge_2027_development_dataset/meta_data/test_average.csv \
  chkpt/trained_baseline_model/evaluation/test
```

<h2 id="result">💯 Results of baseline model</h2>

The model was trained on the 7,500-sample XACLE Challenge 2026 training split.
The best checkpoint was selected using the 3,000-sample validation split and
then evaluated on the 3,000-sample test split.

### Validation data

| Model | SRCC ↑ | LCC ↑ | KTAU ↑ | MSE ↓ |
| :--- | ---: | ---: | ---: | ---: |
| XACLE 2026 Baseline | 0.3844 | 0.3961 | 0.2646 | 4.8361 |
| **XACLE 2027 M2D-CLAP Baseline** | **0.5844** | **0.5995** | **0.4186** | **3.6368** |

### Test data

| Model | SRCC ↑ | LCC ↑ | KTAU ↑ | MSE ↓ |
| :--- | ---: | ---: | ---: | ---: |
| XACLE 2026 Baseline | 0.3345 | 0.3420 | 0.2290 | 4.8110 |
| **XACLE 2027 M2D-CLAP Baseline** | **0.5648** | **0.6142** | **0.3997** | **3.1847** |

<h2 id="license">📄 License</h2>

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software, with or without
modifications, under the conditions of the MIT License. See the
[`LICENSE`](LICENSE) file for the full license text.

The portable M2D runtime includes its upstream license in
`models/m2d/vendor/LICENSE.pdf`.

<h2 id="citation">📚 Citation</h2>

If you use the dataset, please cite the XACLE Challenge 2026 paper:

```bibtex
@INPROCEEDINGS{XACLE2026,
  author={Okamoto, Yuki and Takizawa, Riki and Kishi, Minoru and Kanamori, Yusuke and Tonami, Noriyuki and Nagase, Ryotaro and Takamichi, Shinnosuke and Imoto, Keisuke},
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={XACLE Challenge 2026: The First X-to-Audio Alignment Challenge},
  year={2026},
  pages={21877--21879},
}
```

<h2 id="contributors">🧑‍💻 Contributors</h2>

- Riki Takizawa (Kyoto Sangyo University, Japan)
- Yuki Okamoto (The University of Tokyo, Japan)
