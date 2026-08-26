# XACLE 2027 Baseline Model

Baseline implementation for automatic text–audio alignment scoring in the
XACLE 2027 Challenge.

The model uses the pretrained M2D-CLAP 2025 audio and text encoders as frozen
feature extractors. Only a fully connected projector and scalar score
predictor are trained.

```text
audio ── M2D-CLAP audio encoder ──┐
                                  ├─ concatenate → projector → score predictor
text  ── M2D-CLAP text encoder  ──┘
```

## Status

- macOS/Apple Silicon CPU smoke test: verified
- Real XACLE 2026 data, one-batch training test: verified
- Short end-to-end train/inference/evaluation test: verified
- ABCI/CUDA full training: pending

## Repository structure

```text
.
├── configs/
│   ├── train.json                 # Full training configuration
│   └── mac_smoke.json             # Short CPU end-to-end test
├── datasets/
│   ├── xacle_baseline_dataset.py
│   └── XACLE_dataset/             # Local dataset; ignored by Git
├── losses/
│   └── loss_function.py
├── models/
│   ├── xacle_baseline_model.py
│   ├── m2d/
│   │   ├── encoder.py
│   │   ├── checkpoints/           # Local M2D weights; ignored by Git
│   │   └── vendor/                # Official PortableM2D runtime and license
├── tests/
│   ├── test_checkpoint.py
│   ├── test_config.py
│   └── integration_smoke_test.py
├── utils/
│   ├── checkpoint.py
│   ├── config.py
│   ├── runtime.py
│   └── utils.py
├── train.py
├── inference.py
└── evaluate.py
```

## Environment setup

Python 3.12.11 was used for the macOS smoke test.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For CUDA systems, install the PyTorch build appropriate for the CUDA version
before installing the remaining requirements.

## M2D-CLAP checkpoint

Download the standard 16 kHz M2D-CLAP 2025 checkpoint from the official
[nttcslab/m2d v0.5.0 release](https://github.com/nttcslab/m2d/releases/tag/v0.5.0).

Expected location:

```text
models/m2d/checkpoints/
└── m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/
    └── checkpoint-30.pth
```

See `models/m2d/README.md` for runtime provenance and the verified checkpoint
SHA-256 value. The checkpoint is not committed to this repository.

## Dataset layout

Place the XACLE dataset as follows:

```text
datasets/XACLE_dataset/
├── meta_data/
│   ├── train_average.csv
│   ├── validation_average.csv
│   └── test_average.csv
└── wav/
    ├── train/
    ├── validation/
    └── test/
```

Each metadata CSV must contain `wav_file_name`, `text`, and, for supervised
splits, `average_score`.

## Usage

### Local integration test

The integration test creates temporary WAV/CSV fixtures and verifies dataset
loading, forward, loss, backward, encoder freezing, and head optimization.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/integration_smoke_test.py
```

The training and inference configurations include a fixed `seed`. DataLoader
shuffling, worker processes, NumPy, Python, and PyTorch are initialized from
that value.

### Short macOS end-to-end test

```bash
python train.py --config configs/mac_smoke.json
```

### Full training

```bash
python train.py --config configs/train.json
```

Each run creates a timestamped directory under the configured `output_dir`
containing:

- `config.json`: resolved run configuration
- `best_model.pt`: projector and score predictor weights only
- `log.txt`: training log

The frozen M2D-CLAP weights are intentionally excluded from the trained
checkpoint, reducing it from approximately 1.6 GB to approximately 7 MB.

### Inference

```bash
python inference.py <checkpoint_directory> validation
python inference.py <checkpoint_directory> test
```

### Evaluation

```bash
python evaluate.py \
  <inference_result.csv> \
  datasets/XACLE_dataset/meta_data/validation_average.csv \
  <output_directory>
```

The evaluation reports SRCC, LCC, Kendall's tau-b, MSE, and sample count.

## Previous baseline

The XACLE 2026 BYOL-A + RoBERTa implementation is not included in the XACLE
2027 public source tree. It remains available from the repository's initial
commit (`8d7e995`) when historical reference is needed.

## Third-party code and licenses

`models/m2d/vendor/portable_m2d.py` is derived from the official
[nttcslab/m2d](https://github.com/nttcslab/m2d) repository. Its upstream
license is included at `models/m2d/vendor/LICENSE.pdf`. See
`models/m2d/README.md` for the pinned upstream commit and local integration
changes.

The rest of this repository is distributed under the terms in `LICENSE`.

## Contributors

- Riki Takizawa — Kyoto Sangyo University
- Yusuke Kanamori — The University of Tokyo
- Yuki Okamoto — The University of Tokyo
