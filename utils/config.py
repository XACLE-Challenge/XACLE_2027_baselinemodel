import json
from pathlib import Path

import torch


REQUIRED_TOP_LEVEL_KEYS = {
    "train_list",
    "validation_list",
    "test_list",
    "wav_dir",
    "output_dir",
    "batch_size",
    "val_batch_size",
    "lr",
    "num_workers",
    "epochs",
    "max_len",
    "early_stop_patience",
    "device",
    "seed",
    "loss",
    "m2d_clap",
    "model",
}


def load_config(config_path="configs/train.json"):
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        cfg = json.load(config_file)
    validate_config(cfg)
    cfg["device"] = resolve_device(cfg["device"])
    return cfg


def validate_config(cfg):
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - cfg.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    for key in ("batch_size", "val_batch_size", "epochs", "max_len"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg["num_workers"], int) or cfg["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if cfg["lr"] <= 0:
        raise ValueError("lr must be positive")
    if cfg["m2d_clap"].get("sample_rate") != 16000:
        raise ValueError("The selected M2D-CLAP checkpoint requires 16 kHz audio")
    if cfg["m2d_clap"].get("embedding_dim") != 768:
        raise ValueError("The selected M2D-CLAP checkpoint produces 768-dimensional embeddings")
    for key in ("max_train_batches", "max_val_batches", "max_inference_batches"):
        if key in cfg and (not isinstance(cfg[key], int) or cfg[key] <= 0):
            raise ValueError(f"{key} must be a positive integer when specified")


def resolve_device(requested_device):
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested_device} to cpu")
        return "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        print("MPS is unavailable; falling back to cpu")
        return "cpu"
    return requested_device
