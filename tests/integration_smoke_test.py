"""One-batch integration test for the frozen M2D-CLAP baseline."""

import json
import os
import sys
import tempfile
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.xacle_baseline_dataset import get_dataset
from losses.loss_function import get_loss_function
from models.xacle_baseline_model import XACLEBaselineModel
from utils.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint
from utils.utils import move_to_device


def write_sine_wave(path: Path, frequency: float, seconds: float, sample_rate: int):
    samples = np.arange(int(seconds * sample_rate), dtype=np.float32)
    signal = 0.1 * np.sin(2 * np.pi * frequency * samples / sample_rate)
    pcm = np.round(signal * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main():
    with open("configs/train.json", encoding="utf-8") as config_file:
        cfg = json.load(config_file)

    device = torch.device("cpu")
    sample_rate = cfg["m2d_clap"]["sample_rate"]

    with tempfile.TemporaryDirectory(prefix="xacle_integration_") as tmp:
        root = Path(tmp)
        wav_dir = root / "wav"
        wav_dir.mkdir()
        write_sine_wave(wav_dir / "sample_1.wav", 220.0, 1.0, sample_rate)
        write_sine_wave(wav_dir / "sample_2.wav", 440.0, 1.0, sample_rate)

        metadata_path = root / "train_average.csv"
        pd.DataFrame(
            [
                {"wav_file_name": "sample_1.wav", "text": "A low tone.", "average_score": 3.0},
                {"wav_file_name": "sample_2.wav", "text": "A high tone.", "average_score": 7.0},
            ]
        ).to_csv(metadata_path, index=False)

        dataset = get_dataset(
            str(metadata_path),
            str(wav_dir),
            max_sec=1,
            sr=sample_rate,
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
        batch = move_to_device(next(iter(loader)), device)

        model = XACLEBaselineModel(cfg).to(device)
        model.train()
        loss_fn = get_loss_function(cfg["loss"])
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=cfg["lr"])

        before = model.score_predictor.linear.weight.detach().clone()
        prediction = model(batch)
        loss = loss_fn(prediction, batch["scores"], batch["num_class"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        checkpoint_path = root / "head.pt"
        saved_weight = model.score_predictor.linear.weight.detach().clone()
        save_trainable_checkpoint(model, checkpoint_path)
        with torch.no_grad():
            model.score_predictor.linear.weight.zero_()
        load_trainable_checkpoint(model, checkpoint_path, map_location=device)

        assert tuple(batch["wavs"].shape) == (2, 1, sample_rate)
        assert tuple(prediction.shape) == (2,)
        assert torch.isfinite(loss)
        assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
        assert not any(parameter.grad is not None for parameter in model.encoder.parameters())
        assert model.score_predictor.linear.weight.grad is not None
        assert not torch.equal(before, model.score_predictor.linear.weight.detach())
        assert torch.equal(saved_weight, model.score_predictor.linear.weight.detach())

        print("batch_wavs_shape:", tuple(batch["wavs"].shape))
        print("batch_captions:", batch["captions"])
        print("prediction_shape:", tuple(prediction.shape))
        print("loss:", float(loss.detach()))
        print("encoder_trainable_parameters:", sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
        print("head_updated: True")
        print("checkpoint_roundtrip: True")
        print("integration_smoke_test: PASS")


if __name__ == "__main__":
    main()
