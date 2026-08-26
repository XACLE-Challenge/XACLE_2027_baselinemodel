import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from datasets.xacle_baseline_dataset import get_dataset
from losses.loss_function import get_loss_function
from models.xacle_baseline_model import XACLEBaselineModel
from utils.checkpoint import save_trainable_checkpoint
from utils.config import load_config
from utils.runtime import seed_everything, seed_worker
from utils.utils import Logger, json_dump, move_to_device


def build_dataloaders(cfg):
    sample_rate = cfg["m2d_clap"]["sample_rate"]
    train_dataset = get_dataset(
        cfg["train_list"],
        os.path.join(cfg["wav_dir"], "train"),
        max_sec=cfg["max_len"],
        sr=sample_rate,
    )
    validation_dataset = get_dataset(
        cfg["validation_list"],
        os.path.join(cfg["wav_dir"], "validation"),
        max_sec=cfg["max_len"],
        sr=sample_rate,
    )

    generator = torch.Generator().manual_seed(cfg["seed"])
    common = {
        "num_workers": cfg["num_workers"],
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": cfg["num_workers"] > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        collate_fn=validation_dataset.collate_fn,
        **common,
    )
    return train_loader, validation_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    batches = 0
    sample_predictions = sample_targets = None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = loss_fn(predictions, batch["scores"], batch["num_class"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        sample_predictions = denormalize_scores(predictions.detach())
        sample_targets = denormalize_scores(batch["scores"].detach())

    if batches == 0:
        raise RuntimeError("Training loader produced no batches")
    return total_loss / batches, sample_predictions, sample_targets


def validate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    batches = 0
    predictions_all = []
    targets_all = []
    sample_predictions = sample_targets = None

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_to_device(batch, device)
            predictions = model(batch)
            loss = loss_fn(predictions, batch["scores"], batch["num_class"])

            denormalized_predictions = denormalize_scores(predictions)
            denormalized_targets = denormalize_scores(batch["scores"])
            total_loss += loss.item()
            batches += 1
            predictions_all.extend(denormalized_predictions)
            targets_all.extend(denormalized_targets)
            sample_predictions = denormalized_predictions
            sample_targets = denormalized_targets

    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    srcc = spearmanr(targets_all, predictions_all).correlation
    mse = float(np.mean((np.asarray(targets_all) - np.asarray(predictions_all)) ** 2))
    return total_loss / batches, srcc, mse, sample_predictions, sample_targets


def denormalize_scores(scores):
    return (scores.detach().cpu().numpy() * 5.0 + 5.0).tolist()


def format_scores(scores):
    return [f"{value:05.2f}" for value in scores]


def build_training_components(cfg, device):
    model = XACLEBaselineModel(cfg).to(device)
    loss_fn = get_loss_function(cfg["loss"])
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("Model has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    return model, loss_fn, optimizer, scheduler


def train(cfg):
    seed_everything(cfg["seed"])
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = os.path.join(cfg["output_dir"], run_name)
    os.makedirs(run_directory, exist_ok=True)
    json_dump(os.path.join(run_directory, "config.json"), cfg)
    sys.stdout = Logger(os.path.join(run_directory, "log.txt"))
    device = torch.device(cfg["device"])

    train_loader, validation_loader = build_dataloaders(cfg)
    model, loss_fn, optimizer, scheduler = build_training_components(cfg, device)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Seed: {cfg['seed']}")
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")
    print(f"Train batches: {len(train_loader)}")

    best_srcc = -np.inf
    patience = 0
    best_model_path = os.path.join(run_directory, "best_model.pt")

    for epoch in range(cfg["epochs"]):
        start_time = time.time()
        train_loss, train_predictions, train_targets = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            cfg.get("max_train_batches"),
        )
        validation_loss, srcc, mse, validation_predictions, validation_targets = validate(
            model,
            validation_loader,
            loss_fn,
            device,
            cfg.get("max_val_batches"),
        )
        scheduler.step(validation_loss)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:04d} completed in {elapsed:.2f} seconds | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f} | "
            f"Val SRCC / MSE: {srcc:.4f}, {mse:.4f}"
        )
        print(f"\ttrain pred: {format_scores(train_predictions)}")
        print(f"\ttrain gt:   {format_scores(train_targets)}")
        print(f"\tval pred:   {format_scores(validation_predictions)}")
        print(f"\tval gt:     {format_scores(validation_targets)}")

        improved = np.isfinite(srcc) and srcc > best_srcc
        if not os.path.isfile(best_model_path) or improved:
            if improved:
                best_srcc = srcc
            patience = 0
            save_trainable_checkpoint(model, best_model_path)
            print("Best model updated")
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print("Early stopping: patience exhausted")
                break

    return run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.json")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
