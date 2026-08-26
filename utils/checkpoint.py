from pathlib import Path

import torch


FORMAT_VERSION = 1


def save_trainable_checkpoint(model, checkpoint_path):
    """Save only the trainable XACLE head, excluding frozen M2D-CLAP weights."""
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "projector": model.projector.state_dict(),
        "score_predictor": model.score_predictor.state_dict(),
    }
    torch.save(checkpoint, Path(checkpoint_path))


def load_trainable_checkpoint(model, checkpoint_path, map_location="cpu"):
    """Load a head-only XACLE checkpoint into an initialized model."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid XACLE checkpoint: expected a dictionary")
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported XACLE checkpoint format: {checkpoint.get('format_version')}"
        )
    model.projector.load_state_dict(checkpoint["projector"], strict=True)
    model.score_predictor.load_state_dict(checkpoint["score_predictor"], strict=True)
    return checkpoint
