import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .vendor.portable_m2d import PortableM2D


class M2DCLAPEncoder(nn.Module):
    """Frozen M2D-CLAP audio/text encoder used by the XACLE baseline."""

    def __init__(self, checkpoint_path: str, freeze: bool = True, cache_dir: str = "./hf_cache"):
        super().__init__()
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"M2D-CLAP checkpoint was not found: {checkpoint}. "
                "See models/m2d/README.md for the expected location."
            )

        os.environ.setdefault("HF_HOME", str(Path(cache_dir).resolve()))
        # The 2025 semantic CLAP projector consumes the 768-dimensional patch
        # sequence rather than the frequency-stacked 3840-dimensional feature.
        self.encoder = PortableM2D(str(checkpoint), flat_features=True)
        # The text encoder is created lazily by PortableM2D. Create it before
        # freezing so its parameters are also excluded from optimization.
        self.encoder.get_clap_text_encoder()
        self.frozen = freeze
        if self.frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def encode_audio(self, wavs: torch.Tensor) -> torch.Tensor:
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs = wavs.squeeze(1)
        if wavs.ndim != 2:
            raise ValueError(f"Expected audio shaped (B, T), got {tuple(wavs.shape)}")
        return self._encode(self.encoder.encode_clap_audio, wavs)

    def encode_text(self, captions: Sequence[str]) -> torch.Tensor:
        if not all(isinstance(caption, str) for caption in captions):
            raise TypeError("captions must be a sequence of strings")
        return self._encode(self.encoder.encode_clap_text, list(captions), truncate=True)

    def _encode(self, fn, *args, **kwargs):
        if self.frozen:
            with torch.no_grad():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
