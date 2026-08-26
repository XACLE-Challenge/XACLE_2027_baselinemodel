import torch
import torch.nn as nn
import torch.nn.functional as F

from .m2d import M2DCLAPEncoder


class XACLEBaselineModel(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        encoder_cfg = cfg["m2d_clap"]
        model_cfg = cfg["model"]

        self.encoder = M2DCLAPEncoder(
            checkpoint_path=encoder_cfg["checkpoint"],
            freeze=encoder_cfg.get("freeze", True),
            cache_dir=encoder_cfg.get("cache_dir", "./hf_cache"),
        )
        self.embedding_dim = encoder_cfg["embedding_dim"]
        self.normalize_embeddings = encoder_cfg.get("normalize_embeddings", True)
        self.projector = Projector(
            input_dim=self.embedding_dim * 2,
            hidden_dim=model_cfg["projector"]["hidden_dim"],
            output_dim=model_cfg["projector"]["output_dim"],
            activation=model_cfg["projector"]["activation"],
            dropout=model_cfg["projector"]["dropout"],
        )
        self.score_predictor = ScorePredictor(
            input_dim=model_cfg["projector"]["output_dim"],
            range_clipping=model_cfg["score_predictor"]["range_clipping"],
        )

    def forward(self, batch: dict):
        audio_emb = self.encoder.encode_audio(batch["wavs"])
        text_emb = self.encoder.encode_text(batch["captions"])
        self._validate_embeddings(audio_emb, text_emb)

        if self.normalize_embeddings:
            audio_emb = F.normalize(audio_emb.float(), dim=-1)
            text_emb = F.normalize(text_emb.float(), dim=-1)

        joint_emb = torch.cat([audio_emb, text_emb], dim=-1)
        return self.score_predictor(self.projector(joint_emb))

    def _validate_embeddings(self, audio_emb, text_emb):
        expected = (audio_emb.size(0), self.embedding_dim)
        if tuple(audio_emb.shape) != expected:
            raise ValueError(
                f"Unexpected M2D-CLAP audio embedding shape: {tuple(audio_emb.shape)}; "
                f"expected {expected}"
            )
        if tuple(text_emb.shape) != expected:
            raise ValueError(
                f"Unexpected M2D-CLAP text embedding shape: {tuple(text_emb.shape)}; "
                f"expected {expected}"
            )

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation="ReLU", dropout=0.3):
        super().__init__()
        activation_layer = getattr(nn, activation)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            activation_layer(),
        )

    def forward(self, x):
        return self.net(x)


class ScorePredictor(nn.Module):
    def __init__(self, input_dim, range_clipping=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.range_clipping = range_clipping

    def forward(self, x):
        score = self.linear(x).squeeze(-1)
        return torch.tanh(score) if self.range_clipping else score
