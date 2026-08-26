import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from utils.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(4, 2)
        self.score_predictor = nn.Linear(2, 1)


class CheckpointTest(unittest.TestCase):
    def test_head_only_checkpoint_roundtrip(self):
        model = DummyModel()
        expected = model.score_predictor.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            save_trainable_checkpoint(model, path)
            with torch.no_grad():
                model.score_predictor.weight.zero_()
            checkpoint = load_trainable_checkpoint(model, path)

        self.assertEqual(checkpoint["format_version"], 1)
        self.assertTrue(torch.equal(expected, model.score_predictor.weight.detach()))


if __name__ == "__main__":
    unittest.main()
