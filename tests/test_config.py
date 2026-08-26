import copy
import json
import unittest
from pathlib import Path

from utils.config import validate_config


class ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/train.json").read_text())

    def test_training_config_is_valid(self):
        validate_config(self.config)

    def test_missing_required_key_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        del invalid["seed"]
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_config(invalid)

    def test_invalid_batch_limit_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["max_train_batches"] = 0
        with self.assertRaisesRegex(ValueError, "max_train_batches"):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
