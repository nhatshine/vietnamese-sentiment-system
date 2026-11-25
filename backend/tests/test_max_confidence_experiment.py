"""Fail-closed cache integrity, mapping and ordered sample alignment."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from backend.run_max_confidence_comparison import validate_cache
except ModuleNotFoundError as error:
    if error.name != "backend.run_max_confidence_comparison":
        raise
    validate_cache = None


class ComparisonCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validate_cache, "Verified cache loader is not implemented yet")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "val_probabilities.npz"
        self.fingerprints = {"models/example": "frozen-model"}
        self.data = {"texts": ["a", "b", "c"], "labels": np.array([0, 1, 2]),
                     "sources": ["hieu_sa", "neu_esc", "vsfc"]}
        self.selection = {"selection_dataset": "validation", "test_dataset_used": False,
                          "selected_alpha": .8, "probability_columns": ["negative", "neutral", "positive"],
                          "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
                          "fingerprints": self.fingerprints, "svm_validation_macro_f1": 1.,
                          "transformer_validation_macro_f1": 1., "ensemble_validation_macro_f1": 1.}
        self.arrays = {"svm_probs": np.eye(3), "transformer_probs": np.eye(3),
                       "y_true": self.data["labels"], "sources": np.array(self.data["sources"]),
                       "fingerprint_json": np.array(json.dumps(self.fingerprints))}
        self.write_cache()

    def write_cache(self):
        np.savez_compressed(self.path, **self.arrays)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load(self):
        return validate_cache(self.path, self.data, self.selection, self.fingerprints, self.digest)

    def test_valid_cache_is_reused_without_any_model_calls(self):
        svm, transformer = self.load()
        np.testing.assert_array_equal(svm, np.eye(3))
        np.testing.assert_array_equal(transformer, np.eye(3))

    def test_cache_byte_change_rejected_even_if_shape_and_labels_match(self):
        self.path.write_bytes(self.path.read_bytes() + b"tampering")
        with self.assertRaises(ValueError):
            self.load()

    def test_cache_with_reordered_ground_truth_is_rejected(self):
        self.arrays["y_true"] = np.array([1, 0, 2])
        self.write_cache()
        with self.assertRaises(ValueError):
            self.load()

    def test_cache_with_reordered_sources_is_rejected(self):
        self.arrays["sources"] = np.array(["neu_esc", "hieu_sa", "vsfc"])
        self.write_cache()
        with self.assertRaises(ValueError):
            self.load()

    def test_reordered_label_metadata_is_rejected(self):
        self.selection["probability_columns"] = ["positive", "neutral", "negative"]
        with self.assertRaises(ValueError):
            self.load()

    def test_changed_frozen_model_is_rejected(self):
        self.fingerprints = {"models/example": "changed"}
        with self.assertRaises(ValueError):
            self.load()

    def test_changed_cached_fingerprints_are_rejected(self):
        self.arrays["fingerprint_json"] = np.array(json.dumps({"models/example": "changed"}))
        self.write_cache()
        with self.assertRaises(ValueError):
            self.load()

    def test_other_alpha_or_test_selection_is_rejected(self):
        for key, value in (("selected_alpha", .5), ("selection_dataset", "test"), ("test_dataset_used", True)):
            with self.subTest(key=key):
                old = self.selection[key]
                self.selection[key] = value
                with self.assertRaises(ValueError):
                    self.load()
                self.selection[key] = old

    def test_changed_probability_metrics_fail_baseline_check(self):
        self.arrays["svm_probs"] = np.roll(np.eye(3), 1, axis=0)
        self.write_cache()
        with self.assertRaises(ValueError):
            self.load()

    def test_probability_row_count_must_match_dataset(self):
        self.arrays["svm_probs"] = np.eye(3)[:2]
        self.arrays["transformer_probs"] = np.eye(3)[:2]
        self.write_cache()
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
