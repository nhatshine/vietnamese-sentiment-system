"""Kiểm tra phép kết hợp xác suất và interface của Weighted Ensemble."""

import unittest

from backend.src.config import LABELS
from backend.src.contracts import validate_prediction_output
from backend.src.ensemble import weighted_ensemble
from backend.src.mock_models import predict_svm, predict_transformer


class WeightedEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.svm = {"negative": 0.10, "neutral": 0.25, "positive": 0.65}
        self.transformer = {"negative": 0.05, "neutral": 0.10, "positive": 0.85}

    def test_default_equal_weights_match_hand_calculation(self):
        result = weighted_ensemble(self.svm, self.transformer)
        self.assertEqual(set(result), {
            "method", "label", "confidence", "probabilities", "weights",
        })
        self.assertEqual(result["method"], "ensemble")
        self.assertEqual(result["weights"], {"svm": 0.5, "transformer": 0.5})
        for label, expected in (("negative", 0.075), ("neutral", 0.175), ("positive", 0.75)):
            self.assertAlmostEqual(result["probabilities"][label], expected)
        self.assertEqual(result["label"], "positive")
        self.assertAlmostEqual(result["confidence"], 0.75)

    def test_alpha_zero_equals_transformer(self):
        result = weighted_ensemble(self.svm, self.transformer, alpha=0)
        self.assertEqual(result["probabilities"], self.transformer)
        self.assertEqual(result["weights"], {"svm": 0.0, "transformer": 1.0})

    def test_alpha_one_equals_svm(self):
        result = weighted_ensemble(self.svm, self.transformer, alpha=1)
        self.assertEqual(result["probabilities"], self.svm)
        self.assertEqual(result["weights"], {"svm": 1.0, "transformer": 0.0})

    def test_alpha_point_three_matches_required_example(self):
        result = weighted_ensemble(self.svm, self.transformer, alpha=0.3)
        for label, expected in (("negative", 0.065), ("neutral", 0.145), ("positive", 0.790)):
            self.assertAlmostEqual(result["probabilities"][label], expected)
        self.assertEqual(result["label"], "positive")
        self.assertAlmostEqual(result["confidence"], 0.790)
        self.assertEqual(result["weights"], {"svm": 0.3, "transformer": 0.7})

    def test_negative_alpha(self):
        with self.assertRaises(ValueError):
            weighted_ensemble(self.svm, self.transformer, alpha=-0.1)

    def test_alpha_above_one(self):
        with self.assertRaises(ValueError):
            weighted_ensemble(self.svm, self.transformer, alpha=1.1)

    def test_string_alpha(self):
        with self.assertRaises(TypeError):
            weighted_ensemble(self.svm, self.transformer, alpha="0.5")

    def test_boolean_alpha(self):
        for alpha in (True, False):
            with self.subTest(alpha=alpha):
                with self.assertRaises(TypeError):
                    weighted_ensemble(self.svm, self.transformer, alpha=alpha)

    def test_other_invalid_alpha_types(self):
        for alpha in (None, [], {}, complex(0.5, 0)):
            with self.subTest(alpha=alpha):
                with self.assertRaises(TypeError):
                    weighted_ensemble(self.svm, self.transformer, alpha=alpha)

    def test_non_finite_alpha(self):
        for alpha in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    weighted_ensemble(self.svm, self.transformer, alpha=alpha)

    def test_invalid_svm_input_even_when_its_weight_is_zero(self):
        for invalid in (
            None,
            {"negative": 0.1, "positive": 0.9},
            {"negative": -0.1, "neutral": 0.2, "positive": 0.9},
            {"negative": 0.1, "neutral": 0.2, "positive": 0.6},
            {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7},
        ):
            with self.subTest(prediction=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    weighted_ensemble(invalid, self.transformer, alpha=0)

    def test_invalid_transformer_input_even_when_its_weight_is_zero(self):
        for invalid in (
            None,
            {"positive": 1.0},
            {"negative": 0.0, "neutral": 0.0, "positive": 1.1},
            {"negative": 0.1, "neutral": 0.2, "positive": float("nan")},
            {"negative": 0.1, "neutral": 0.2, "positive": 0.6},
        ):
            with self.subTest(prediction=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    weighted_ensemble(self.svm, invalid, alpha=1)

    def test_output_probability_contract_and_confidence(self):
        result = weighted_ensemble(self.svm, self.transformer, alpha=0.3)
        probabilities = result["probabilities"]
        self.assertIsNone(validate_prediction_output(probabilities))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertEqual(list(probabilities), LABELS)
        self.assertEqual(result["confidence"], probabilities[result["label"]])

    def test_inputs_are_not_mutated_or_shared_with_output(self):
        svm_before = self.svm.copy()
        transformer_before = self.transformer.copy()
        result = weighted_ensemble(self.svm, self.transformer, alpha=0.3)
        self.assertEqual(self.svm, svm_before)
        self.assertEqual(self.transformer, transformer_before)
        result["probabilities"]["positive"] = 0.0
        self.assertEqual(self.svm, svm_before)
        self.assertEqual(self.transformer, transformer_before)

    def test_dictionary_order_does_not_change_class_alignment(self):
        result = weighted_ensemble(
            {"positive": 0.65, "negative": 0.10, "neutral": 0.25},
            {"neutral": 0.10, "positive": 0.85, "negative": 0.05},
            alpha=0.3,
        )
        for label, expected in (("negative", 0.065), ("neutral", 0.145), ("positive", 0.790)):
            self.assertAlmostEqual(result["probabilities"][label], expected)

    def test_tied_final_probabilities_follow_existing_label_order(self):
        result = weighted_ensemble(
            {"positive": 1.0, "neutral": 0.0, "negative": 0.0},
            {"negative": 1.0, "neutral": 0.0, "positive": 0.0},
        )
        self.assertEqual(result["label"], "negative")
        self.assertEqual(result["confidence"], 0.5)

    def test_engine_with_existing_mock_models(self):
        for text, expected_label, expected_confidence in (
            ("Sản phẩm này rất tốt", "positive", 0.80),
            ("Tôi rất thất vọng vì sản phẩm quá tệ", "negative", 0.80),
            ("Sản phẩm này bình thường", "neutral", 0.65),
        ):
            with self.subTest(text=text):
                result = weighted_ensemble(predict_svm(text), predict_transformer(text))
                self.assertEqual(result["label"], expected_label)
                self.assertAlmostEqual(result["confidence"], expected_confidence)

    def test_integer_probabilities_remain_supported(self):
        result = weighted_ensemble(
            {"negative": 1, "neutral": 0, "positive": 0},
            {"negative": 0, "neutral": 0, "positive": 1},
        )
        self.assertEqual(result["probabilities"], {
            "negative": 0.5, "neutral": 0.0, "positive": 0.5,
        })

    def test_rounded_input_probabilities_are_not_renormalized(self):
        rounded = {"negative": 0.333333, "neutral": 0.333333, "positive": 0.333333}
        result = weighted_ensemble(rounded, rounded)
        self.assertEqual(result["probabilities"], rounded)
        self.assertIsNone(validate_prediction_output(result["probabilities"]))


if __name__ == "__main__":
    unittest.main()
