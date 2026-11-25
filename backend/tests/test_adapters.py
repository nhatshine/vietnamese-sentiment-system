"""Kiểm tra chuyển output SVM Phần 1 về chuẩn nội bộ Phần 3."""

import unittest

from backend.src.adapters import normalize_svm_output
from backend.src.config import LABELS
from backend.src.contracts import validate_prediction_output


class SVMOutputAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}

    def test_normalizes_uppercase_labels(self):
        self.assertEqual(normalize_svm_output(self.prediction), {
            "negative": 0.1, "neutral": 0.2, "positive": 0.7,
        })

    def test_output_passes_internal_contract(self):
        self.assertIsNone(validate_prediction_output(
            normalize_svm_output(self.prediction)
        ))

    def test_missing_label(self):
        del self.prediction["Neutral"]
        with self.assertRaisesRegex(ValueError, "Neutral"):
            normalize_svm_output(self.prediction)

    def test_unknown_label(self):
        self.prediction["Mixed"] = 0.0
        with self.assertRaisesRegex(ValueError, "Mixed"):
            normalize_svm_output(self.prediction)

    def test_integer_probabilities_become_floats(self):
        result = normalize_svm_output({"Negative": 0, "Neutral": 0, "Positive": 1})
        self.assertEqual(result, {"negative": 0.0, "neutral": 0.0, "positive": 1.0})
        self.assertTrue(all(type(value) is float for value in result.values()))

    def test_input_is_not_modified(self):
        original = self.prediction.copy()
        normalize_svm_output(self.prediction)
        self.assertEqual(self.prediction, original)

    def test_input_order_does_not_change_label_mapping(self):
        result = normalize_svm_output({"Positive": 0.88, "Negative": 0.03, "Neutral": 0.09})
        self.assertEqual(result, {"negative": 0.03, "neutral": 0.09, "positive": 0.88})
        self.assertEqual(list(result), LABELS)

    def test_non_dictionary_input(self):
        for prediction in (None, [], "Positive"):
            with self.subTest(prediction=prediction):
                with self.assertRaises(TypeError):
                    normalize_svm_output(prediction)

    def test_incorrect_case_and_duplicate_labels_are_rejected(self):
        for prediction in (
            {"negative": 0.1, "Neutral": 0.2, "Positive": 0.7},
            {**self.prediction, "positive": 0.0},
        ):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    normalize_svm_output(prediction)

    def test_invalid_probability_values_are_rejected(self):
        for probability in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(probability=probability):
                self.prediction["Positive"] = probability
                with self.assertRaises(ValueError):
                    normalize_svm_output(self.prediction)

    def test_unconvertible_and_boolean_values_are_rejected(self):
        for probability in (None, "invalid", [], True, False, complex(0.7, 1)):
            with self.subTest(probability=probability):
                self.prediction["Positive"] = probability
                with self.assertRaises(TypeError):
                    normalize_svm_output(self.prediction)

    def test_invalid_total_is_not_silently_renormalized(self):
        self.prediction["Positive"] = 0.6
        with self.assertRaises(ValueError):
            normalize_svm_output(self.prediction)

    def test_rounded_probabilities_are_preserved(self):
        for positive in (0.333333, 0.333335):
            with self.subTest(positive=positive):
                result = normalize_svm_output({
                    "Negative": 0.333333, "Neutral": 0.333333, "Positive": positive,
                })
                self.assertEqual(result, {
                    "negative": 0.333333, "neutral": 0.333333, "positive": positive,
                })


if __name__ == "__main__":
    unittest.main()
