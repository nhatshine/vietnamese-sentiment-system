"""Kiểm tra chuẩn đầu ra model bằng thư viện chuẩn unittest."""

import unittest

from backend.src.contracts import get_predicted_label, validate_prediction_output


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}

    def test_valid_prediction(self):
        self.assertIsNone(validate_prediction_output(self.prediction))

    def test_missing_label(self):
        del self.prediction["neutral"]
        with self.assertRaises(ValueError):
            validate_prediction_output(self.prediction)

    def test_probability_above_one(self):
        self.prediction.update(negative=0.0, neutral=0.0, positive=1.1)
        with self.assertRaises(ValueError):
            validate_prediction_output(self.prediction)

    def test_negative_probability(self):
        self.prediction.update(negative=-0.1, neutral=0.2, positive=0.9)
        with self.assertRaises(ValueError):
            validate_prediction_output(self.prediction)

    def test_invalid_probability_sum(self):
        self.prediction["positive"] = 0.6
        with self.assertRaises(ValueError):
            validate_prediction_output(self.prediction)

    def test_get_predicted_label(self):
        self.assertEqual(get_predicted_label(self.prediction), "positive")

    def test_extra_or_incorrect_label(self):
        for prediction in (
            {**self.prediction, "other": 0.0},
            {"negative": 0.1, "Neutral": 0.2, "positive": 0.7},
        ):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    validate_prediction_output(prediction)

    def test_non_dictionary_prediction(self):
        for prediction in (None, "positive", [0.1, 0.2, 0.7]):
            with self.subTest(prediction=prediction):
                with self.assertRaises(TypeError):
                    validate_prediction_output(prediction)

    def test_non_numeric_probability(self):
        for probability in ("0.7", None, [], complex(0.7, 0)):
            with self.subTest(probability=probability):
                self.prediction["positive"] = probability
                with self.assertRaises(TypeError):
                    validate_prediction_output(self.prediction)

    def test_boolean_probability(self):
        for probability in (True, False):
            with self.subTest(probability=probability):
                prediction = {"negative": 0, "neutral": 0, "positive": probability}
                with self.assertRaises(TypeError):
                    validate_prediction_output(prediction)

    def test_non_finite_probability(self):
        for probability in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(probability=probability):
                self.prediction["positive"] = probability
                with self.assertRaises(ValueError):
                    validate_prediction_output(self.prediction)

    def test_integer_probabilities_and_boundary_values(self):
        self.assertIsNone(validate_prediction_output(
            {"negative": 0, "neutral": 1, "positive": 0}
        ))

    def test_small_floating_point_error(self):
        self.prediction["positive"] = 0.7000005
        self.assertIsNone(validate_prediction_output(self.prediction))
        self.prediction["positive"] = 0.70002
        with self.assertRaises(ValueError):
            validate_prediction_output(self.prediction)

    def test_rounded_probability_distribution(self):
        self.assertIsNone(validate_prediction_output({
            "negative": 0.333333, "neutral": 0.333333, "positive": 0.333334,
        }))

    def test_rounded_totals_close_to_one(self):
        for positive in (0.333333, 0.333335):
            with self.subTest(positive=positive):
                self.assertIsNone(validate_prediction_output({
                    "negative": 0.333333, "neutral": 0.333333, "positive": positive,
                }))

    def test_get_predicted_label_validates_input(self):
        with self.assertRaises(ValueError):
            get_predicted_label({"positive": 1.0})

    def test_tie_uses_label_order_independent_of_dictionary_order(self):
        self.assertEqual(get_predicted_label(
            {"positive": 0.5, "neutral": 0.0, "negative": 0.5}
        ), "negative")


if __name__ == "__main__":
    unittest.main()
