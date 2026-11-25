"""Kiểm tra interface và các luật đơn giản của hai mock model."""

import unittest

from backend.src.config import LABELS
from backend.src.contracts import get_predicted_label, validate_prediction_output
from backend.src.mock_models import predict_svm, predict_transformer


class MockModelTests(unittest.TestCase):
    predictors = (predict_svm, predict_transformer)

    def test_svm_positive_output(self):
        result = predict_svm("Sản phẩm này rất tốt")
        self.assertIsInstance(result, dict)
        self.assertIsNone(validate_prediction_output(result))
        self.assertEqual(get_predicted_label(result), "positive")

    def test_transformer_positive_output(self):
        result = predict_transformer("Sản phẩm này rất tốt")
        self.assertIsInstance(result, dict)
        self.assertIsNone(validate_prediction_output(result))
        self.assertEqual(get_predicted_label(result), "positive")

    def test_negative_output(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                result = predict("Sản phẩm này rất tệ và thất vọng")
                self.assertEqual(get_predicted_label(result), "negative")

    def test_neutral_output(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                result = predict("Sản phẩm này bình thường")
                self.assertEqual(get_predicted_label(result), "neutral")

    def test_empty_text(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                with self.assertRaises(ValueError):
                    predict("")

    def test_whitespace_text(self):
        for predict in self.predictors:
            for text in ("   ", "\t\n\r", "\u00a0"):
                with self.subTest(model=predict.__name__, text=text):
                    with self.assertRaises(ValueError):
                        predict(text)

    def test_output_probabilities_for_each_branch(self):
        texts = (
            "Sản phẩm này rất tốt",
            "Sản phẩm này rất tệ và thất vọng",
            "Sản phẩm này bình thường",
        )
        cases = (
            (predict_svm, (
                (0.08, 0.17, 0.75),
                (0.75, 0.17, 0.08),
                (0.20, 0.60, 0.20),
            )),
            (predict_transformer, (
                (0.05, 0.10, 0.85),
                (0.85, 0.10, 0.05),
                (0.15, 0.70, 0.15),
            )),
        )
        for predict, expected_outputs in cases:
            for text, expected in zip(texts, expected_outputs):
                with self.subTest(model=predict.__name__, text=text):
                    result = predict(text)
                    self.assertIsInstance(result, dict)
                    self.assertEqual(list(result), LABELS)
                    self.assertIsNone(validate_prediction_output(result))
                    self.assertTrue(all(isinstance(value, float) for value in result.values()))
                    self.assertAlmostEqual(sum(result.values()), 1.0)
                    self.assertEqual(tuple(result[label] for label in LABELS), expected)

    def test_deterministic_output(self):
        for predict in self.predictors:
            for text in ("rất tốt", "quá tệ", "bình thường"):
                with self.subTest(model=predict.__name__, text=text):
                    self.assertEqual(predict(text), predict(text))

    def test_non_string_input(self):
        for predict in self.predictors:
            for text in (None, 123, 0.5, True, [], {}, b"text"):
                with self.subTest(model=predict.__name__, text=text):
                    with self.assertRaises(TypeError):
                        predict(text)

    def test_positive_occurrences_outweigh_negative(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                self.assertEqual(get_predicted_label(predict("tốt tốt tệ")), "positive")

    def test_negative_occurrences_outweigh_positive(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                self.assertEqual(get_predicted_label(predict("tệ tệ tốt")), "negative")

    def test_equal_counts_are_neutral(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                self.assertEqual(get_predicted_label(predict("đẹp nhưng chậm")), "neutral")

    def test_case_punctuation_and_phrase_whitespace(self):
        for predict in self.predictors:
            for text, expected in (
                ("RẤT TỐT!", "positive"),
                ("Tôi HÀI\t LÒNG.", "positive"),
                ("THẤT\nVỌNG!", "negative"),
            ):
                with self.subTest(model=predict.__name__, text=text):
                    self.assertEqual(get_predicted_label(predict(text)), expected)

    def test_keywords_do_not_match_inside_other_words(self):
        for predict in self.predictors:
            with self.subTest(model=predict.__name__):
                self.assertEqual(get_predicted_label(predict("Đây là tệp dữ liệu")), "neutral")


if __name__ == "__main__":
    unittest.main()
