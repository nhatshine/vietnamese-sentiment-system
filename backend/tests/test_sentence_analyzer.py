"""Kiểm tra phân tích từng câu bằng các thành phần mock hiện có."""

import unittest
from unittest.mock import patch

from backend.src.contracts import validate_prediction_output
from backend.src.sentence_analyzer import analyze_sentence, analyze_sentences


class AnalyzeSentenceTests(unittest.TestCase):
    def test_positive_sentence(self):
        result = analyze_sentence("Camera rất đẹp.")
        for method in ("svm", "transformer", "ensemble"):
            with self.subTest(method=method):
                self.assertEqual(result[method]["label"], "positive")

    def test_negative_sentence(self):
        result = analyze_sentence("Sản phẩm rất tệ và thất vọng.")
        for method in ("svm", "transformer", "ensemble"):
            with self.subTest(method=method):
                self.assertEqual(result[method]["label"], "negative")

    def test_neutral_sentence(self):
        result = analyze_sentence("Sản phẩm này bình thường.")
        for method in ("svm", "transformer", "ensemble"):
            with self.subTest(method=method):
                self.assertEqual(result[method]["label"], "neutral")

    def test_sentence_field_keeps_exact_raw_input(self):
        raw = "  Camera\t  rất đẹp!  "
        self.assertEqual(analyze_sentence(raw)["sentence"], raw)

    def test_output_contains_svm_result(self):
        self.assertEqual(analyze_sentence("Tốt.")["svm"]["method"], "svm")

    def test_output_contains_transformer_result(self):
        self.assertEqual(analyze_sentence("Tốt.")["transformer"]["method"], "transformer")

    def test_output_contains_ensemble_result(self):
        self.assertEqual(analyze_sentence("Tốt.")["ensemble"]["method"], "ensemble")

    def test_svm_result_fields_and_values(self):
        self.assertEqual(analyze_sentence("Camera đẹp.")["svm"], {
            "method": "svm",
            "label": "positive",
            "confidence": 0.75,
            "probabilities": {"negative": 0.08, "neutral": 0.17, "positive": 0.75},
        })

    def test_transformer_result_fields_and_values(self):
        self.assertEqual(analyze_sentence("Camera đẹp.")["transformer"], {
            "method": "transformer",
            "label": "positive",
            "confidence": 0.85,
            "probabilities": {"negative": 0.05, "neutral": 0.10, "positive": 0.85},
        })

    def test_ensemble_result_fields_and_values(self):
        result = analyze_sentence("Camera đẹp.")["ensemble"]
        self.assertEqual(set(result), {"method", "label", "confidence", "probabilities", "weights"})
        self.assertEqual(result["method"], "ensemble")
        self.assertEqual(result["label"], "positive")
        self.assertAlmostEqual(result["confidence"], 0.80)
        for label, expected in (("negative", 0.065), ("neutral", 0.135), ("positive", 0.80)):
            self.assertAlmostEqual(result["probabilities"][label], expected)

    def test_ensemble_weights_follow_alpha(self):
        result = analyze_sentence("Tốt.", alpha=0.3)
        self.assertEqual(result["ensemble"]["weights"], {"svm": 0.3, "transformer": 0.7})

    def test_alpha_zero_equals_transformer(self):
        result = analyze_sentence("Tốt.", alpha=0)
        self.assertEqual(result["ensemble"]["probabilities"], result["transformer"]["probabilities"])
        self.assertEqual(result["ensemble"]["confidence"], result["transformer"]["confidence"])

    def test_alpha_one_equals_svm(self):
        result = analyze_sentence("Tốt.", alpha=1)
        self.assertEqual(result["ensemble"]["probabilities"], result["svm"]["probabilities"])
        self.assertEqual(result["ensemble"]["confidence"], result["svm"]["confidence"])

    def test_alpha_point_three_uses_existing_mock_probabilities(self):
        result = analyze_sentence("Camera đẹp.", alpha=0.3)["ensemble"]
        for label, expected in (("negative", 0.059), ("neutral", 0.121), ("positive", 0.82)):
            self.assertAlmostEqual(result["probabilities"][label], expected)
        self.assertAlmostEqual(result["confidence"], 0.82)

    def test_invalid_alpha_is_rejected(self):
        cases = (
            (-0.1, ValueError), (1.1, ValueError),
            (float("nan"), ValueError), (float("inf"), ValueError),
            ("0.5", TypeError), (True, TypeError), (None, TypeError),
        )
        for alpha, error_type in cases:
            with self.subTest(alpha=alpha):
                with self.assertRaises(error_type):
                    analyze_sentence("Camera đẹp.", alpha=alpha)

    def test_empty_sentence_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_sentence("")

    def test_whitespace_sentence_is_rejected(self):
        for sentence in ("   ", "\t\r\n", "\u00a0"):
            with self.subTest(sentence=sentence):
                with self.assertRaises(ValueError):
                    analyze_sentence(sentence)

    def test_non_string_sentence_is_rejected(self):
        for sentence in (None, 123, True, [], {}, b"text"):
            with self.subTest(sentence=sentence):
                with self.assertRaises(TypeError):
                    analyze_sentence(sentence)

    def test_provider_dictionaries_and_raw_input_are_not_mutated(self):
        sentence = "  Camera đẹp!  "
        svm_probs = {"negative": 0.08, "neutral": 0.17, "positive": 0.75}
        transformer_probs = {"negative": 0.05, "neutral": 0.10, "positive": 0.85}
        svm_before = svm_probs.copy()
        transformer_before = transformer_probs.copy()
        # Chỉ giữ lại tham chiếu output provider để kiểm tra quyền sở hữu dữ liệu.
        with patch("backend.src.sentence_analyzer.predict_svm", return_value=svm_probs):
            with patch("backend.src.sentence_analyzer.predict_transformer", return_value=transformer_probs):
                result = analyze_sentence(sentence)
        self.assertEqual(result["sentence"], "  Camera đẹp!  ")
        self.assertEqual(svm_probs, svm_before)
        self.assertEqual(transformer_probs, transformer_before)
        result["svm"]["probabilities"]["positive"] = 0.0
        result["transformer"]["probabilities"]["positive"] = 0.0
        self.assertEqual(svm_probs, svm_before)
        self.assertEqual(transformer_probs, transformer_before)

    def test_all_probability_outputs_pass_contract(self):
        result = analyze_sentence("Tốt.", alpha=0.3)
        for method in ("svm", "transformer", "ensemble"):
            with self.subTest(method=method):
                probabilities = result[method]["probabilities"]
                self.assertIsNone(validate_prediction_output(probabilities))
                self.assertEqual(result[method]["confidence"], probabilities[result[method]["label"]])


class AnalyzeSentencesTests(unittest.TestCase):
    paragraph = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"

    def test_single_sentence_paragraph(self):
        results = analyze_sentences("Camera rất đẹp.")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sentence"], "Camera rất đẹp.")
        self.assertEqual(results[0]["ensemble"]["label"], "positive")

    def test_multiple_sentence_paragraph(self):
        self.assertEqual(len(analyze_sentences(self.paragraph)), 3)

    def test_sentence_order_is_preserved(self):
        results = analyze_sentences(self.paragraph)
        self.assertEqual([result["sentence"] for result in results], [
            "Camera rất đẹp.", "Nhưng pin quá yếu.", "Giao hàng rất nhanh!",
        ])

    def test_sentence_indices_are_zero_based(self):
        results = analyze_sentences(self.paragraph)
        self.assertEqual([result["sentence_index"] for result in results], [0, 1, 2])

    def test_empty_paragraph_returns_empty_list(self):
        self.assertEqual(analyze_sentences(""), [])

    def test_whitespace_paragraph_returns_empty_list(self):
        self.assertEqual(analyze_sentences(" \t\n\r\n "), [])

    def test_vietnamese_unicode_is_preserved(self):
        results = analyze_sentences("Tôi hài lòng 😊. Giao hàng nhanh!")
        self.assertEqual([result["sentence"] for result in results], [
            "Tôi hài lòng 😊.", "Giao hàng nhanh!",
        ])

    def test_raw_sentence_punctuation_is_preserved(self):
        results = analyze_sentences("Tuyệt quá!!! Nhưng giá cao... Ổn chứ?!")
        self.assertEqual([result["sentence"] for result in results], [
            "Tuyệt quá!!!", "Nhưng giá cao...", "Ổn chứ?!",
        ])

    def test_abbreviation_is_handled_by_splitter(self):
        results = analyze_sentences("TS. Nguyễn Văn A đánh giá sản phẩm tốt. Tôi đồng ý.")
        self.assertEqual([result["sentence"] for result in results], [
            "TS. Nguyễn Văn A đánh giá sản phẩm tốt.", "Tôi đồng ý.",
        ])

    def test_decimal_is_handled_by_splitter(self):
        results = analyze_sentences("Giá khoảng 1.5 triệu. Tôi thấy khá hợp lý.")
        self.assertEqual([result["sentence"] for result in results], [
            "Giá khoảng 1.5 triệu.", "Tôi thấy khá hợp lý.",
        ])

    def test_each_sentence_has_three_methods_with_forwarded_alpha(self):
        for result in analyze_sentences(self.paragraph, alpha=0.3):
            self.assertEqual({result[method]["method"] for method in ("svm", "transformer", "ensemble")}, {
                "svm", "transformer", "ensemble",
            })
            self.assertEqual(result["ensemble"]["weights"], {"svm": 0.3, "transformer": 0.7})

    def test_output_has_only_sentence_results_without_document_metadata(self):
        results = analyze_sentences("Camera đẹp. Sản phẩm tệ. Giao hàng nhanh!")
        self.assertIsInstance(results, list)
        for result in results:
            self.assertEqual(set(result), {
                "sentence_index", "sentence", "svm", "transformer", "ensemble",
            })

    def test_non_string_paragraph_is_rejected_by_splitter(self):
        for text in (None, 123, True, [], {}, b"text"):
            with self.subTest(text=text):
                with self.assertRaises(TypeError):
                    analyze_sentences(text)


if __name__ == "__main__":
    unittest.main()
