"""Kiểm tra alpha sweep bằng dữ liệu tổng hợp và đường inference cleaned."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend.src.alpha_tuning import (
    compute_ensemble_probabilities, evaluate_alpha, evaluate_probabilities, tune_alpha,
)
from backend.src.config import LABELS
from backend.src.ensemble import weighted_ensemble
from backend.run_alpha_tuning import load_validation


class AlphaTuningTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 2])
        self.svm = np.array([[.6, .3, .1], [.4, .5, .1], [.5, .1, .4]])
        self.transformer = np.array([[.4, .5, .1], [.1, .7, .2], [.1, .2, .7]])

    def test_weighted_formula(self):
        actual = compute_ensemble_probabilities(self.svm, self.transformer, .3)
        np.testing.assert_allclose(actual, [[.46, .44, .10], [.19, .64, .17], [.22, .17, .61]])

    def test_alpha_zero_equals_transformer(self):
        np.testing.assert_array_equal(compute_ensemble_probabilities(self.svm, self.transformer, 0), self.transformer)

    def test_alpha_one_equals_svm(self):
        np.testing.assert_array_equal(compute_ensemble_probabilities(self.svm, self.transformer, 1), self.svm)

    def test_equal_weights(self):
        np.testing.assert_allclose(compute_ensemble_probabilities(self.svm, self.transformer, .5),
                                   [[.5, .4, .1], [.25, .6, .15], [.3, .15, .55]])

    def test_vectorized_formula_matches_existing_engine(self):
        for alpha in (0, .3, .5, 1):
            actual = compute_ensemble_probabilities(self.svm, self.transformer, alpha)
            for i, (svm, transformer) in enumerate(zip(self.svm, self.transformer)):
                expected = weighted_ensemble(dict(zip(LABELS, svm)), dict(zip(LABELS, transformer)), alpha)
                np.testing.assert_array_equal(actual[i], [expected["probabilities"][label] for label in LABELS])

    def test_invalid_alpha(self):
        for alpha, error in ((True, TypeError), (".5", TypeError), (None, TypeError),
                             (-.1, ValueError), (1.1, ValueError), (float("nan"), ValueError),
                             (float("inf"), ValueError)):
            with self.subTest(alpha=alpha):
                with self.assertRaises(error):
                    compute_ensemble_probabilities(self.svm, self.transformer, alpha)

    def test_invalid_shapes(self):
        for values in (np.array([.1, .2, .7]), np.ones((3, 4)), np.zeros((0, 3)), np.ones((1, 1, 3))):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    compute_ensemble_probabilities(values, self.transformer, .5)

    def test_unequal_sample_counts(self):
        with self.assertRaises(ValueError):
            compute_ensemble_probabilities(self.svm[:2], self.transformer, .5)

    def test_non_finite_probabilities(self):
        for value in (np.nan, np.inf, -np.inf):
            invalid = self.svm.copy()
            invalid[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compute_ensemble_probabilities(invalid, self.transformer, .5)

    def test_out_of_range_probabilities(self):
        for row in ((-.1, .4, .7), (1.1, 0, 0)):
            invalid = self.svm.copy()
            invalid[0] = row
            with self.assertRaises(ValueError):
                compute_ensemble_probabilities(invalid, self.transformer, .5)

    def test_invalid_probability_totals(self):
        invalid = self.svm.copy()
        invalid[0] = [.1, .2, .6]
        with self.assertRaises(ValueError):
            compute_ensemble_probabilities(invalid, self.transformer, .5)

    def test_non_numeric_or_boolean_probability_arrays(self):
        for invalid in (self.svm.astype(str), np.eye(3, dtype=bool), self.svm.astype(complex)):
            with self.assertRaises(TypeError):
                compute_ensemble_probabilities(invalid, self.transformer, .5)

    def test_class_order_and_endpoint_baseline(self):
        metrics = evaluate_alpha(self.y, self.svm, self.transformer, 1)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics, {"alpha": 1.0, **evaluate_probabilities(self.y, self.svm)})

    def test_macro_metrics_match_hand_confusion_matrix(self):
        metrics = evaluate_probabilities(self.y, self.svm)
        self.assertAlmostEqual(metrics["macro_precision"], .5)
        self.assertAlmostEqual(metrics["macro_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 5 / 9)

    def test_macro_f1_is_from_combined_predictions_not_average_model_scores(self):
        self.assertEqual(evaluate_alpha(self.y, self.svm, self.transformer, .5)["macro_f1"], 1.0)
        self.assertAlmostEqual(evaluate_alpha(self.y, self.svm, self.transformer, 0)["macro_f1"], 5 / 9)

    def test_best_alpha_selected_by_macro_f1(self):
        result = tune_alpha(self.y, self.svm, self.transformer, [0, .5, 1])
        self.assertEqual(result["selected_alpha"], .5)
        self.assertEqual(result["selected_validation_macro_f1"], 1.0)

    def test_tie_prefers_endpoint_over_mixture(self):
        perfect = np.eye(3)
        self.assertEqual(tune_alpha(self.y, perfect, perfect, [.5, 1])["selected_alpha"], 1.0)

    def test_endpoint_tie_prefers_zero_independent_of_grid_order(self):
        perfect = np.eye(3)
        self.assertEqual(tune_alpha(self.y, perfect, perfect, [1, .5, 0])["selected_alpha"], 0.0)

    def test_mixture_tie_uses_ascending_alpha(self):
        perfect = np.eye(3)
        self.assertEqual(tune_alpha(self.y, perfect, perfect, [.7, .3, .5])["selected_alpha"], .3)

    def test_inputs_are_not_mutated(self):
        before = [self.y.copy(), self.svm.copy(), self.transformer.copy()]
        grid = [0, .5, 1]
        result = tune_alpha(self.y, self.svm, self.transformer, grid)
        for actual, expected in zip((self.y, self.svm, self.transformer), before):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(grid, [0, .5, 1])
        self.assertEqual(len(result["scores"]), 3)

    def test_invalid_ground_truth(self):
        for labels in ([0, 1], [0, 1, 3], [0.0, 1.0, 2.0], [[0, 1, 2]], [True, False, True]):
            with self.subTest(labels=labels):
                with self.assertRaises((TypeError, ValueError)):
                    evaluate_probabilities(labels, self.svm)

    def test_empty_or_duplicate_grid_is_rejected(self):
        for grid in ([], [0, 0]):
            with self.assertRaises(ValueError):
                tune_alpha(self.y, self.svm, self.transformer, grid)

    def test_fixed_three_class_macro_including_zero_division(self):
        result = evaluate_probabilities([0], [[1., 0., 0.]])
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)

    def test_synthetic_tuning_never_loads_models_or_selects_best_method(self):
        with patch("backend.src.real_svm.load_model", side_effect=AssertionError("No checkpoint")):
            with patch("backend.src.real_transformer.load_model", side_effect=AssertionError("No checkpoint")):
                result = tune_alpha(self.y, self.svm, self.transformer, [0, .5, 1])
        self.assertNotIn("BEST_METHOD", result)
        self.assertNotIn("best_method", result)


class ValidationDatasetTests(unittest.TestCase):
    def write_validation(self, folder, content):
        path = Path(folder) / "val.csv"
        path.write_text(content, encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_validation_fields_and_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path, sha = self.write_validation(folder, 'text,label,source\nđã làm sạch,2,vsfc\nkhông tốt,0,hieu_sa\n')
            result = load_validation(path, sha, expected_rows=2, expected_sources=("vsfc", "hieu_sa"))
            self.assertEqual(result["texts"], ["đã làm sạch", "không tốt"])
            np.testing.assert_array_equal(result["labels"], [2, 0])
            self.assertEqual(result["sources"], ["vsfc", "hieu_sa"])

    def test_hash_mismatch_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.write_validation(folder, 'text,label,source\nx,0,vsfc\n')
            with self.assertRaisesRegex(ValueError, "SHA"):
                load_validation(path, "wrong-hash")

    def test_missing_column_or_null_field_is_rejected(self):
        for content in ('text,label\nx,0\n', 'text,label,source\n,0,vsfc\n',
                        'text,label,source\nx,,vsfc\n', 'text,label,source\nx,3,vsfc\n'):
            with tempfile.TemporaryDirectory() as folder:
                path, sha = self.write_validation(folder, content)
                with self.assertRaises(ValueError):
                    load_validation(path, sha, expected_rows=1, expected_sources=("vsfc",))

    def test_wrong_row_count_or_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, sha = self.write_validation(folder, 'text,label,source\nx,0,vsfc\n')
            with self.assertRaises(ValueError):
                load_validation(path, sha, expected_rows=2, expected_sources=("vsfc",))
            with self.assertRaises(ValueError):
                load_validation(path, sha, expected_rows=1, expected_sources=("hieu_sa",))

    def test_test_filename_is_refused_without_opening_it(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "val.csv"):
                load_validation(Path(folder) / "test.csv", "unused")


class CleanedBatchInferenceTests(unittest.TestCase):
    def test_svm_batch_does_not_call_clean_text(self):
        from backend.src import real_svm
        texts = ["sản phẩm này rất tốt", "sản phẩm này rất tệ"]
        with patch.object(real_svm, "clean_text", side_effect=AssertionError("Double preprocessing")):
            result = real_svm.predict_svm_cleaned_batch(texts, batch_size=1)
        self.assertEqual(len(result), 2)
        direct = real_svm.load_model().predict_proba(texts)
        for index, row in enumerate(result):
            np.testing.assert_array_equal([row[label] for label in LABELS], direct[index])

    def test_transformer_cleaned_encoder_has_original_specials_and_padding(self):
        from backend.src import real_transformer
        from backend.models.transformer import tokenizer
        resources = real_transformer.load_model()
        with patch.object(tokenizer, "clean_text", side_effect=AssertionError("Double preprocessing")):
            ids, mask = real_transformer.encode_cleaned_text("rất tốt", resources.vocab, resources.max_len)
        self.assertEqual(ids[:3], [2, resources.vocab["rất"], resources.vocab["tốt"]])
        self.assertEqual(ids[3:], [0] * 125)
        self.assertEqual(mask, [1, 1, 1] + [0] * 125)

    def test_transformer_batch_does_not_call_clean_text_and_keeps_order(self):
        from backend.src import real_transformer
        from backend.models.transformer import tokenizer
        texts = ["sản phẩm này rất tốt", "sản phẩm này rất tệ"]
        expected = [real_transformer.predict_transformer(text) for text in texts]
        progress = []
        with patch.object(tokenizer, "clean_text", side_effect=AssertionError("Double preprocessing")):
            result = real_transformer.predict_transformer_cleaned_batch(
                texts, batch_size=1, progress=lambda done, total: progress.append((done, total)))
        for actual, reference in zip(result, expected):
            np.testing.assert_allclose([actual[label] for label in LABELS], [reference[label] for label in LABELS], atol=1e-7)
        self.assertEqual(progress, [(1, 2), (2, 2)])

    def test_cleaned_batch_validation_and_empty_input(self):
        from backend.src import real_svm, real_transformer
        for predictor in (real_svm.predict_svm_cleaned_batch, real_transformer.predict_transformer_cleaned_batch):
            self.assertEqual(predictor([]), [])
            with self.assertRaises(TypeError):
                predictor("not a list")
            with self.assertRaises(ValueError):
                predictor(["   "])
            with self.assertRaises(ValueError):
                predictor(["tốt"], batch_size=0)


if __name__ == "__main__":
    unittest.main()
