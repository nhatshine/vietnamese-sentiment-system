"""Error analysis bằng synthetic data; không thay đổi model hoặc Bước 9."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend.src.error_analysis import (
    complementary_errors, model_metrics, confidence_analysis, ensemble_effects,
    source_analysis, length_analysis, extract_error_examples, analyze_validation_errors,
)
from backend import run_error_analysis as runner


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 2, 0])
        self.svm = np.array([[.8, .1, .1], [.1, .8, .1], [.7, .1, .2], [.1, .8, .1]])
        self.transformer = np.array([[.7, .2, .1], [.7, .2, .1], [.1, .1, .8], [.1, .1, .8]])
        self.texts = ["mẫu một", "mẫu hai", "mẫu ba", "mẫu bốn"]
        self.sources = ["hieu_sa", "hieu_sa", "neu_esc", "vsfc"]

    def test_four_correctness_groups(self):
        result = complementary_errors(self.y, self.svm, self.transformer)
        self.assertEqual([result["groups"][key]["count"] for key in ("A", "B", "C", "D")], [1, 1, 1, 1])
        self.assertTrue(all(group["rate"] == .25 for group in result["groups"].values()))

    def test_group_counts_partition_all_samples(self):
        result = complementary_errors(self.y, self.svm, self.transformer)
        self.assertEqual(sum(group["count"] for group in result["groups"].values()), 4)
        self.assertEqual(result["svm_correct_count"], 2)
        self.assertEqual(result["transformer_wrong_count"], 2)

    def test_rescue_rates(self):
        result = complementary_errors(self.y, self.svm, self.transformer)
        self.assertEqual(result["transformer_rescue_rate"], .5)
        self.assertEqual(result["svm_rescue_rate"], .5)

    def test_zero_error_denominator_is_explicitly_undefined(self):
        result = complementary_errors([0, 1, 2], np.eye(3), np.eye(3))
        self.assertIsNone(result["transformer_rescue_rate"])
        self.assertIsNone(result["svm_rescue_rate"])
        self.assertEqual(result["oracle_accuracy"], 1.0)

    def test_oracle_accuracy_and_fixed_svm_fallback(self):
        result = complementary_errors(self.y, self.svm, self.transformer)
        self.assertEqual(result["oracle_accuracy"], .75)
        self.assertAlmostEqual(result["oracle_macro_f1"], 7 / 9)
        self.assertIn("ground truth", result["oracle_rule"])

    def test_confusion_matrix_row_truth_column_prediction(self):
        result = model_metrics(self.y, self.svm)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_allclose(result["confusion_matrix_row_normalized"], [[.5, .5, 0], [0, 1, 0], [1, 0, 0]])

    def test_per_class_precision_recall_f1(self):
        rows = model_metrics(self.y, self.svm)["per_class"]
        self.assertEqual([row["label"] for row in rows], ["negative", "neutral", "positive"])
        self.assertEqual(rows[0]["precision"], .5)
        self.assertEqual(rows[0]["recall"], .5)
        self.assertEqual(rows[0]["f1"], .5)
        self.assertAlmostEqual(rows[1]["f1"], 2 / 3)
        self.assertEqual([row["support"] for row in rows], [2, 1, 1])

    def test_macro_f1_from_all_predictions(self):
        result = model_metrics(self.y, self.svm)
        self.assertAlmostEqual(result["macro_f1"], 7 / 18)
        self.assertEqual(result["accuracy"], .5)

    def test_missing_truth_class_has_zero_normalized_row(self):
        result = model_metrics([0], [[1., 0., 0.]])
        self.assertEqual(result["confusion_matrix_row_normalized"][1], [0., 0., 0.])
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)

    def test_confidence_mean_median_and_error_denominator(self):
        result = confidence_analysis([1, 1, 1, 0], [[.8, .1, .1], [.9, .05, .05], [.95, .03, .02], [.7, .2, .1]])
        self.assertEqual(result["wrong_count"], 3)
        self.assertAlmostEqual(result["mean_confidence_correct"], .7)
        self.assertAlmostEqual(result["mean_confidence_wrong"], (.8 + .9 + .95) / 3)
        self.assertAlmostEqual(result["median_confidence_wrong"], .9)

    def test_high_confidence_thresholds_are_inclusive_among_errors(self):
        result = confidence_analysis([1, 1, 1, 0], [[.8, .1, .1], [.9, .05, .05], [.95, .03, .02], [.7, .2, .1]])
        self.assertEqual(result["wrong_high_confidence_counts"], {"0.8": 3, "0.9": 2, "0.95": 1})
        self.assertEqual(result["wrong_high_confidence_rates"], {"0.8": 1., "0.9": 2 / 3, "0.95": 1 / 3})

    def test_ece_known_two_sample_case(self):
        result = confidence_analysis([0, 1], [[.8, .1, .1], [.8, .1, .1]])
        self.assertAlmostEqual(result["ece"], .3)
        self.assertEqual(result["bins"][8]["count"], 2)

    def test_ece_includes_probability_one_and_empty_error_group(self):
        result = confidence_analysis([0, 1, 2], np.eye(3))
        self.assertEqual(result["ece"], 0.0)
        self.assertEqual(result["bins"][-1]["count"], 3)
        self.assertIsNone(result["mean_confidence_wrong"])
        self.assertTrue(all(value is None for value in result["wrong_high_confidence_rates"].values()))

    def test_ece_all_confident_predictions_wrong(self):
        self.assertEqual(confidence_analysis([1, 2, 0], np.eye(3))["ece"], 1.)

    def test_length_boundaries_and_actual_content_limit(self):
        lengths = [1, 10, 11, 20, 21, 50, 51, 128, 129]
        texts = [" ".join(["từ"] * count) for count in lengths]
        labels = np.arange(9) % 3
        probabilities = np.eye(3)[labels]
        rows = length_analysis(labels, probabilities, probabilities, texts, .8, max_content_tokens=127)
        self.assertEqual([row["n"] for row in rows], [2, 2, 2, 2, 1])
        self.assertEqual(sum(row["truncation_exposed_count"] for row in rows), 2)
        self.assertTrue(rows[-1]["small_sample"])
        self.assertEqual(rows[-1]["present_labels"], ["positive"])

    def test_empty_length_groups_have_undefined_scores(self):
        rows = length_analysis([0], [[1., 0., 0.]], [[1., 0., 0.]], ["một câu"], .8)
        self.assertEqual(rows[-1]["n"], 0)
        self.assertIsNone(rows[-1]["svm_macro_f1"])

    def test_source_counts_complementarity_and_f1(self):
        rows = source_analysis(self.y, self.svm, self.transformer, self.sources, .8)
        self.assertEqual([row["source"] for row in rows], ["hieu_sa", "neu_esc", "vsfc"])
        self.assertEqual([row["n"] for row in rows], [2, 1, 1])
        self.assertEqual(rows[0]["svm_correct_count"], 2)
        self.assertEqual(rows[1]["svm_wrong_transformer_correct"], 1)
        self.assertEqual(rows[2]["both_wrong_count"], 1)

    def test_source_and_text_alignment_errors(self):
        with self.assertRaises(ValueError):
            source_analysis(self.y, self.svm, self.transformer, self.sources[:2], .8)
        with self.assertRaises(ValueError):
            length_analysis(self.y, self.svm, self.transformer, self.texts[:2], .8)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            complementary_errors(self.y, self.svm[:2], self.transformer)

    def test_nan_infinity_and_invalid_probabilities(self):
        for value in (float("nan"), float("inf"), -.1, 1.1):
            invalid = self.svm.copy()
            invalid[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    complementary_errors(self.y, invalid, self.transformer)
        with self.assertRaises(ValueError):
            model_metrics([0], [[.1, .2, .6]])

    def test_no_mutation(self):
        before = [self.y.copy(), self.svm.copy(), self.transformer.copy(), self.texts.copy(), self.sources.copy()]
        analyze_validation_errors(self.y, self.svm, self.transformer, self.texts, self.sources, .8)
        for actual, expected in zip((self.y, self.svm, self.transformer), before[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.texts, before[3])
        self.assertEqual(self.sources, before[4])

    def test_ensemble_uses_supplied_alpha(self):
        half = ensemble_effects(self.y, self.svm, self.transformer, .5)
        other = ensemble_effects(self.y, self.svm, self.transformer, .2)
        self.assertEqual(half["ensemble_correct_count"], 3)
        self.assertEqual(other["ensemble_correct_count"], 2)

    def test_rescue_loss_net_accuracy_are_distinct_from_macro_f1(self):
        result = ensemble_effects(self.y, self.svm, self.transformer, .2)
        self.assertEqual(result["svm_wrong_to_ensemble_correct"], 1)
        self.assertEqual(result["svm_correct_to_ensemble_wrong"], 1)
        self.assertEqual(result["net_correct_vs_svm"], 0)
        self.assertEqual(result["accuracy_difference_vs_svm"], 0.)

    def test_both_models_wrong_can_be_rescued_by_probability_blend(self):
        result = ensemble_effects([2], [[.5, .05, .45]], [[.05, .5, .45]], .8)
        self.assertEqual(result["both_wrong_to_ensemble_correct"], 1)

    def test_examples_keep_text_index_and_separate_group_from_effect(self):
        examples = extract_error_examples(self.y, self.svm, self.transformer, self.texts, self.sources, .2, limit_per_group=10)
        self.assertTrue(examples)
        for row in examples:
            self.assertEqual(row["text"], self.texts[row["sample_index"]])
            self.assertIn("error_group", row)
            self.assertIn("ensemble_effect", row)
        rescued = next(row for row in examples if row["sample_index"] == 2)
        self.assertEqual(rescued["error_group"], "svm_wrong_transformer_correct")
        self.assertEqual(rescued["ensemble_effect"], "ensemble_rescues_svm")

    def test_example_selection_is_deterministic(self):
        first = extract_error_examples(self.y, self.svm, self.transformer, self.texts, self.sources, .8, limit_per_group=1)
        second = extract_error_examples(self.y, self.svm, self.transformer, self.texts, self.sources, .8, limit_per_group=1)
        self.assertEqual(first, second)
        self.assertEqual([row["sample_index"] for row in first], sorted(row["sample_index"] for row in first))

    def test_no_checkpoint_loading_or_model_selection(self):
        with patch("backend.src.real_svm.load_model", side_effect=AssertionError("No inference")):
            with patch("backend.src.real_transformer.load_model", side_effect=AssertionError("No inference")):
                summary = analyze_validation_errors(self.y, self.svm, self.transformer, self.texts, self.sources, .8)
        self.assertNotIn("BEST_METHOD", summary)
        self.assertNotIn("optimal_result", summary)

    def test_empty_dataset_and_invalid_bins_are_rejected(self):
        with self.assertRaises(ValueError):
            complementary_errors([], np.zeros((0, 3)), np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            confidence_analysis(self.y, self.svm, bins=0)


class FrozenValidationCacheTests(unittest.TestCase):
    def test_report_handles_undefined_error_rates(self):
        summary = analyze_validation_errors([0, 1, 2], np.eye(3), np.eye(3),
                                            ["một", "hai", "ba"], ["hieu_sa", "neu_esc", "vsfc"], .8)
        summary["provenance"] = {"cache_status": "synthetic-unit-test"}
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.md"
            with patch.object(runner, "REPORT_PATH", report):
                runner._write_report(summary)
            self.assertTrue(report.is_file())

    def test_report_renders_from_synthetic_summary(self):
        summary = analyze_validation_errors(
            [0, 1, 2], [[.8, .1, .1], [.8, .1, .1], [.1, .1, .8]],
            [[.7, .2, .1], [.1, .8, .1], [.8, .1, .1]],
            ["một", "hai", "ba"], ["hieu_sa", "neu_esc", "vsfc"], .8,
        )
        summary["provenance"] = {"cache_status": "synthetic-unit-test"}
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.md"
            with patch.object(runner, "REPORT_PATH", report):
                runner._write_report(summary)
            self.assertTrue(report.is_file())
            self.assertGreater(report.stat().st_size, 0)

    @unittest.skipUnless(runner.VALIDATION_PATH.is_file(), "Requires original backend/data/val.csv; dataset texts are not redistributed")
    def test_valid_cache_reused_without_inference(self):
        with patch.object(runner, "_recompute_probabilities", side_effect=AssertionError("Cache should be reused")):
            inputs = runner.load_analysis_inputs()
        self.assertEqual(inputs["cache_status"], "reused")
        self.assertEqual(inputs["alpha"], .8)
        self.assertEqual(len(inputs["data"]["texts"]), 8238)

    @unittest.skipUnless(runner.VALIDATION_PATH.is_file(), "Requires original backend/data/val.csv; dataset texts are not redistributed")
    def test_changed_frozen_fingerprints_stop(self):
        with patch.object(runner, "collect_fingerprints", return_value={}):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                runner.load_analysis_inputs()

    @unittest.skipUnless(runner.VALIDATION_PATH.is_file(), "Requires original backend/data/val.csv; dataset texts are not redistributed")
    def test_misaligned_cache_recomputed_without_editing_original(self):
        with np.load(runner.CACHE_PATH, allow_pickle=False) as cache:
            content = {name: cache[name].copy() for name in cache.files}
        original_svm, original_transformer = content["svm_probs"], content["transformer_probs"]
        content["y_true"] = content["y_true"][::-1]
        with tempfile.TemporaryDirectory() as folder:
            altered = Path(folder) / "invalid_cache.npz"
            np.savez_compressed(altered, **content)
            with patch.object(runner, "CACHE_PATH", altered):
                with patch.object(runner, "_recompute_probabilities", return_value=(original_svm, original_transformer)) as recompute:
                    inputs = runner.load_analysis_inputs()
        self.assertEqual(inputs["cache_status"], "recomputed")
        self.assertEqual(recompute.call_count, 1)

    @unittest.skipUnless(runner.VALIDATION_PATH.is_file(), "Requires original backend/data/val.csv; dataset texts are not redistributed")
    def test_selected_alpha_or_score_inconsistency_stops(self):
        selection = json.loads(runner.SELECTION_PATH.read_text(encoding="utf-8"))
        selection["selected_alpha"] = .5
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "alpha_selection.json"
            path.write_text(json.dumps(selection), encoding="utf-8")
            with patch.object(runner, "SELECTION_PATH", path):
                with self.assertRaises(ValueError):
                    runner.load_analysis_inputs()

    def test_validation_loader_refuses_test_filename(self):
        with patch.object(runner, "VALIDATION_PATH", Path("never_open/test.csv")):
            with self.assertRaisesRegex(ValueError, "val.csv"):
                runner.load_analysis_inputs()


if __name__ == "__main__":
    unittest.main()
