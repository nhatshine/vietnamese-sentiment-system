"""Step 14.1: hand-checked selection, validation and comparison cases."""

import unittest

import numpy as np

try:
    from backend.src.max_confidence import select_max_confidence, compare_methods
except ModuleNotFoundError as error:
    if error.name != "backend.src.max_confidence":
        raise
    select_max_confidence = compare_methods = None


class MaxConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(select_max_confidence, "Max Confidence is not implemented yet")
        self.svm = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .1, .8]])
        self.transformer = np.array([[.1, .8, .1], [.1, .2, .7], [.8, .1, .1]])

    def test_selects_entire_distribution_and_preserves_row_order(self):
        result = select_max_confidence(self.svm, self.transformer)
        np.testing.assert_array_equal(result["selected_models"], ["transformer", "transformer", "svm"])
        np.testing.assert_array_equal(result["probabilities"], [[.1, .8, .1], [.1, .2, .7], [.1, .1, .8]])
        np.testing.assert_array_equal(result["ties"], [False, False, True])

    def test_exact_tie_prefers_svm_even_with_different_winning_label(self):
        result = select_max_confidence([[.8, .1, .1]], [[.1, .8, .1]])
        self.assertEqual(result["selected_models"].tolist(), ["svm"])
        np.testing.assert_array_equal(result["probabilities"], [[.8, .1, .1]])

    def test_near_tie_is_not_rounded_into_a_tie(self):
        result = select_max_confidence([[.8, .1, .1]], [[.1, .8000000001, .0999999999]])
        self.assertEqual(result["selected_models"].tolist(), ["transformer"])
        self.assertFalse(result["ties"][0])

    def test_canonical_label_columns_are_negative_neutral_positive(self):
        result = select_max_confidence(self.svm, self.transformer)
        self.assertEqual(result["probabilities"].argmax(1).tolist(), [1, 2, 2])

    def test_rejects_declared_wrong_label_order(self):
        with self.assertRaises(ValueError):
            select_max_confidence(self.svm, self.transformer, label_order=["positive", "neutral", "negative"])

    def test_rejects_mismatched_sample_counts(self):
        with self.assertRaises(ValueError):
            select_max_confidence(self.svm, self.transformer[:2])

    def test_rejects_invalid_distributions_without_normalizing(self):
        for bad in ([], [.2, .3, .5], [[.5, .5]], [[.5, .5, .5]],
                    [[-.1, .6, .5]], [[float("nan"), .5, .5]],
                    [[float("inf"), 0, 0]], [[1.1, 0, -.1]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_max_confidence(bad, bad)

    def test_rejects_non_numeric_bool_and_complex_probabilities(self):
        for bad in ([[".7", ".2", ".1"]], [[True, False, False]], [[1j, 0, 0]]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                select_max_confidence(bad, bad)

    def test_selection_does_not_mutate_or_alias_input(self):
        svm, transformer = self.svm.copy(), self.transformer.copy()
        result = select_max_confidence(self.svm, self.transformer)
        result["probabilities"][0, 0] = 0
        np.testing.assert_array_equal(self.svm, svm)
        np.testing.assert_array_equal(self.transformer, transformer)

    def test_single_row_and_class_argmax_tie_are_deterministic(self):
        result = select_max_confidence([[.5, .5, 0]], [[0, .4, .6]])
        self.assertEqual(result["probabilities"].argmax(1).tolist(), [2])
        result = select_max_confidence([[.5, .5, 0]], [[.4, .3, .3]])
        self.assertEqual(result["probabilities"].argmax(1).tolist(), [0])


class MethodComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(compare_methods, "Comparison is not implemented yet")
        # Rows represent both correct / Max only / Weighted only / both wrong.
        self.y = np.array([0, 2, 0, 1])
        self.svm = np.array([[.8, .1, .1], [.6, .2, .2], [.6, .2, .2], [.8, .1, .1]])
        self.transformer = np.array([[.7, .2, .1], [.05, .05, .9], [.05, .05, .9], [.7, .2, .1]])
        self.sources = ["hieu_sa", "hieu_sa", "neu_esc", "vsfc"]

    def test_comparison_uses_fixed_weight_and_hand_checked_confusions(self):
        result = compare_methods(self.y, self.svm, self.transformer, self.sources)
        self.assertEqual(result["alpha"], .8)
        mc, we = result["models"]["max_confidence"], result["models"]["weighted_ensemble"]
        self.assertEqual(mc["confusion_matrix"], [[1, 0, 1], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(we["confusion_matrix"], [[2, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertAlmostEqual(mc["macro_f1"], 7 / 18)
        self.assertAlmostEqual(we["macro_f1"], 2 / 9)
        self.assertEqual(mc["accuracy"], .5)
        self.assertAlmostEqual(mc["macro_precision"], 1 / 3)
        self.assertEqual(mc["macro_recall"], .5)

    def test_correctness_groups_and_disagreement_indices(self):
        result = compare_methods(self.y, self.svm, self.transformer, self.sources)
        self.assertEqual(result["correctness_groups"], {"both_correct": 1, "max_confidence_only": 1,
                                                       "weighted_ensemble_only": 1, "both_wrong": 1})
        self.assertEqual(result["disagreements"]["indices"], [1, 2])
        self.assertEqual(result["selection_counts"], {"svm": 2, "transformer": 2, "exact_ties": 0})

    def test_source_and_per_class_metrics_keep_three_classes(self):
        result = compare_methods(self.y, self.svm, self.transformer, self.sources)
        self.assertEqual([r["source"] for r in result["by_source"]], ["hieu_sa", "neu_esc", "vsfc"])
        self.assertEqual([r["sample_count"] for r in result["by_source"]], [2, 1, 1])
        rows = result["models"]["max_confidence"]["per_class"]
        self.assertEqual([r["label"] for r in rows], ["negative", "neutral", "positive"])
        self.assertAlmostEqual(rows[0]["f1"], .5)
        self.assertAlmostEqual(rows[2]["f1"], 2 / 3)

    def test_standalone_baselines_are_reported_for_the_same_samples(self):
        result = compare_methods(self.y, self.svm, self.transformer, self.sources)
        self.assertAlmostEqual(result["baselines"]["svm"]["macro_f1"], 2 / 9)
        self.assertAlmostEqual(result["baselines"]["transformer"]["macro_f1"], 7 / 18)

    def test_rejects_bad_truth_or_source_alignment(self):
        for y, sources in (([0], self.sources), ([0., 2., 0., 1.], self.sources),
                           ([0, 2, 0, 3], self.sources), (self.y, ["hieu_sa"]),
                           (self.y, ["unknown"] * 4)):
            with self.subTest(y=y, sources=sources), self.assertRaises((TypeError, ValueError)):
                compare_methods(y, self.svm, self.transformer, sources)


if __name__ == "__main__":
    unittest.main()
