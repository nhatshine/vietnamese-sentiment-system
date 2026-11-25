"""Kiểm tra aggregation bằng dữ liệu tính tay và pipeline mock hiện có."""

import copy
import unittest

from backend.src.config import LABELS
from backend.src.contracts import validate_prediction_output
from backend.src.document_aggregator import (
    aggregate_all_models,
    aggregate_confidence_weighted,
    aggregate_majority_vote,
    aggregate_mean_probability,
    aggregate_model_document,
    analyze_document,
)
from backend.src.sentence_analyzer import analyze_sentences


HAND_PROBABILITIES = (
    {"negative": 0.05, "neutral": 0.10, "positive": 0.85},
    {"negative": 0.80, "neutral": 0.15, "positive": 0.05},
    {"negative": 0.70, "neutral": 0.20, "positive": 0.10},
)
MODELS = ("svm", "transformer", "ensemble")
AGGREGATIONS = ("majority_vote", "mean_probability", "confidence_weighted")
DIRECT_AGGREGATORS = (
    aggregate_majority_vote, aggregate_mean_probability,
    aggregate_confidence_weighted, aggregate_model_document,
)


def make_sentence_results(probabilities):
    """Tạo fixture hợp lệ; kỳ vọng trong test được tính tay độc lập."""
    results = []
    for index, distribution in enumerate(probabilities):
        label = max(LABELS, key=distribution.__getitem__)
        result = {"sentence_index": index, "sentence": f"Câu {index + 1}"}
        for model in MODELS:
            result[model] = {
                "method": model,
                "label": label,
                "confidence": distribution[label],
                "probabilities": distribution.copy(),
            }
            if model == "ensemble":
                result[model]["weights"] = {"svm": 0.5, "transformer": 0.5}
        results.append(result)
    return results


class MajorityVoteTests(unittest.TestCase):
    def setUp(self):
        self.results = make_sentence_results(HAND_PROBABILITIES)

    def test_majority_positive(self):
        results = make_sentence_results((HAND_PROBABILITIES[0],) * 2 + (HAND_PROBABILITIES[1],))
        self.assertEqual(aggregate_majority_vote(results, "svm")["label"], "positive")

    def test_majority_negative(self):
        self.assertEqual(aggregate_majority_vote(self.results, "svm")["label"], "negative")

    def test_majority_neutral(self):
        neutral = {"negative": 0.1, "neutral": 0.8, "positive": 0.1}
        results = make_sentence_results((neutral, neutral, HAND_PROBABILITIES[0]))
        self.assertEqual(aggregate_majority_vote(results, "svm")["label"], "neutral")

    def test_vote_counts(self):
        result = aggregate_majority_vote(self.results, "svm")
        self.assertEqual(result["vote_counts"], {"negative": 2, "neutral": 0, "positive": 1})
        self.assertEqual(result["aggregation_method"], "majority_vote")

    def test_vote_distribution_sums_to_one(self):
        distribution = aggregate_majority_vote(self.results, "svm")["vote_distribution"]
        self.assertEqual(distribution, {"negative": 2 / 3, "neutral": 0.0, "positive": 1 / 3})
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_confidence_is_winning_vote_ratio(self):
        self.assertEqual(aggregate_majority_vote(self.results, "svm")["confidence"], 2 / 3)

    def test_exact_tie_uses_label_order(self):
        results = make_sentence_results((
            {"negative": 0.0, "neutral": 0.0, "positive": 1.0},
            {"negative": 1.0, "neutral": 0.0, "positive": 0.0},
        ))
        self.assertEqual(aggregate_majority_vote(results, "svm")["label"], "negative")

    def test_tie_prefers_mean_probability_before_label_order(self):
        results = make_sentence_results((
            {"negative": 0.05, "neutral": 0.05, "positive": 0.90},
            {"negative": 0.55, "neutral": 0.05, "positive": 0.40},
        ))
        result = aggregate_majority_vote(results, "svm")
        self.assertEqual(result["label"], "positive")
        self.assertEqual(result["confidence"], 0.5)

    def test_repeated_ties_are_deterministic(self):
        results = make_sentence_results((
            {"negative": 0.5, "neutral": 0.0, "positive": 0.5},
            {"negative": 0.0, "neutral": 0.0, "positive": 1.0},
        ))
        outputs = [aggregate_majority_vote(results, "svm") for _ in range(5)]
        self.assertTrue(all(result == outputs[0] for result in outputs))
        self.assertEqual(outputs[0]["label"], "positive")

    def test_tie_break_only_considers_labels_tied_for_votes(self):
        results = make_sentence_results((
            {"negative": 0.50, "neutral": 0.49, "positive": 0.01},
            {"negative": 0.01, "neutral": 0.49, "positive": 0.50},
        ))
        # Neutral có mean cao nhất nhưng không có phiếu, nên không được thắng tie.
        self.assertEqual(aggregate_majority_vote(results, "svm")["label"], "negative")


class MeanProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.results = make_sentence_results(HAND_PROBABILITIES)

    def test_two_sentence_formula(self):
        probabilities = aggregate_mean_probability(self.results[:2], "svm")["probabilities"]
        for label, expected in (("negative", 0.425), ("neutral", 0.125), ("positive", 0.45)):
            self.assertAlmostEqual(probabilities[label], expected)

    def test_three_sentence_formula(self):
        probabilities = aggregate_mean_probability(self.results, "svm")["probabilities"]
        self.assertAlmostEqual(probabilities["negative"], 0.5166666666666667)
        self.assertAlmostEqual(probabilities["neutral"], 0.15)
        self.assertAlmostEqual(probabilities["positive"], 0.3333333333333333)

    def test_each_class_is_averaged_by_its_name(self):
        results = make_sentence_results((
            {"positive": 0.85, "negative": 0.05, "neutral": 0.10},
            {"neutral": 0.15, "positive": 0.05, "negative": 0.80},
        ))
        probabilities = aggregate_mean_probability(results, "svm")["probabilities"]
        self.assertEqual(list(probabilities), LABELS)
        self.assertAlmostEqual(probabilities["positive"], 0.45)
        self.assertAlmostEqual(probabilities["negative"], 0.425)

    def test_mean_probability_contract(self):
        probabilities = aggregate_mean_probability(self.results, "svm")["probabilities"]
        self.assertIsNone(validate_prediction_output(probabilities))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_mean_label_is_argmax(self):
        self.assertEqual(aggregate_mean_probability(self.results, "svm")["label"], "negative")

    def test_mean_confidence_is_winning_probability(self):
        result = aggregate_mean_probability(self.results, "svm")
        self.assertAlmostEqual(result["confidence"], 0.5166666666666667)
        self.assertEqual(result["confidence"], result["probabilities"][result["label"]])


class ConfidenceWeightedTests(unittest.TestCase):
    def setUp(self):
        self.results = make_sentence_results(HAND_PROBABILITIES)

    def test_weight_is_maximum_sentence_probability(self):
        result = aggregate_confidence_weighted(self.results, "svm")
        self.assertEqual(result["sentence_weights"], [0.85, 0.80, 0.70])

    def test_higher_confidence_has_more_influence(self):
        results = make_sentence_results((
            {"negative": 0.05, "neutral": 0.05, "positive": 0.90},
            {"negative": 0.55, "neutral": 0.25, "positive": 0.20},
        ))
        result = aggregate_confidence_weighted(results, "svm")
        self.assertAlmostEqual(result["probabilities"]["positive"], 0.92 / 1.45)
        self.assertGreater(result["probabilities"]["positive"], 0.55)

    def test_weighted_probabilities_match_hand_calculation(self):
        result = aggregate_confidence_weighted(self.results, "svm")
        # Tính tay: mẫu số 2.35, tử số lần lượt 1.1725, 0.345, 0.8325.
        for label, expected in (
            ("negative", 1.1725 / 2.35),
            ("neutral", 0.345 / 2.35),
            ("positive", 0.8325 / 2.35),
        ):
            self.assertAlmostEqual(result["probabilities"][label], expected)

    def test_weighted_probability_contract(self):
        probabilities = aggregate_confidence_weighted(self.results, "svm")["probabilities"]
        self.assertIsNone(validate_prediction_output(probabilities))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_weighted_label(self):
        self.assertEqual(aggregate_confidence_weighted(self.results, "svm")["label"], "negative")

    def test_weighted_confidence(self):
        result = aggregate_confidence_weighted(self.results, "svm")
        self.assertAlmostEqual(result["confidence"], 1.1725 / 2.35)
        self.assertEqual(result["confidence"], result["probabilities"]["negative"])

    def test_sentence_weights_preserve_order_and_are_not_normalized(self):
        result = aggregate_confidence_weighted(list(reversed(self.results)), "svm")
        self.assertEqual(result["sentence_weights"], [0.70, 0.80, 0.85])
        self.assertAlmostEqual(sum(result["sentence_weights"]), 2.35)


class AggregationValidationTests(unittest.TestCase):
    def setUp(self):
        self.results = make_sentence_results(HAND_PROBABILITIES)

    def test_empty_list_is_rejected(self):
        for aggregate in DIRECT_AGGREGATORS:
            with self.subTest(function=aggregate.__name__):
                with self.assertRaises(ValueError):
                    aggregate([], "svm")
        with self.assertRaises(ValueError):
            aggregate_all_models([])

    def test_non_list_input_is_rejected(self):
        for value in (None, {}, (), "text"):
            for aggregate in DIRECT_AGGREGATORS:
                with self.subTest(value=value, function=aggregate.__name__):
                    with self.assertRaises(TypeError):
                        aggregate(value, "svm")
            with self.assertRaises(TypeError):
                aggregate_all_models(value)

    def test_invalid_model_key_is_rejected(self):
        for model in ("bert", "random", None, [], 123):
            for aggregate in DIRECT_AGGREGATORS:
                with self.subTest(model=model, function=aggregate.__name__):
                    with self.assertRaises(ValueError):
                        aggregate(self.results, model)

    def test_missing_model_key_is_rejected(self):
        del self.results[1]["svm"]
        for aggregate in DIRECT_AGGREGATORS:
            with self.subTest(function=aggregate.__name__):
                with self.assertRaisesRegex(ValueError, "svm"):
                    aggregate(self.results, "svm")

    def test_invalid_probabilities_are_rejected(self):
        for invalid in (
            {"positive": 1.0},
            {"negative": -0.1, "neutral": 0.2, "positive": 0.9},
            {"negative": 0.0, "neutral": 0.0, "positive": 1.1},
            {"negative": 0.1, "neutral": 0.2, "positive": 0.6},
            {"negative": 0.1, "neutral": 0.2, "positive": float("nan")},
        ):
            for aggregate in DIRECT_AGGREGATORS:
                with self.subTest(probabilities=invalid, function=aggregate.__name__):
                    results = copy.deepcopy(self.results)
                    results[0]["svm"]["probabilities"] = invalid
                    with self.assertRaises(ValueError):
                        aggregate(results, "svm")

    def test_aggregation_does_not_mutate_any_input(self):
        original = copy.deepcopy(self.results)
        for model in MODELS:
            for aggregate in DIRECT_AGGREGATORS:
                aggregate(self.results, model)
                self.assertEqual(self.results, original)
        output = aggregate_all_models(self.results)
        output["svm"]["mean_probability"]["probabilities"]["positive"] = 0.0
        output["ensemble"]["confidence_weighted"]["sentence_weights"][0] = 0.0
        self.assertEqual(self.results, original)

    def test_non_dictionary_sentence_item_is_rejected(self):
        for aggregate in DIRECT_AGGREGATORS:
            with self.subTest(function=aggregate.__name__):
                with self.assertRaises(TypeError):
                    aggregate([None], "svm")

    def test_non_dictionary_prediction_is_rejected(self):
        self.results[0]["svm"] = "positive"
        for aggregate in DIRECT_AGGREGATORS:
            with self.subTest(function=aggregate.__name__):
                with self.assertRaises(TypeError):
                    aggregate(self.results, "svm")

    def test_missing_prediction_fields_are_rejected(self):
        for field in ("label", "confidence", "probabilities"):
            for aggregate in DIRECT_AGGREGATORS:
                results = copy.deepcopy(self.results)
                del results[0]["svm"][field]
                with self.subTest(field=field, function=aggregate.__name__):
                    with self.assertRaisesRegex(ValueError, field):
                        aggregate(results, "svm")

    def test_invalid_label_is_rejected(self):
        for label in ("Positive", "mixed", None, []):
            for aggregate in DIRECT_AGGREGATORS:
                results = copy.deepcopy(self.results)
                results[0]["svm"]["label"] = label
                with self.subTest(label=label, function=aggregate.__name__):
                    with self.assertRaises(ValueError):
                        aggregate(results, "svm")

    def test_non_numeric_or_boolean_confidence_is_rejected(self):
        for confidence in ("0.85", None, True, []):
            for aggregate in DIRECT_AGGREGATORS:
                results = copy.deepcopy(self.results)
                results[0]["svm"]["confidence"] = confidence
                with self.subTest(confidence=confidence, function=aggregate.__name__):
                    with self.assertRaises(TypeError):
                        aggregate(results, "svm")

    def test_out_of_range_or_non_finite_confidence_is_rejected(self):
        for confidence in (-0.1, 1.1, float("nan"), float("inf")):
            for aggregate in DIRECT_AGGREGATORS:
                results = copy.deepcopy(self.results)
                results[0]["svm"]["confidence"] = confidence
                with self.subTest(confidence=confidence, function=aggregate.__name__):
                    with self.assertRaises(ValueError):
                        aggregate(results, "svm")

    def test_label_must_be_a_probability_argmax(self):
        self.results[0]["svm"]["label"] = "negative"
        self.results[0]["svm"]["confidence"] = 0.05
        for aggregate in DIRECT_AGGREGATORS:
            with self.subTest(function=aggregate.__name__):
                with self.assertRaises(ValueError):
                    aggregate(self.results, "svm")

    def test_confidence_must_match_winning_probability(self):
        self.results[0]["svm"]["confidence"] = 0.5
        for aggregate in DIRECT_AGGREGATORS:
            with self.subTest(function=aggregate.__name__):
                with self.assertRaises(ValueError):
                    aggregate(self.results, "svm")


class AggregationCompositionTests(unittest.TestCase):
    def setUp(self):
        self.results = make_sentence_results(HAND_PROBABILITIES)

    def test_model_document_has_three_methods(self):
        result = aggregate_model_document(self.results, "svm")
        self.assertEqual(set(result), {"model", *AGGREGATIONS})

    def test_model_metadata_matches_selected_model(self):
        for model in MODELS:
            result = aggregate_model_document(self.results, model)
            self.assertEqual(result["model"], model)
            for aggregation in AGGREGATIONS:
                self.assertEqual(result[aggregation]["model"], model)
                self.assertEqual(result[aggregation]["aggregation_method"], aggregation)

    def test_all_model_keys_are_present(self):
        self.assertEqual(set(aggregate_all_models(self.results)), set(MODELS))

    def test_all_models_have_three_aggregation_methods(self):
        result = aggregate_all_models(self.results)
        for model in MODELS:
            self.assertTrue(set(AGGREGATIONS).issubset(result[model]))
            self.assertEqual(result[model]["majority_vote"]["label"], "negative")

    def test_single_sentence_reduces_to_its_original_prediction(self):
        results = make_sentence_results(HAND_PROBABILITIES[:1])
        result = aggregate_model_document(results, "svm")
        self.assertEqual(result["majority_vote"]["confidence"], 1.0)
        for method in ("mean_probability", "confidence_weighted"):
            self.assertEqual(result[method]["label"], "positive")
            self.assertAlmostEqual(result[method]["confidence"], 0.85)
            for label in LABELS:
                self.assertAlmostEqual(result[method]["probabilities"][label], HAND_PROBABILITIES[0][label])


class AnalyzeDocumentTests(unittest.TestCase):
    paragraph = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"

    def test_single_sentence_document(self):
        result = analyze_document("Camera đẹp.")
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(set(result), {"text", "sentence_count", "sentences", "document_results"})
        self.assertEqual(result["document_results"]["svm"]["mean_probability"]["label"], "positive")

    def test_multiple_sentence_document(self):
        result = analyze_document(self.paragraph)
        self.assertEqual(set(result["document_results"]), set(MODELS))
        self.assertEqual([item["sentence"] for item in result["sentences"]], [
            "Camera rất đẹp.", "Nhưng pin quá yếu.", "Giao hàng rất nhanh!",
        ])

    def test_sentence_count(self):
        self.assertEqual(analyze_document(self.paragraph)["sentence_count"], 3)

    def test_raw_text_is_preserved(self):
        text = "  Camera đẹp.\n\n  Giao hàng nhanh!  "
        self.assertEqual(analyze_document(text)["text"], text)

    def test_sentence_results_are_preserved(self):
        expected = analyze_sentences(self.paragraph, alpha=0.3)
        self.assertEqual(analyze_document(self.paragraph, alpha=0.3)["sentences"], expected)

    def test_empty_document(self):
        self.assertEqual(analyze_document(""), {
            "text": "", "sentence_count": 0, "sentences": [], "document_results": {},
        })

    def test_whitespace_document(self):
        text = " \t\n\r\n "
        self.assertEqual(analyze_document(text), {
            "text": text, "sentence_count": 0, "sentences": [], "document_results": {},
        })

    def test_vietnamese_unicode(self):
        result = analyze_document("Tôi hài lòng 😊. Giao hàng nhanh!")
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["sentences"][0]["sentence"], "Tôi hài lòng 😊.")

    def test_alpha_is_forwarded_to_sentence_predictions(self):
        result = analyze_document("Camera đẹp.", alpha=0.3)
        ensemble = result["sentences"][0]["ensemble"]
        self.assertEqual(ensemble["weights"], {"svm": 0.3, "transformer": 0.7})
        self.assertAlmostEqual(ensemble["confidence"], 0.82)
        self.assertAlmostEqual(result["document_results"]["ensemble"]["mean_probability"]["confidence"], 0.82)

    def test_no_selected_best_method_metadata(self):
        result = analyze_document(self.paragraph)
        self.assertNotIn("BEST_METHOD", result)
        self.assertNotIn("best_model", result)
        self.assertNotIn("best_aggregation", result)
        for model_result in result["document_results"].values():
            self.assertEqual(set(model_result), {"model", *AGGREGATIONS})

    def test_no_optimal_result(self):
        result = analyze_document(self.paragraph)
        self.assertNotIn("optimal_result", result)
        self.assertNotIn("optimal", result["document_results"])

    def test_non_string_document_is_rejected(self):
        for text in (None, 123, True, [], {}, b"text"):
            with self.subTest(text=text):
                with self.assertRaises(TypeError):
                    analyze_document(text)

    def test_invalid_alpha_is_rejected_for_nonempty_document(self):
        for alpha, error in ((-0.1, ValueError), (1.1, ValueError), (True, TypeError), ("0.5", TypeError)):
            with self.subTest(alpha=alpha):
                with self.assertRaises(error):
                    analyze_document("Camera đẹp.", alpha=alpha)


if __name__ == "__main__":
    unittest.main()
