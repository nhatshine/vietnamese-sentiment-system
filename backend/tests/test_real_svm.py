"""Inference thật, ánh xạ class và hybrid wiring; không đánh giá model."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.src import real_svm
from backend.src.adapters import normalize_svm_output
from backend.src.config import LABELS
from backend.src.contracts import get_predicted_label, validate_prediction_output
from backend.src.document_aggregator import analyze_document
from backend.src.mock_models import predict_svm as mock_svm
from backend.src.sentence_analyzer import analyze_sentence, analyze_sentences


class PipelineStub:
    """Chỉ dùng kiểm tra mapping/error, không thay artifact trong integration test."""
    def __init__(self, classes, row):
        self.classes_ = classes
        self.row = row

    def predict_proba(self, texts):
        return [self.row]


class RealSVMInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = real_svm.load_model()

    def assert_prediction(self, result):
        self.assertEqual(list(result), LABELS)
        self.assertTrue(all(type(value) is float for value in result.values()))
        self.assertIsNone(validate_prediction_output(result))
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertIn(get_predicted_label(result), LABELS)
        json.dumps(result, allow_nan=False)

    def test_artifact_loads(self):
        self.assertTrue(real_svm.MODEL_PATH.is_file())
        self.assertIsNotNone(self.model)

    def test_predict_proba_is_available(self):
        self.assertTrue(callable(self.model.predict_proba))

    def test_actual_class_mapping(self):
        self.assertEqual(self.model.classes_.tolist(), [0, 1, 2])

    def test_positive_example(self):
        self.assert_prediction(real_svm.predict_svm("Sản phẩm này rất tốt."))

    def test_negative_example(self):
        self.assert_prediction(real_svm.predict_svm("Tôi rất thất vọng vì sản phẩm quá tệ."))

    def test_neutral_example(self):
        self.assert_prediction(real_svm.predict_svm("Sản phẩm này bình thường."))

    def test_vietnamese_unicode_and_slang(self):
        self.assert_prediction(real_svm.predict_svm("Tôi ko thik sản phẩm này :( Quááá chán!"))

    def test_probabilities_equal_raw_artifact_inference_without_rounding(self):
        raw = "Sản phẩm này rất tốt."
        expected = self.model.predict_proba([real_svm.clean_text(raw)])[0]
        result = real_svm.predict_svm(raw)
        for column, class_id in enumerate(self.model.classes_):
            self.assertEqual(result[LABELS[int(class_id)]], float(expected[column]))

    def test_label_and_confidence_can_be_derived_by_contract(self):
        result = real_svm.predict_svm("Giao hàng chậm và đóng gói cẩu thả.")
        label = get_predicted_label(result)
        self.assertEqual(result[label], max(result.values()))

    def test_non_string_input_is_rejected(self):
        for value in (None, 123, True, [], {}, b"text"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    real_svm.predict_svm(value)

    def test_empty_and_whitespace_input_is_rejected(self):
        for value in ("", "   ", "\t\r\n"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    real_svm.predict_svm(value)

    def test_text_empty_after_preprocessing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            real_svm.predict_svm("!!!")

    def test_preprocessing_is_called_once_with_raw_input(self):
        raw = "  Camera rất đẹp nhưng pin quá yếu!  "
        with patch.object(real_svm, "clean_text", wraps=real_svm.clean_text) as clean:
            result = real_svm.predict_svm(raw)
        self.assert_prediction(result)
        clean.assert_called_once_with(raw)

    def test_demo_details_do_not_preprocess_a_second_time(self):
        with patch.object(real_svm, "clean_text", wraps=real_svm.clean_text) as clean:
            details = real_svm.predict_svm_with_details("Sản phẩm này rất tốt.")
        self.assertEqual(details["cleaned_text"], "sản phẩm này rất tốt")
        self.assert_prediction(details["probabilities"])
        clean.assert_called_once_with("Sản phẩm này rất tốt.")

    def test_model_is_loaded_only_once_for_multiple_predictions(self):
        import joblib
        real_svm.load_model.cache_clear()
        try:
            with patch.object(joblib, "load", wraps=joblib.load) as load:
                first = real_svm.predict_svm("Sản phẩm tốt.")
                second = real_svm.predict_svm("Sản phẩm tệ.")
            self.assert_prediction(first)
            self.assert_prediction(second)
            self.assertEqual(load.call_count, 1)
        finally:
            real_svm.load_model.cache_clear()


class SVMClassMappingTests(unittest.TestCase):
    def predict_with_stub(self, classes, row):
        with patch.object(real_svm, "load_model", return_value=PipelineStub(classes, row)):
            return real_svm.predict_svm("Văn bản thử mapping.")

    def test_reordered_numeric_classes(self):
        result = self.predict_with_stub([2, 0, 1], [0.7, 0.1, 0.2])
        self.assertEqual(result, {"negative": 0.1, "neutral": 0.2, "positive": 0.7})

    def test_titlecase_classes_use_existing_adapter(self):
        with patch.object(real_svm, "normalize_svm_output", wraps=normalize_svm_output) as adapter:
            result = self.predict_with_stub(["Positive", "Negative", "Neutral"], [0.7, 0.1, 0.2])
        self.assertEqual(result, {"negative": 0.1, "neutral": 0.2, "positive": 0.7})
        adapter.assert_called_once()

    def test_lowercase_classes(self):
        result = self.predict_with_stub(["neutral", "positive", "negative"], [0.2, 0.7, 0.1])
        self.assertEqual(result, {"negative": 0.1, "neutral": 0.2, "positive": 0.7})

    def test_unexpected_or_duplicate_classes_are_rejected(self):
        for classes in ([0, 1, 3], [0, 0, 2], [True, 1, 2], [0, 1], [0, 1, 2.0], ["Negative", "neutral", "Positive"]):
            with self.subTest(classes=classes):
                with self.assertRaisesRegex(ValueError, "classes_"):
                    self.predict_with_stub(classes, [0.1, 0.2, 0.7])

    def test_probability_column_count_must_match_classes(self):
        with self.assertRaises(ValueError):
            self.predict_with_stub([0, 1, 2], [0.1, 0.9])

    def test_invalid_probability_is_rejected(self):
        for row in ([0.1, 0.2, float("nan")], [-0.1, 0.2, 0.9], [0.1, 0.2, 0.6]):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    self.predict_with_stub([0, 1, 2], row)


class SVMLoadErrorTests(unittest.TestCase):
    def tearDown(self):
        real_svm.load_model.cache_clear()

    def test_missing_model_has_clear_path_error(self):
        real_svm.load_model.cache_clear()
        absent = real_svm.MODEL_PATH.with_name("missing_svm_for_test.joblib")
        self.assertFalse(absent.exists())
        with patch.object(real_svm, "MODEL_PATH", absent):
            with self.assertRaisesRegex(FileNotFoundError, "missing_svm_for_test"):
                real_svm.load_model()

    def test_missing_dependency_has_clear_installation_error(self):
        real_svm.load_model.cache_clear()
        with patch.dict("sys.modules", {"joblib": None}):
            with self.assertRaisesRegex(ImportError, "requirements"):
                real_svm.load_model()

    def test_pipeline_without_predict_proba_is_rejected(self):
        import joblib
        real_svm.load_model.cache_clear()
        with patch.object(joblib, "load", return_value=SimpleNamespace(classes_=[0, 1, 2])):
            with self.assertRaisesRegex(TypeError, "predict_proba"):
                real_svm.load_model()


class HybridIntegrationTests(unittest.TestCase):
    """Real SVM + Mock Transformer: chỉ kiểm tra wiring, không so sánh model."""

    def test_single_sentence_uses_injected_real_predictor(self):
        text = "Sản phẩm này rất tốt."
        with patch.object(real_svm, "predict_svm", wraps=real_svm.predict_svm) as predictor:
            result = analyze_sentence(text, svm_predictor=predictor)
        self.assertEqual(result["svm"]["probabilities"], real_svm.predict_svm(text))
        self.assertNotEqual(result["svm"]["probabilities"], mock_svm(text))
        predictor.assert_called_once_with(text)

    def test_paragraph_preprocesses_each_raw_sentence_once(self):
        text = "Camera rất đẹp. Giao hàng chậm!"
        with patch.object(real_svm, "clean_text", wraps=real_svm.clean_text) as clean:
            results = analyze_sentences(text, svm_predictor=real_svm.predict_svm)
        self.assertEqual([call.args[0] for call in clean.call_args_list], ["Camera rất đẹp.", "Giao hàng chậm!"])
        self.assertEqual([result["sentence_index"] for result in results], [0, 1])
        for result in results:
            validate_prediction_output(result["svm"]["probabilities"])

    def test_document_forwards_real_predictor_and_alpha(self):
        text = "Camera rất đẹp. Giao hàng chậm!"
        with patch.object(real_svm, "predict_svm", wraps=real_svm.predict_svm) as predictor:
            result = analyze_document(text, alpha=0.3, svm_predictor=predictor)
        self.assertEqual(predictor.call_count, 2)
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["sentences"][0]["ensemble"]["weights"], {"svm": 0.3, "transformer": 0.7})
        self.assertEqual(set(result["document_results"]), {"svm", "transformer", "ensemble"})
        for model in result["document_results"].values():
            for method in ("mean_probability", "confidence_weighted"):
                validate_prediction_output(model[method]["probabilities"])

    def test_default_predictor_still_uses_mock(self):
        text = "Sản phẩm này rất tốt."
        self.assertEqual(analyze_sentence(text)["svm"]["probabilities"], mock_svm(text))

    def test_empty_document_does_not_invoke_predictor(self):
        def unused_predictor(text):
            raise AssertionError("Không được gọi predictor cho document rỗng.")
        self.assertEqual(analyze_document("", svm_predictor=unused_predictor)["document_results"], {})

    def test_invalid_predictor_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "svm_predictor"):
            analyze_sentence("Sản phẩm tốt.", svm_predictor=123)


if __name__ == "__main__":
    unittest.main()
