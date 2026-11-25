"""Transformer thật, tokenizer gốc, numerical consistency và real pipeline."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from backend.models.transformer import predict_transformer as original
from backend.models.transformer import tokenizer
from backend.src import real_transformer
from backend.src.config import LABELS
from backend.src.contracts import validate_prediction_output
from backend.src.document_aggregator import analyze_document
from backend.src.mock_models import predict_svm as mock_svm
from backend.src.real_svm import predict_svm as real_svm
from backend.src.sentence_analyzer import analyze_sentence, analyze_sentences


SAMPLES = (
    "Sản phẩm này rất tốt.",
    "Tôi rất thất vọng vì sản phẩm quá tệ.",
    "Sản phẩm này bình thường.",
    "Camera rất đẹp nhưng pin quá yếu.",
    "Giao hàng chậm và đóng gói cẩu thả.",
)


class TransformerLoadingTests(unittest.TestCase):
    def tearDown(self):
        real_transformer._load_cached_model.cache_clear()

    def test_checkpoint_loads_strictly_on_cpu(self):
        resources = real_transformer.load_model()
        self.assertEqual(str(resources.device), "cpu")
        self.assertEqual(next(resources.model.parameters()).device.type, "cpu")
        self.assertEqual(len(resources.model.state_dict()), 38)

    def test_architecture_and_learned_position_embeddings_match_artifact(self):
        resources = real_transformer.load_model()
        model = resources.model
        self.assertIsInstance(model.position_embedding, torch.nn.Embedding)
        self.assertEqual(tuple(model.position_embedding.weight.shape), (128, 128))
        self.assertTrue(model.position_embedding.weight.requires_grad)
        self.assertEqual(tuple(model.token_embedding.weight.shape), (9169, 128))
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(model.blocks[0].attention.num_heads, 4)
        self.assertEqual(model.blocks[0].ffn[0].out_features, 256)

    def test_model_and_dropout_are_in_eval_mode(self):
        model = real_transformer.load_model().model
        self.assertFalse(model.training)
        self.assertTrue(all(not layer.training for layer in model.modules()))

    def test_checkpoint_is_loaded_once_with_safe_cpu_options(self):
        real_transformer._load_cached_model.cache_clear()
        with patch.object(torch, "load", wraps=torch.load) as load:
            real_transformer.predict_transformer(SAMPLES[0])
            real_transformer.predict_transformer(SAMPLES[1])
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertIs(load.call_args.kwargs["weights_only"], True)

    def test_missing_artifacts_list_exact_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(real_transformer, "MODEL_DIR", Path(temporary)):
                with self.assertRaisesRegex(FileNotFoundError, "best_transformer.pth"):
                    real_transformer.load_model()

    def test_missing_torch_has_project_environment_guidance(self):
        real_transformer._load_cached_model.cache_clear()
        with patch.dict("sys.modules", {"torch": None}):
            with self.assertRaisesRegex(ImportError, "requirements"):
                real_transformer.load_model()

    def test_preprocessing_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preprocess.py"
            path.write_text("# wrong preprocessing", encoding="utf-8")
            with patch.object(real_transformer, "PREPROCESS_PATH", path):
                with self.assertRaisesRegex(ValueError, "preprocess"):
                    real_transformer.load_model()

    def test_cpu_is_available_without_cuda(self):
        self.assertEqual(str(real_transformer.load_model().device), "cpu")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                real_transformer.load_model("cuda")


class TransformerTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resources = real_transformer.load_model()

    def test_vocab_and_special_ids(self):
        vocab = self.resources.vocab
        self.assertEqual(len(vocab), 9169)
        self.assertEqual([vocab[name] for name in ("<pad>", "<unk>", "<cls>")], [0, 1, 2])
        self.assertEqual(set(vocab.values()), set(range(len(vocab))))

    def test_cls_is_first_token(self):
        ids, mask = tokenizer.encode_text("Tốt.", self.resources.vocab, 128)
        self.assertEqual(ids[0], 2)
        self.assertEqual(mask[0], 1)
        self.assertEqual(ids[1], self.resources.vocab["tốt"])

    def test_padding_and_mask(self):
        ids, mask = tokenizer.encode_text("rất tốt", self.resources.vocab, 128)
        self.assertEqual(len(ids), 128)
        self.assertEqual(len(mask), 128)
        self.assertEqual(mask[:3], [1, 1, 1])
        self.assertEqual(mask[3:], [0] * 125)
        self.assertEqual(ids[3:], [0] * 125)

    def test_unknown_token(self):
        ids, _ = tokenizer.encode_text("qzxvtokenunknown", self.resources.vocab, 128)
        self.assertEqual(ids[:2], [2, 1])

    def test_truncation_reserves_one_position_for_cls(self):
        ids, mask = tokenizer.encode_text(" ".join(["tốt"] * 200), self.resources.vocab, 128)
        self.assertEqual(ids, [2] + [self.resources.vocab["tốt"]] * 127)
        self.assertEqual(mask, [1] * 128)

    def test_model_receives_training_shapes_dtypes_and_inference_mode(self):
        resources = real_transformer.load_model()
        states = []
        forward = resources.model.forward

        def observed_forward(ids, mask):
            states.append((tuple(ids.shape), ids.dtype, tuple(mask.shape), mask.dtype,
                           torch.is_inference_mode_enabled()))
            return forward(ids, mask)

        with patch.object(resources.model, "forward", side_effect=observed_forward):
            result = real_transformer.predict_transformer("Sản phẩm tốt.")
        validate_prediction_output(result)
        self.assertEqual(states, [((1, 128), torch.long, (1, 128), torch.long, True)])


class TransformerPredictionTests(unittest.TestCase):
    def assert_prediction(self, result):
        self.assertEqual(list(result), LABELS)
        self.assertTrue(all(type(value) is float for value in result.values()))
        self.assertIsNone(validate_prediction_output(result))
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        json.dumps(result, allow_nan=False)

    def test_five_examples_produce_real_probabilities(self):
        for text in SAMPLES:
            with self.subTest(text=text):
                self.assert_prediction(real_transformer.predict_transformer(text))

    def test_unicode_slang_and_negation(self):
        self.assert_prediction(real_transformer.predict_transformer("Tôi ko thik! Quááá chán :("))

    def test_non_string_input_is_rejected(self):
        for text in (None, 123, True, [], {}, b"text"):
            with self.subTest(text=text):
                with self.assertRaises(TypeError):
                    real_transformer.predict_transformer(text)

    def test_empty_or_whitespace_input_is_rejected(self):
        for text in ("", " \t\r\n "):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    real_transformer.predict_transformer(text)

    def test_label_map_is_read_from_artifact(self):
        resources = real_transformer.load_model()
        self.assertEqual(resources.labels, ("negative", "neutral", "positive"))

    def test_label_map_validation_and_explicit_column_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "label_map.json"
            path.write_text(json.dumps({"0": "positive", "1": "negative", "2": "neutral"}), encoding="utf-8")
            self.assertEqual(real_transformer._read_labels(path, 3), ("positive", "negative", "neutral"))
            path.write_text(json.dumps({"0": "positive", "1": "positive", "2": "neutral"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                real_transformer._read_labels(path, 3)

    def test_tokenizer_calls_shared_preprocessing_once(self):
        raw = "  Camera rất đẹp nhưng pin quá yếu!  "
        with patch.object(tokenizer, "clean_text", wraps=tokenizer.clean_text) as clean:
            result = real_transformer.predict_transformer(raw)
        self.assert_prediction(result)
        clean.assert_called_once_with(raw)

    def test_deterministic_cpu_inference(self):
        self.assertEqual(real_transformer.predict_transformer(SAMPLES[0]),
                         real_transformer.predict_transformer(SAMPLES[0]))

    def test_numerical_consistency_with_supplied_predictor_on_cpu(self):
        reference = original.TransformerPredictor(real_transformer.MODEL_DIR, device="cpu")
        for text in SAMPLES + ("Tôi ko thik sản phẩm này :( Quááá chán!",):
            with self.subTest(text=text):
                expected = reference.predict(text)
                actual = real_transformer.predict_transformer(text)
                for label in LABELS:
                    self.assertLessEqual(abs(actual[label] - expected[label]), 1e-7)


class BothRealModelsIntegrationTests(unittest.TestCase):
    def test_transformer_only_injection_keeps_svm_mock_by_default(self):
        text = SAMPLES[0]
        result = analyze_sentence(text, transformer_predictor=real_transformer.predict_transformer)
        self.assertEqual(result["svm"]["probabilities"], mock_svm(text))
        self.assertEqual(result["transformer"]["probabilities"], real_transformer.predict_transformer(text))

    def test_both_real_outputs_and_ensemble_formula(self):
        result = analyze_sentence(SAMPLES[0], alpha=0.5,
                                  svm_predictor=real_svm,
                                  transformer_predictor=real_transformer.predict_transformer)
        for label in LABELS:
            expected = 0.5 * result["svm"]["probabilities"][label] + 0.5 * result["transformer"]["probabilities"][label]
            self.assertEqual(result["ensemble"]["probabilities"][label], expected)
        self.assertEqual(result["svm"]["probabilities"], real_svm(SAMPLES[0]))
        self.assertEqual(result["transformer"]["probabilities"], real_transformer.predict_transformer(SAMPLES[0]))

    def test_each_model_called_once_per_raw_sentence_in_order(self):
        raw_sentences = ["Camera rất đẹp.", "Nhưng pin quá yếu.", "Giao hàng rất nhanh!"]
        with patch("backend.src.real_svm.clean_text", wraps=tokenizer.clean_text) as svm_clean:
            with patch.object(tokenizer, "clean_text", wraps=tokenizer.clean_text) as transformer_clean:
                with patch("backend.src.real_svm.predict_svm", wraps=real_svm) as svm:
                    with patch.object(real_transformer, "predict_transformer", wraps=real_transformer.predict_transformer) as transformer:
                        results = analyze_sentences(" ".join(raw_sentences),
                                                    svm_predictor=svm, transformer_predictor=transformer)
        self.assertEqual([x["sentence"] for x in results], raw_sentences)
        for recorded in (svm.call_args_list, transformer.call_args_list,
                         svm_clean.call_args_list, transformer_clean.call_args_list):
            self.assertEqual([call.args[0] for call in recorded], raw_sentences)

    def test_real_document_preserves_all_three_aggregation_methods(self):
        result = analyze_document("Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!",
                                  svm_predictor=real_svm,
                                  transformer_predictor=real_transformer.predict_transformer)
        self.assertEqual(result["sentence_count"], 3)
        self.assertEqual(set(result["document_results"]), {"svm", "transformer", "ensemble"})
        for model in result["document_results"].values():
            self.assertEqual(set(model), {"model", "majority_vote", "mean_probability", "confidence_weighted"})
            validate_prediction_output(model["majority_vote"]["vote_distribution"])
            validate_prediction_output(model["mean_probability"]["probabilities"])
            validate_prediction_output(model["confidence_weighted"]["probabilities"])

    def test_real_ensemble_boundary_weights(self):
        for alpha, expected_method in ((0, "transformer"), (1, "svm")):
            result = analyze_sentence(SAMPLES[0], alpha=alpha, svm_predictor=real_svm,
                                      transformer_predictor=real_transformer.predict_transformer)
            self.assertEqual(result["ensemble"]["probabilities"], result[expected_method]["probabilities"])

    def test_invalid_transformer_predictor(self):
        with self.assertRaisesRegex(TypeError, "transformer_predictor"):
            analyze_sentence("Sản phẩm tốt.", transformer_predictor=123)

    def test_empty_document_does_not_call_either_predictor(self):
        def unused(text):
            raise AssertionError("Không được gọi predictor cho document rỗng.")
        result = analyze_document("", svm_predictor=unused, transformer_predictor=unused)
        self.assertEqual(result["document_results"], {})


if __name__ == "__main__":
    unittest.main()
