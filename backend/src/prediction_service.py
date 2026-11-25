"""Production service: real raw predictors, frozen selection, reusable presentation."""

from copy import deepcopy

from .deployment_config import DEPLOYMENT_PATH, load_deployment_config, validate_deployment_config
from .document_aggregator import analyze_document
from .result_selector import SUPPORTED_METHODS, select_result


class PredictionService:
    def __init__(self, config_path=DEPLOYMENT_PATH, *, svm_predictor=None, transformer_predictor=None):
        self._config = deepcopy(load_deployment_config(config_path))
        validate_deployment_config(self._config)
        # Importing predictors does not load checkpoints; their existing caches own lifetime.
        from .real_svm import predict_svm
        from .real_transformer import predict_transformer
        if self._config.get("artifact_profile") == "transformer_epoch21":
            from .epoch21_deployment import predict_transformer_epoch21
            predict_transformer = predict_transformer_epoch21
        for predictor in (svm_predictor, transformer_predictor):
            if predictor is not None and not callable(predictor):
                raise TypeError("Predictor phải là callable hoặc None (real mặc định).")
        self._svm_predictor = predict_svm if svm_predictor is None else svm_predictor
        self._transformer_predictor = predict_transformer if transformer_predictor is None else transformer_predictor

    @property
    def config(self):
        return deepcopy(self._config)

    @staticmethod
    def _guard(predictor, method):
        def predict(sentence):
            try:
                return predictor(sentence)
            except (TypeError, ValueError) as error:
                raise type(error)(f"{method}: không phân tích được câu {sentence!r}: {error}") from error
        return predict

    def analyze(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("text phải là chuỗi (str).")
        if not text.strip():
            raise ValueError("text không được rỗng hoặc chỉ whitespace.")
        config = self._config
        selected = config["selected_method"]
        alpha = config["selected_alpha"]
        document = analyze_document(
            text, alpha=alpha,
            svm_predictor=self._guard(self._svm_predictor, "svm"),
            transformer_predictor=self._guard(self._transformer_predictor, "transformer"),
        )
        if not document["sentences"]:
            raise ValueError("Không có câu để phân tích; không âm thầm bỏ input.")
        sentences = []
        for sentence in document["sentences"]:
            comparison = {method: deepcopy(sentence[method]) for method in SUPPORTED_METHODS}
            sentences.append({"sentence_index": sentence["sentence_index"], "sentence": sentence["sentence"],
                              "final_result": select_result(comparison, selected), "comparison": comparison})
        multiple = len(sentences) > 1
        aggregation = config["document_aggregation_default"]
        if multiple:
            # Add method metadata to comparison only; keep original document_results intact.
            comparison = {method: {"method": method, **deepcopy(document["document_results"][method][aggregation])}
                          for method in SUPPORTED_METHODS}
        else:
            # Exact single-sentence prediction, including original confidence/weights.
            comparison = deepcopy(sentences[0]["comparison"])
        return {
            "input_text": text, "sentence_count": len(sentences), "sentences": sentences,
            "document_results": deepcopy(document["document_results"]),
            "final_result": select_result(comparison, selected),
            "selection_info": {"selected_method": selected, "selected_alpha": alpha, "selected_by": "validation_macro_f1"},
            "aggregation_info": {"method": aggregation, "applied_to_final": multiple, "validated_optimal": False,
                                 "policy": "deployment_demo_default_not_document_benchmark"},
            "comparison": comparison,
            "display_info": {"default_view": config["default_view"], "comparison_view_available": config["comparison_view_available"]},
            "research_info": {"validation_macro_f1": deepcopy(config["validation_macro_f1"]),
                              "test_macro_f1": deepcopy(config["test_macro_f1"]),
                              "ensemble_weights": {"svm": alpha, "transformer": 1.0 - alpha},
                              "metric_scope": config["metric_scope"],
                              "frozen_selection_verified": config["frozen_selection_verified"]},
        }


def present_result(response: dict, view: str = "user") -> dict:
    """Pure presentation of a retained canonical response; never calls a predictor."""
    if not isinstance(view, str) or view not in ("user", "research"):
        raise ValueError("view phải là 'user' hoặc 'research'.")
    if not isinstance(response, dict):
        raise TypeError("response phải là dictionary từ PredictionService.analyze().")
    if view == "research":
        return deepcopy(response)
    result = {key: deepcopy(response[key]) for key in ("input_text", "sentence_count", "final_result", "selection_info", "aggregation_info")}
    result["sentences"] = [{key: deepcopy(sentence[key]) for key in ("sentence_index", "sentence", "final_result")}
                           for sentence in response["sentences"]]
    return result
