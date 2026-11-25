"""Phân tích từng câu raw; mặc định mock, cho phép injection cả hai predictor."""

from typing import Callable, Optional

from .contracts import PredictionOutput, get_predicted_label
from .ensemble import weighted_ensemble
from .mock_models import predict_svm, predict_transformer
from .sentence_splitter import split_sentences


Predictor = Callable[[str], PredictionOutput]


def _format_model_result(method: str, probabilities: PredictionOutput) -> dict:
    """Validate probability qua utility nhãn và tạo output mới, không round."""
    label = get_predicted_label(probabilities)
    return {
        "method": method,
        "label": label,
        "confidence": probabilities[label],
        "probabilities": probabilities.copy(),
    }


def analyze_sentence(
    sentence: str,
    alpha: float = 0.5,
    *,
    svm_predictor: Optional[Predictor] = None,
    transformer_predictor: Optional[Predictor] = None,
) -> dict:
    """Trả cả ba kết quả cho một câu raw, giữ nguyên nội dung và dấu câu.

    Không gọi clean_text(): prediction adapter/function của model thật sau
    này chịu trách nhiệm preprocessing. Alpha được validate bởi Ensemble.
    Không chọn BEST_METHOD hoặc tính sentiment toàn đoạn. Không truyền
    predictor thì tiếp tục dùng mock tương ứng để tương thích các bước cũ.
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence phải là chuỗi (str).")
    if not sentence.strip():
        raise ValueError("sentence không được rỗng hoặc chỉ chứa khoảng trắng.")

    if svm_predictor is not None and not callable(svm_predictor):
        raise TypeError("svm_predictor phải là callable hoặc None.")
    if transformer_predictor is not None and not callable(transformer_predictor):
        raise TypeError("transformer_predictor phải là callable hoặc None.")
    selected_svm = predict_svm if svm_predictor is None else svm_predictor
    selected_transformer = predict_transformer if transformer_predictor is None else transformer_predictor
    svm_probs = selected_svm(sentence)
    transformer_probs = selected_transformer(sentence)
    ensemble_result = weighted_ensemble(svm_probs, transformer_probs, alpha=alpha)

    return {
        "sentence": sentence,
        "svm": _format_model_result("svm", svm_probs),
        "transformer": _format_model_result("transformer", transformer_probs),
        "ensemble": ensemble_result,
    }


def analyze_sentences(
    text: str,
    alpha: float = 0.5,
    *,
    svm_predictor: Optional[Predictor] = None,
    transformer_predictor: Optional[Predictor] = None,
) -> list[dict]:
    """Tách raw paragraph rồi phân tích từng câu theo thứ tự, index từ 0.

    Paragraph rỗng trả []; kiểu input được kiểm tra bởi splitter. Chỉ trả
    kết quả từng câu, không voting, average hoặc aggregation toàn đoạn.
    """
    return [
        {"sentence_index": index, **analyze_sentence(
            sentence, alpha=alpha, svm_predictor=svm_predictor,
            transformer_predictor=transformer_predictor,
        )}
        for index, sentence in enumerate(split_sentences(text))
    ]
