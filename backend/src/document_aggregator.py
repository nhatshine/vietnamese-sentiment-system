"""Tổng hợp kết quả theo câu, độc lập với logic model và sentence analyzer."""

import math
from typing import Optional

from .config import LABELS
from .contracts import (
    PROBABILITY_SUM_TOLERANCE,
    get_predicted_label,
    validate_prediction_output,
)
from .sentence_analyzer import Predictor, analyze_sentences


MODEL_KEYS = ("svm", "transformer", "ensemble")


def _validated_predictions(sentence_results: list[dict], model_key: str) -> list[dict]:
    """Kiểm tra cấu trúc và tính nhất quán prediction, chỉ đọc input."""
    if not isinstance(sentence_results, list):
        raise TypeError("sentence_results phải là list.")
    if not sentence_results:
        raise ValueError("sentence_results không được rỗng khi gọi aggregation trực tiếp.")
    if not isinstance(model_key, str) or model_key not in MODEL_KEYS:
        raise ValueError(f"model_key phải thuộc {MODEL_KEYS}; nhận được {model_key!r}.")

    predictions = []
    for index, sentence_result in enumerate(sentence_results):
        if not isinstance(sentence_result, dict):
            raise TypeError(f"sentence_results[{index}] phải là dictionary.")
        if model_key not in sentence_result:
            raise ValueError(f"Câu tại index {index} thiếu model key '{model_key}'.")
        prediction = sentence_result[model_key]
        if not isinstance(prediction, dict):
            raise TypeError(f"Prediction '{model_key}' tại index {index} phải là dictionary.")
        missing = [field for field in ("label", "confidence", "probabilities") if field not in prediction]
        if missing:
            raise ValueError(f"Prediction '{model_key}' tại index {index} thiếu field {missing}.")

        probabilities = prediction["probabilities"]
        validate_prediction_output(probabilities)
        label = prediction["label"]
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError(f"Label của câu tại index {index} phải thuộc {LABELS}.")
        confidence = prediction["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"Confidence của câu tại index {index} phải là int/float, không nhận bool.")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence của câu tại index {index} phải hữu hạn trong [0, 1].")
        if probabilities[label] != max(probabilities.values()):
            raise ValueError(f"Label của câu tại index {index} phải có probability lớn nhất.")
        if not math.isclose(
            confidence, probabilities[label], rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE
        ):
            raise ValueError(f"Confidence của câu tại index {index} không khớp probability của label.")
        predictions.append(prediction)

    return predictions


def _mean_probabilities(predictions: list[dict]) -> dict:
    """Tính trung bình theo tên nhãn từ danh sách đã validate và không rỗng."""
    return {
        label: math.fsum(prediction["probabilities"][label] for prediction in predictions) / len(predictions)
        for label in LABELS
    }


def _probability_result(aggregation_method: str, model_key: str, probabilities: dict) -> dict:
    """Tạo metadata chung cho hai phương án tổng hợp probability."""
    validate_prediction_output(probabilities)
    label = get_predicted_label(probabilities)
    return {
        "aggregation_method": aggregation_method,
        "model": model_key,
        "label": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
    }


def aggregate_majority_vote(sentence_results: list[dict], model_key: str) -> dict:
    """Bỏ phiếu theo câu; hòa phiếu dùng mean probability rồi thứ tự LABELS.

    Confidence là tỷ lệ phiếu của nhãn thắng, không phải probability model.
    """
    predictions = _validated_predictions(sentence_results, model_key)
    vote_counts = {
        label: sum(prediction["label"] == label for prediction in predictions)
        for label in LABELS
    }
    most_votes = max(vote_counts.values())
    tied_labels = [label for label in LABELS if vote_counts[label] == most_votes]
    mean_probabilities = _mean_probabilities(predictions)
    label = max(tied_labels, key=mean_probabilities.__getitem__)
    vote_distribution = {
        label: vote_counts[label] / len(predictions) for label in LABELS
    }
    validate_prediction_output(vote_distribution)
    return {
        "aggregation_method": "majority_vote",
        "model": model_key,
        "label": label,
        "confidence": vote_distribution[label],
        "vote_counts": vote_counts,
        "vote_distribution": vote_distribution,
    }


def aggregate_mean_probability(sentence_results: list[dict], model_key: str) -> dict:
    """Trung bình probability của các câu, mỗi câu có trọng số bằng nhau."""
    predictions = _validated_predictions(sentence_results, model_key)
    return _probability_result("mean_probability", model_key, _mean_probabilities(predictions))


def aggregate_confidence_weighted(sentence_results: list[dict], model_key: str) -> dict:
    """Lấy max probability mỗi câu làm weight, chia tổng weights khi kết hợp."""
    predictions = _validated_predictions(sentence_results, model_key)
    sentence_weights = [max(prediction["probabilities"].values()) for prediction in predictions]
    # Các phân phối hợp lệ, không rỗng bảo đảm mỗi weight > 0.
    total_weight = math.fsum(sentence_weights)
    probabilities = {
        label: math.fsum(
            weight * prediction["probabilities"][label]
            for weight, prediction in zip(sentence_weights, predictions)
        ) / total_weight
        for label in LABELS
    }
    result = _probability_result("confidence_weighted", model_key, probabilities)
    result["sentence_weights"] = sentence_weights
    return result


def aggregate_model_document(sentence_results: list[dict], model_key: str) -> dict:
    """Trả đủ ba chiến lược aggregation cho một model, không chọn chiến lược."""
    return {
        "model": model_key,
        "majority_vote": aggregate_majority_vote(sentence_results, model_key),
        "mean_probability": aggregate_mean_probability(sentence_results, model_key),
        "confidence_weighted": aggregate_confidence_weighted(sentence_results, model_key),
    }


def aggregate_all_models(sentence_results: list[dict]) -> dict:
    """Tổng hợp riêng SVM, Transformer và Ensemble từ cùng sentence results."""
    return {
        model_key: aggregate_model_document(sentence_results, model_key)
        for model_key in MODEL_KEYS
    }


def analyze_document(
    text: str,
    alpha: float = 0.5,
    *,
    svm_predictor: Optional[Predictor] = None,
    transformer_predictor: Optional[Predictor] = None,
) -> dict:
    """Chạy pipeline Bước 5 rồi aggregation; giữ raw text và sentence results.

    Document rỗng có document_results {}, không gọi aggregation trên list rỗng.
    Không trực tiếp gọi model, preprocessing hoặc lựa chọn phương án.
    """
    sentences = analyze_sentences(
        text, alpha=alpha, svm_predictor=svm_predictor,
        transformer_predictor=transformer_predictor,
    )
    return {
        "text": text,
        "sentence_count": len(sentences),
        "sentences": sentences,
        "document_results": aggregate_all_models(sentences) if sentences else {},
    }
