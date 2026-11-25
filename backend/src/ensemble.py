"""Kết hợp probability của hai mô hình cơ sở bằng trọng số cố định."""

from .config import LABELS
from .contracts import get_predicted_label, validate_prediction_output


def weighted_ensemble(
    svm_probs: dict,
    transformer_probs: dict,
    alpha: float = 0.5,
) -> dict:
    """Kết hợp xác suất lowercase; alpha là trọng số SVM trong [0, 1].

    Không tải model, không tuning trọng số, không sửa input hoặc round output.
    Giá trị mặc định 0.5 chỉ phục vụ demo, chưa phải trọng số tối ưu.
    """
    validate_prediction_output(svm_probs)
    validate_prediction_output(transformer_probs)

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha phải là int hoặc float; không chấp nhận bool.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha phải là số hữu hạn trong khoảng [0, 1].")

    alpha = float(alpha)
    final_probs = {
        label: alpha * svm_probs[label] + (1.0 - alpha) * transformer_probs[label]
        for label in LABELS
    }
    validate_prediction_output(final_probs)
    label = get_predicted_label(final_probs)

    return {
        "method": "ensemble",
        "label": label,
        "confidence": final_probs[label],
        "probabilities": final_probs,
        "weights": {"svm": alpha, "transformer": 1.0 - alpha},
    }
