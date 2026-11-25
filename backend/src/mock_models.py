"""Mock theo luật từ khóa để kiểm tra interface; không phải model NLP thật.

Đếm số lần xuất hiện từ/cụm từ tích cực và tiêu cực, không phân biệt hoa
thường. Nếu hai nhóm bằng nhau hoặc không có từ khóa, chọn nhánh neutral.
Không xử lý phủ định, ngữ cảnh hoặc mức độ cảm xúc.
"""

import re

from .config import LABELS
from .contracts import PredictionOutput, validate_prediction_output


_POSITIVE_WORDS = ("tốt", "đẹp", "tuyệt", "hài lòng", "thích", "ổn", "nhanh")
_NEGATIVE_WORDS = ("tệ", "xấu", "thất vọng", "chán", "kém", "lỗi", "chậm")
_POSITIVE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _POSITIVE_WORDS) + r")\b"
)
_NEGATIVE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _NEGATIVE_WORDS) + r")\b"
)


def _mock_prediction(
    text: str,
    positive: tuple[float, float, float],
    negative: tuple[float, float, float],
    neutral: tuple[float, float, float],
) -> PredictionOutput:
    """Chọn bộ xác suất theo từ khóa; các tuple tuân theo thứ tự LABELS."""
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi (str).")
    if not text.strip():
        raise ValueError("text không được rỗng hoặc chỉ chứa khoảng trắng.")

    normalized_text = " ".join(text.casefold().split())
    positive_count = len(_POSITIVE_PATTERN.findall(normalized_text))
    negative_count = len(_NEGATIVE_PATTERN.findall(normalized_text))

    if positive_count > negative_count:
        probabilities = positive
    elif negative_count > positive_count:
        probabilities = negative
    else:
        probabilities = neutral

    return dict(zip(LABELS, probabilities))


def predict_svm(text: str) -> dict:
    """Trả xác suất SVM giả lập, deterministic, chưa tải model thật."""
    result = _mock_prediction(
        text,
        positive=(0.08, 0.17, 0.75),
        negative=(0.75, 0.17, 0.08),
        neutral=(0.20, 0.60, 0.20),
    )
    validate_prediction_output(result)
    return result


def predict_transformer(text: str) -> dict:
    """Trả xác suất Transformer giả lập, deterministic, chưa tải model thật."""
    result = _mock_prediction(
        text,
        positive=(0.05, 0.10, 0.85),
        negative=(0.85, 0.10, 0.05),
        neutral=(0.15, 0.70, 0.15),
    )
    validate_prediction_output(result)
    return result
