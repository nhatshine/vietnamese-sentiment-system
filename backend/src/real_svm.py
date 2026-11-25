"""Inference TF-IDF + SVM thật, lazy load và preprocessing gốc đúng một lần."""

from functools import lru_cache
from numbers import Integral
from pathlib import Path

from .adapters import normalize_svm_output
from .config import ID_TO_LABEL, LABELS
from .contracts import validate_prediction_output
from .preprocess import clean_text


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svm" / "svm_pipeline.joblib"


def _class_labels(classes) -> list[str]:
    """Đọc classes_ theo thứ tự cột thật, chỉ nhận ba encoding đã thống nhất."""
    try:
        values = list(classes)
    except TypeError as error:
        raise ValueError("classes_ của SVM phải là danh sách ba class.") from error

    if len(values) != len(LABELS):
        raise ValueError(f"classes_ của SVM phải có đúng ba class; nhận được {values!r}.")
    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in values):
        if any(int(value) not in ID_TO_LABEL for value in values):
            raise ValueError(f"classes_ của SVM có class ID không được hỗ trợ: {values!r}.")
        labels = [ID_TO_LABEL[int(value)] for value in values]
    elif all(isinstance(value, str) for value in values):
        title_to_label = {label.title(): label for label in LABELS}
        if set(values) == set(LABELS):
            labels = values
        elif set(values) == set(title_to_label):
            labels = [title_to_label[value] for value in values]
        else:
            raise ValueError(f"classes_ của SVM có nhãn khác chuẩn: {values!r}.")
    else:
        raise ValueError(f"classes_ của SVM có kiểu hoặc encoding không được hỗ trợ: {values!r}.")

    if set(labels) != set(LABELS):
        raise ValueError(f"classes_ của SVM thiếu hoặc trùng nhãn: {values!r}.")
    return labels


@lru_cache(maxsize=1)
def load_model():
    """Load artifact tin cậy một lần mỗi process; không load lúc import module."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model SVM: {MODEL_PATH}")
    try:
        from joblib import load
        model = load(MODEL_PATH)
    except ImportError as error:
        raise ImportError(
            "Real SVM cần joblib, scikit-learn và NumPy tương thích artifact. "
            "Cài bằng: python -m pip install -r backend/requirements.txt"
        ) from error

    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Artifact SVM phải cung cấp predict_proba().")
    if not hasattr(model, "classes_"):
        raise TypeError("Artifact SVM phải cung cấp classes_ để ánh xạ probability.")
    _class_labels(model.classes_)
    return model


def _normalize_probability_row(classes, row) -> dict[str, float]:
    labels = _class_labels(classes)
    if len(row) != len(labels):
        raise ValueError("Số cột predict_proba() không khớp classes_ của SVM.")

    class_values = list(classes)
    if all(isinstance(value, str) and value in {label.title() for label in LABELS} for value in class_values):
        result = normalize_svm_output(dict(zip(class_values, row)))
    else:
        result = {label: float(value) for label, value in zip(labels, row)}
    result = {label: result[label] for label in LABELS}
    validate_prediction_output(result)
    return result


def predict_svm_with_details(text: str) -> dict:
    """Inference một lần và trả thêm cleaned text để demo không phải clean lại."""
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi (str).")
    if not text.strip():
        raise ValueError("text không được rỗng hoặc chỉ chứa khoảng trắng.")

    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        raise ValueError("Văn bản rỗng sau preprocessing; không thực hiện inference.")
    model = load_model()
    rows = model.predict_proba([cleaned_text])
    if len(rows) != 1:
        raise ValueError("predict_proba() phải trả đúng một hàng cho một text.")
    probabilities = _normalize_probability_row(model.classes_, rows[0])
    return {"text": text, "cleaned_text": cleaned_text, "probabilities": probabilities}


def predict_svm(text: str) -> dict[str, float]:
    """Nhận raw text, trả probability lowercase bằng artifact SVM thật."""
    return predict_svm_with_details(text)["probabilities"]


def predict_svm_cleaned_batch(texts: list[str], *, batch_size: int = 256, progress=None) -> list[dict[str, float]]:
    """Inference CSV đã preprocessing; không gọi clean_text hoặc fit lại TF-IDF."""
    if not isinstance(texts, list):
        raise TypeError("texts phải là list[str] đã preprocessing.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương.")
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Mỗi cleaned text phải là str.")
        if not text.strip():
            raise ValueError("Cleaned text không được rỗng.")
    if not texts:
        return []
    model = load_model()
    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        rows = model.predict_proba(chunk)
        if len(rows) != len(chunk):
            raise ValueError("SVM trả sai số mẫu trong batch.")
        results.extend(_normalize_probability_row(model.classes_, row) for row in rows)
        if progress is not None:
            progress(len(results), len(texts))
    return results
