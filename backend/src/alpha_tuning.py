"""Alpha sweep trên probability đã có; không load model hoặc đọc dataset."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import LABELS, LABEL_TO_ID
from .contracts import PROBABILITY_SUM_TOLERANCE


DEFAULT_ALPHAS = tuple(index / 10 for index in range(11))
TIE_BREAK_RULE = "Exact Macro-F1 ties: prefer endpoints; alpha=0 before alpha=1; otherwise ascending alpha."


def _alpha(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("alpha phải là int/float, không nhận bool.")
    if not 0 <= value <= 1:
        raise ValueError("alpha phải hữu hạn trong [0,1].")
    return float(value)


def _probabilities(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != len(LABELS):
        raise ValueError(f"{name} phải có shape (N,3), N > 0, theo thứ tự LABELS.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} phải chứa int/float, không nhận bool/chuỗi/complex.")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError(f"{name} phải chứa probability hữu hạn trong [0,1].")
    if not np.isclose(array.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_TOLERANCE).all():
        raise ValueError(f"Tổng từng hàng của {name} phải gần bằng 1.")
    return array


def _labels(y_true, count: int) -> np.ndarray:
    labels = np.asarray(y_true)
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError("y_true phải là vector có cùng số mẫu với probability.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("y_true phải chứa class ID nguyên, không nhận bool hoặc float.")
    if not np.isin(labels, list(LABEL_TO_ID.values())).all():
        raise ValueError("y_true chỉ được chứa các class ID 0,1,2.")
    return labels


def compute_ensemble_probabilities(svm_probs, transformer_probs, alpha) -> np.ndarray:
    """Cùng công thức ensemble.py, vector hóa theo hàng, không round/mutate."""
    weight = _alpha(alpha)
    svm = _probabilities(svm_probs, "svm_probs")
    transformer = _probabilities(transformer_probs, "transformer_probs")
    if svm.shape != transformer.shape:
        raise ValueError("Hai model phải có cùng số mẫu và thứ tự cột LABELS.")
    result = weight * svm + (1.0 - weight) * transformer
    return _probabilities(result, "ensemble_probs")


def evaluate_probabilities(y_true, probabilities) -> dict:
    """Metrics từ nhãn argmax toàn tập; macro cố định trên đủ ba class."""
    values = _probabilities(probabilities, "probabilities")
    truth = _labels(y_true, len(values))
    class_ids = np.array([LABEL_TO_ID[label] for label in LABELS])
    predictions = class_ids[values.argmax(axis=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=class_ids, average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def evaluate_alpha(y_true, svm_probs, transformer_probs, alpha) -> dict:
    values = compute_ensemble_probabilities(svm_probs, transformer_probs, alpha)
    return {"alpha": _alpha(alpha), **evaluate_probabilities(y_true, values)}


def tune_alpha(y_true, svm_probs, transformer_probs, alphas=DEFAULT_ALPHAS) -> dict:
    """Chọn alpha bằng Validation Macro-F1; không chọn model/aggregation."""
    grid = [_alpha(alpha) for alpha in alphas]
    if not grid or len(grid) != len(set(grid)):
        raise ValueError("Alpha grid phải không rỗng và không trùng giá trị.")
    scores = [evaluate_alpha(y_true, svm_probs, transformer_probs, alpha) for alpha in grid]
    maximum = max(row["macro_f1"] for row in scores)
    tied = [row for row in scores if row["macro_f1"] == maximum]
    selected = min(tied, key=lambda row: (row["alpha"] not in (0.0, 1.0), row["alpha"]))
    return {
        "scores": scores,
        "selected_alpha": selected["alpha"],
        "selected_validation_macro_f1": selected["macro_f1"],
        "tied_alphas": sorted(row["alpha"] for row in tied),
        "tie_break_rule": TIE_BREAK_RULE,
    }
