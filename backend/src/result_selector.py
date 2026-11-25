"""Select a frozen method, never the prediction with highest confidence."""

from copy import deepcopy
import math

from .config import LABELS
from .contracts import PROBABILITY_SUM_TOLERANCE, get_predicted_label, validate_prediction_output

SUPPORTED_METHODS = ("svm", "transformer", "ensemble")


def select_result(predictions: dict, selected_method: str) -> dict:
    if not isinstance(selected_method, str) or selected_method not in SUPPORTED_METHODS:
        raise ValueError(f"selected_method phải thuộc {SUPPORTED_METHODS}.")
    if not isinstance(predictions, dict):
        raise TypeError("predictions phải là dictionary của ba phương án.")
    for method in SUPPORTED_METHODS:
        prediction = predictions.get(method)
        if not isinstance(prediction, dict):
            raise ValueError(f"Thiếu prediction hợp lệ của {method}.")
        if not {"label", "confidence", "probabilities"}.issubset(prediction):
            raise ValueError(f"Prediction {method} thiếu label/confidence/probabilities.")
        if "method" in prediction and prediction["method"] != method:
            raise ValueError(f"Prediction {method} có method không nhất quán.")
        probabilities = prediction["probabilities"]
        validate_prediction_output(probabilities)
        label = prediction["label"]
        if not isinstance(label, str) or label not in LABELS or label != get_predicted_label(probabilities):
            raise ValueError(f"Prediction {method} có label không khớp probabilities/tie-break.")
        confidence = prediction["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"Confidence {method} phải là int/float, không nhận bool.")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1 or not math.isclose(
                confidence, probabilities[label], rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError(f"Confidence {method} không khớp probability của label.")
    return deepcopy(predictions[selected_method])
