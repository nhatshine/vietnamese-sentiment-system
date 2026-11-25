"""Kiểm tra chuẩn đầu ra xác suất; không thực hiện dự đoán hoặc ensemble."""

import math

from .config import LABELS

# Model bàn giao float; validator cũng chấp nhận int theo hợp đồng.
PredictionOutput = dict[str, float]
PROBABILITY_SUM_TOLERANCE = 1e-5


def validate_prediction_output(prediction: object) -> None:
    """Raise TypeError/ValueError nếu đầu ra không đúng chuẩn xác suất.

    Tổng xác suất được phép lệch tối đa 1e-5 so với 1 do làm tròn đầu ra model.
    Hàm chỉ kiểm tra, không sửa dữ liệu hoặc tự chuẩn hóa xác suất.
    """
    if not isinstance(prediction, dict):
        raise TypeError("prediction phải là dictionary chứa xác suất của các nhãn.")

    if set(prediction) != set(LABELS):
        raise ValueError(
            f"prediction phải có chính xác các nhãn {LABELS}; "
            f"nhận được {list(prediction)}."
        )

    for label in LABELS:
        probability = prediction[label]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError(
                f"Xác suất của '{label}' phải là int hoặc float, không nhận bool."
            )
        # Kiểm tra miền trước để số nguyên rất lớn không gây lỗi chuyển sang float.
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError(
                f"Xác suất của '{label}' phải là số hữu hạn trong [0, 1]; "
                f"nhận được {probability!r}."
            )

    total = math.fsum(prediction[label] for label in LABELS)
    if not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE
    ):
        raise ValueError(
            f"Tổng xác suất phải gần bằng 1 "
            f"(sai số tối đa {PROBABILITY_SUM_TOLERANCE}); nhận được {total}."
        )


def get_predicted_label(prediction: PredictionOutput) -> str:
    """Trả nhãn có xác suất cao nhất; nếu hòa, ưu tiên thứ tự LABELS."""
    validate_prediction_output(prediction)
    return max(LABELS, key=prediction.__getitem__)
