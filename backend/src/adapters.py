"""Chuẩn hóa output bên ngoài sang interface xác suất nội bộ Phần 3."""

from .config import LABELS
from .contracts import validate_prediction_output


def normalize_svm_output(prediction: dict) -> dict:
    """Đổi key SVM dạng viết hoa đầu từ thành lowercase và value thành float.

    Không sửa dictionary đầu vào, không làm tròn hoặc chuẩn hóa lại xác suất.
    Chỉ chấp nhận đúng bộ key của SVM Phần 1, được suy ra từ LABELS.
    """
    if not isinstance(prediction, dict):
        raise TypeError("Output SVM phải là dictionary chứa xác suất của các nhãn.")

    external_to_internal = {label.title(): label for label in LABELS}
    missing = [label for label in external_to_internal if label not in prediction]
    unknown = [label for label in prediction if label not in external_to_internal]
    if missing or unknown:
        raise ValueError(
            f"Output SVM phải có chính xác các nhãn {list(external_to_internal)}. "
            f"Thiếu nhãn: {missing}; nhãn lạ: {unknown}."
        )

    result = {}
    for external_label, internal_label in external_to_internal.items():
        value = prediction[external_label]
        if isinstance(value, bool):
            raise TypeError(f"Xác suất của '{external_label}' không được là bool.")
        try:
            result[internal_label] = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                f"Xác suất của '{external_label}' phải chuyển được thành float."
            ) from error

    validate_prediction_output(result)
    return result
