"""
predict_svm.py - Dự đoán cảm xúc bằng model TF-IDF + SVM đã huấn luyện.

Dùng trong code (ví dụ Ensemble của Người 3):
    from predict_svm import predict_svm, predict_svm_label

    predict_svm("giảng viên dạy rất nhiệt tình")
    # {'Negative': 0.03, 'Neutral': 0.09, 'Positive': 0.88}   (tổng = 1)

    predict_svm(["câu 1", "câu 2"])      # list đầu vào -> list dict đầu ra (nhanh hơn gọi từng câu)
    predict_svm_label("câu 1")           # -> 'Positive'

Dùng từ dòng lệnh:
    python predict_svm.py "ko hiểu bài chán quá"

Hàm này TỰ gọi clean_text (preprocess.py) nên truyền chữ gốc vào là được.
Đặt predict_svm.py và preprocess.py cùng thư mục; model ở models/svm_pipeline.joblib
(hoặc chỉ đường dẫn bằng biến môi trường SVM_MODEL_PATH).
"""
import os
import sys
from pathlib import Path

import joblib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from preprocess import clean_text  # noqa: E402

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
_model = None


def _find_model_path() -> Path:
    env = os.environ.get("SVM_MODEL_PATH")
    candidates = [Path(env)] if env else []
    candidates += [_HERE / "models" / "svm_pipeline.joblib",
                   _HERE.parent / "models" / "svm_pipeline.joblib",
                   Path.cwd() / "models" / "svm_pipeline.joblib"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Không tìm thấy svm_pipeline.joblib. Đã thử:\n  "
                            + "\n  ".join(map(str, candidates)))


def load_model():
    """Nạp model một lần rồi dùng lại."""
    global _model
    if _model is None:
        _model = joblib.load(_find_model_path())
    return _model


def predict_svm(text):
    """
    text: str hoặc list[str] (chữ gốc, chưa làm sạch).
    Trả về dict {'Negative': p0, 'Neutral': p1, 'Positive': p2} (tổng = 1),
    hoặc list các dict nếu đầu vào là list.
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)
    model = load_model()
    cleaned = [clean_text(t) for t in texts]
    proba = model.predict_proba(cleaned)
    classes = [int(c) for c in model.classes_]
    out = [{LABEL_NAMES[c]: round(float(p), 6) for c, p in zip(classes, row)} for row in proba]
    return out[0] if single else out


def predict_svm_label(text):
    """Trả về tên nhãn có xác suất cao nhất ('Negative'/'Neutral'/'Positive')."""
    res = predict_svm(text)
    if isinstance(res, dict):
        return max(res, key=res.get)
    return [max(r, key=r.get) for r in res]


if __name__ == "__main__":
    samples = sys.argv[1:] or ["giảng viên dạy rất nhiệt tình", "ko hiểu bài chán quá :("]
    for s in samples:
        print(s, "->", predict_svm(s))
