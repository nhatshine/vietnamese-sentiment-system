"""Demo mock: chạy python -m backend.demo_ensemble từ thư mục gốc project."""

from .src.config import LABELS
from .src.contracts import get_predicted_label
from .src.ensemble import weighted_ensemble
from .src.mock_models import predict_svm, predict_transformer


def main() -> None:
    """Hiển thị ba phương án trên ba câu mẫu, không thực hiện đánh giá thật."""
    alpha = 0.5
    texts = [
        "Sản phẩm này rất tốt",
        "Tôi rất thất vọng vì sản phẩm quá tệ",
        "Sản phẩm này bình thường",
    ]
    print("Demo sử dụng hai mock model.")
    print("alpha = 0.5 hiện chỉ là giá trị demo. Không phải trọng số tối ưu.")

    for text in texts:
        svm_probs = predict_svm(text)
        transformer_probs = predict_transformer(text)
        result = weighted_ensemble(svm_probs, transformer_probs, alpha=alpha)

        print("\n================================")
        print(f"\nTEXT:\n{text}")
        for method, probabilities in (("SVM", svm_probs), ("TRANSFORMER", transformer_probs)):
            print(f"\n{method}")
            print(f"Label: {get_predicted_label(probabilities)}")
            print("Probabilities:")
            for label in LABELS:
                print(f"  {label}: {probabilities[label]:.6f}")

        print("\nWEIGHTED ENSEMBLE")
        print("Weights:")
        print(f"SVM         = {result['weights']['svm']}")
        print(f"Transformer = {result['weights']['transformer']}")
        print(f"\nLabel:\n{result['label']}")
        print(f"\nConfidence:\n{result['confidence']:.6f}")
        print("\nProbabilities:")
        for label in LABELS:
            print(f"  {label}: {result['probabilities'][label]:.6f}")
        print("\n================================")


if __name__ == "__main__":
    main()
