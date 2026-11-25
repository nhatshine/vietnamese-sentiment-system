"""Chạy từ thư mục gốc: python -m backend.demo_real_svm."""

from .src.config import LABELS
from .src.contracts import get_predicted_label
from .src.real_svm import predict_svm_with_details


def main() -> None:
    """In kết quả thật; cleaned text và probability đến từ cùng lần inference."""
    texts = [
        "Sản phẩm này rất tốt.",
        "Tôi rất thất vọng vì sản phẩm quá tệ.",
        "Sản phẩm này bình thường.",
        "Camera rất đẹp nhưng pin quá yếu.",
        "Giao hàng chậm và đóng gói cẩu thả.",
    ]
    for text in texts:
        details = predict_svm_with_details(text)
        probabilities = details["probabilities"]
        label = get_predicted_label(probabilities)
        print("\n==================================================")
        print(f"INPUT TEXT:\n{text}")
        print(f"\nCLEANED TEXT:\n{details['cleaned_text']}")
        print("\nMODEL:\nTF-IDF + SVM (REAL)")
        print(f"\nPREDICTION:\n{label}")
        print(f"\nCONFIDENCE:\n{probabilities[label]:.9f}")
        print("\nPROBABILITIES:")
        for name in LABELS:
            print(f"{name} = {probabilities[name]:.9f}")


if __name__ == "__main__":
    main()
