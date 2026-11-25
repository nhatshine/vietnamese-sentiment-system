"""Chạy từ thư mục gốc: python -m backend.demo_sentence_analyzer."""

from .src.config import LABELS
from .src.sentence_analyzer import analyze_sentences


def main() -> None:
    """In kết quả riêng SVM, Transformer và Ensemble cho từng câu mẫu."""
    text = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"
    alpha = 0.5
    results = analyze_sentences(text, alpha=alpha)

    print("========================================")
    print(f"ORIGINAL TEXT\n\n{text}")
    print(f"\nSentence count: {len(results)}")
    print("\nĐây là kết quả mock để kiểm tra luồng, chưa phải đánh giá NLP thật.")

    for result in results:
        print("\n========================================")
        print(f"CÂU {result['sentence_index'] + 1}\n{result['sentence']}")
        for method in ("svm", "transformer", "ensemble"):
            prediction = result[method]
            print(f"\n{method.upper()}")
            print(f"Label: {prediction['label']}")
            print(f"Confidence: {prediction['confidence']:.6f}")
            print("Probabilities:")
            for label in LABELS:
                print(f"  {label}: {prediction['probabilities'][label]:.6f}")
            if method == "ensemble":
                print("Weights:")
                print(f"  SVM: {prediction['weights']['svm']}")
                print(f"  Transformer: {prediction['weights']['transformer']}")

    print("\n========================================")
    print("alpha = 0.5 chỉ là giá trị demo.")
    print("Chưa phải alpha tối ưu.")
    print("Chưa có kết quả sentiment toàn đoạn.")


if __name__ == "__main__":
    main()
