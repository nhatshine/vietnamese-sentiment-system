"""Chạy từ thư mục gốc: python -m backend.demo_document_aggregation."""

from .src.config import LABELS
from .src.document_aggregator import MODEL_KEYS, analyze_document


def main() -> None:
    """In kết quả theo câu và cả ba aggregation cho mỗi phương án model."""
    text = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"
    result = analyze_document(text, alpha=0.5)

    print("==================================================")
    print(f"ORIGINAL TEXT\n{text}")
    print(f"\nSENTENCE COUNT: {result['sentence_count']}")
    print("\n==================================================\nSENTENCE RESULTS")
    for sentence_result in result["sentences"]:
        print(f"\n[{sentence_result['sentence_index'] + 1}] {sentence_result['sentence']}")
        for model in MODEL_KEYS:
            prediction = sentence_result[model]
            print(f"  {model.upper()}: {prediction['label']}, confidence={prediction['confidence']:.6f}")

    methods = (
        ("majority_vote", "Majority Voting"),
        ("mean_probability", "Mean Probability"),
        ("confidence_weighted", "Confidence Weighted"),
    )
    for model in MODEL_KEYS:
        print(f"\n==================================================\nDOCUMENT RESULTS — {model.upper()}")
        for method, title in methods:
            prediction = result["document_results"][model][method]
            print(f"\n{title}")
            print(f"Label: {prediction['label']}")
            print(f"Confidence: {prediction['confidence']:.6f}")
            if method == "majority_vote":
                print("Confidence ở đây là tỷ lệ phiếu, không phải probability của model.")
                print(f"Vote counts: {prediction['vote_counts']}")
                print(f"Vote distribution: {prediction['vote_distribution']}")
            else:
                print("Probabilities:")
                for label in LABELS:
                    print(f"  {label}: {prediction['probabilities'][label]:.6f}")
                if method == "confidence_weighted":
                    print(f"Sentence weights: {prediction['sentence_weights']}")

    print("\n==================================================")
    print("Các kết quả hiện tại sử dụng MOCK MODELS.")
    print("alpha = 0.5 chỉ để demo.")
    print("Chưa xác định aggregation method tối ưu.")
    print("Chưa xác định SVM / Transformer / Ensemble phương án nào tốt nhất.")


if __name__ == "__main__":
    main()
