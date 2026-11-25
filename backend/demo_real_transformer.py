"""Demo hai model thật và Ensemble: python -m backend.demo_real_transformer."""

from .src.config import LABELS
from .src.document_aggregator import MODEL_KEYS, analyze_document
from .src.real_svm import predict_svm
from .src.real_transformer import predict_transformer
from .src.sentence_analyzer import analyze_sentence


def print_predictions(result: dict) -> None:
    """Chỉ định dạng output khi in; không round dữ liệu trước khi Ensemble."""
    titles = {"svm": "REAL SVM", "transformer": "REAL TRANSFORMER", "ensemble": "REAL WEIGHTED ENSEMBLE"}
    for method in MODEL_KEYS:
        prediction = result[method]
        print(f"\n{titles[method]}")
        if method == "ensemble":
            print(f"Alpha: {prediction['weights']['svm']}")
        print(f"Label: {prediction['label']}")
        print(f"Confidence: {prediction['confidence']:.9f}")
        print("Probabilities:")
        for label in LABELS:
            print(f"  {label}: {prediction['probabilities'][label]:.9f}")


def main() -> None:
    texts = [
        "Sản phẩm này rất tốt.",
        "Tôi rất thất vọng vì sản phẩm quá tệ.",
        "Sản phẩm này bình thường.",
        "Camera rất đẹp nhưng pin quá yếu.",
        "Giao hàng chậm và đóng gói cẩu thả.",
    ]
    alpha = 0.5
    print("Inference CPU; hai model thật; alpha=0.5 chỉ để kiểm thử tích hợp.")
    for text in texts:
        print("\n==================================================")
        print(f"INPUT:\n{text}")
        result = analyze_sentence(text, alpha=alpha, svm_predictor=predict_svm,
                                  transformer_predictor=predict_transformer)
        print_predictions(result)

    text = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"
    document = analyze_document(text, alpha=alpha, svm_predictor=predict_svm,
                                transformer_predictor=predict_transformer)
    print("\n==================================================\nDOCUMENT INPUT:")
    print(text)
    print(f"Sentence count: {document['sentence_count']}")
    for sentence in document["sentences"]:
        print(f"\nCÂU {sentence['sentence_index'] + 1}: {sentence['sentence']}")
        print_predictions(sentence)

    for model in MODEL_KEYS:
        print(f"\nDOCUMENT AGGREGATION — {model.upper()}")
        for method in ("majority_vote", "mean_probability", "confidence_weighted"):
            result = document["document_results"][model][method]
            print(f"\n{method}: label={result['label']}, confidence={result['confidence']:.9f}")
            if method == "majority_vote":
                print(f"Vote counts: {result['vote_counts']}")
                print("Confidence là tỷ lệ phiếu.")
            else:
                for label in LABELS:
                    print(f"  {label}: {result['probabilities'][label]:.9f}")
                if method == "confidence_weighted":
                    print(f"Sentence weights: {result['sentence_weights']}")

    print("\nĐây chưa phải kết quả thực nghiệm tối ưu.")
    print("Chưa tuning alpha, chọn BEST_METHOD hoặc chọn aggregation tối ưu.")
    print("Chưa chạy Validation/Test để lựa chọn hoặc đánh giá lại model.")


if __name__ == "__main__":
    main()
