"""Chạy từ thư mục gốc: python -m backend.demo_mock."""

from .src.mock_models import predict_svm, predict_transformer


def main() -> None:
    """In kết quả hai mock model cho ba câu mẫu."""
    texts = [
        "Sản phẩm này rất tốt",
        "Tôi rất thất vọng vì sản phẩm quá tệ",
        "Sản phẩm này bình thường",
    ]

    for text in texts:
        print("TEXT:")
        print(text)
        print("\nSVM:")
        print(predict_svm(text))
        print("\nTRANSFORMER:")
        print(predict_transformer(text))
        print()


if __name__ == "__main__":
    main()
