"""Chạy từ thư mục gốc: python -m backend.demo_sentence_splitter."""

from .src.sentence_splitter import split_sentences


def main() -> None:
    """In danh sách câu của bốn đoạn mẫu; không gọi model sentiment."""
    paragraphs = [
        "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!",
        "Sản phẩm tuyệt quá!!! Nhưng giá hơi cao... Có nên mua không?",
        "TS. Nguyễn Văn A đánh giá sản phẩm tốt. Tôi đồng ý.",
        "Giá khoảng 1.5 triệu. Tôi thấy khá hợp lý.",
    ]

    for index, paragraph in enumerate(paragraphs, start=1):
        sentences = split_sentences(paragraph)
        print(f"\n================================\nĐOẠN {index}")
        print(f"\nORIGINAL TEXT\n{paragraph}")
        print(f"\nSENTENCE COUNT\n{len(sentences)}\n")
        for sentence_index, sentence in enumerate(sentences, start=1):
            print(f"[{sentence_index}] {sentence}")


if __name__ == "__main__":
    main()
