"""Tách câu thô bằng luật; không preprocessing hoặc phân tích sentiment."""

import re


_ABBREVIATIONS = ("TS", "ThS", "PGS", "GS", "TP", "Mr", "Mrs", "Dr")
_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)
_DECIMAL_POINT_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")
_SENTENCE_END_PATTERN = re.compile(r"[.!?]+")


def split_sentences(text: str) -> list[str]:
    """Tách đoạn văn raw thành câu, giữ dấu câu và whitespace bên trong.

    Newline hoặc cụm .!? là boundary, trừ dấu chấm trong số thập phân và
    danh sách viết tắt cơ bản. Chỉ trim đầu/cuối mỗi câu, bỏ mảnh rỗng.
    Không gọi clean_text(), model hoặc ensemble. Đây là heuristic đơn giản,
    không giải quyết mọi trường hợp viết tắt, URL hoặc dấu ngoặc kép.
    """
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi (str).")
    if not text.strip():
        return []

    sentences = []
    for line in text.splitlines():
        protected_dots = {
            match.start() for match in _DECIMAL_POINT_PATTERN.finditer(line)
        }
        protected_dots.update(
            match.end() - 1 for match in _ABBREVIATION_PATTERN.finditer(line)
        )

        start = 0
        for boundary in _SENTENCE_END_PATTERN.finditer(line):
            if boundary.group() == "." and boundary.start() in protected_dots:
                continue
            sentence = line[start:boundary.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = boundary.end()

        remainder = line[start:].strip()
        if remainder:
            sentences.append(remainder)

    return sentences
