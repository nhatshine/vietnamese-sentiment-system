"""Kiểm tra tách câu thô, giữ dấu câu và các ngoại lệ cơ bản."""

import unittest

from backend.src.sentence_splitter import split_sentences


class SentenceSplitterTests(unittest.TestCase):
    def test_single_sentence_without_terminal_punctuation(self):
        self.assertEqual(split_sentences("Sản phẩm này khá tốt"), ["Sản phẩm này khá tốt"])

    def test_sentences_separated_by_periods(self):
        self.assertEqual(split_sentences("Camera đẹp. Pin yếu."), ["Camera đẹp.", "Pin yếu."])

    def test_exclamation_boundary(self):
        self.assertEqual(split_sentences("Rất đẹp! Tôi thích."), ["Rất đẹp!", "Tôi thích."])

    def test_question_boundary(self):
        self.assertEqual(split_sentences("Ổn không? Tôi chưa biết."), ["Ổn không?", "Tôi chưa biết."])

    def test_ellipsis_stays_with_sentence(self):
        self.assertEqual(split_sentences("Hơi đắt... Nhưng đẹp."), ["Hơi đắt...", "Nhưng đẹp."])

    def test_question_exclamation_cluster(self):
        self.assertEqual(split_sentences("Ổn chứ?! Tôi lo."), ["Ổn chứ?!", "Tôi lo."])

    def test_exclamation_question_cluster(self):
        self.assertEqual(split_sentences("Thật sao!? Khó tin."), ["Thật sao!?", "Khó tin."])

    def test_repeated_exclamation_marks(self):
        self.assertEqual(split_sentences("Tuyệt quá!!! Tôi thích."), ["Tuyệt quá!!!", "Tôi thích."])

    def test_repeated_question_marks(self):
        self.assertEqual(split_sentences("Thật sao??? Tôi chưa rõ."), ["Thật sao???", "Tôi chưa rõ."])

    def test_newline_boundaries_without_punctuation(self):
        expected = ["Camera rất đẹp", "Pin hơi yếu", "Giao hàng nhanh"]
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=newline):
                self.assertEqual(split_sentences(newline.join(expected)), expected)

    def test_boundary_whitespace_is_trimmed(self):
        self.assertEqual(split_sentences("  Camera đẹp.     Nhưng pin yếu.  "), [
            "Camera đẹp.", "Nhưng pin yếu.",
        ])

    def test_empty_string(self):
        self.assertEqual(split_sentences(""), [])

    def test_whitespace_only(self):
        for text in ("   ", "\t\n\r\n", "\u00a0\u2003"):
            with self.subTest(text=text):
                self.assertEqual(split_sentences(text), [])

    def test_non_string_input(self):
        for text in (None, 123, 0.5, True, [], {}, b"text"):
            with self.subTest(text=text):
                with self.assertRaises(TypeError):
                    split_sentences(text)

    def test_vietnamese_unicode_is_preserved(self):
        self.assertEqual(split_sentences("Tôi rất hài lòng 😊. Cảm ơn!"), [
            "Tôi rất hài lòng 😊.", "Cảm ơn!",
        ])

    def test_punctuation_is_preserved(self):
        self.assertEqual(split_sentences("Camera rất đẹp! Pin yếu?"), [
            "Camera rất đẹp!", "Pin yếu?",
        ])

    def test_decimal_points_are_not_boundaries(self):
        for number in ("1.5", "2.0", "3.14"):
            with self.subTest(number=number):
                self.assertEqual(split_sentences(f"Giá khoảng {number} triệu. Tôi thấy khá hợp lý."), [
                    f"Giá khoảng {number} triệu.", "Tôi thấy khá hợp lý.",
                ])

    def test_supported_abbreviations_inside_sentences(self):
        for abbreviation in ("TS.", "ThS.", "PGS.", "GS.", "TP.", "Mr.", "Mrs.", "Dr."):
            with self.subTest(abbreviation=abbreviation):
                first = f"{abbreviation} Nguyễn Văn A đánh giá sản phẩm tốt."
                self.assertEqual(split_sentences(f"{first} Tôi đồng ý."), [first, "Tôi đồng ý."])

    def test_mixed_punctuation_and_newlines(self):
        self.assertEqual(split_sentences("Rất đẹp!\r\nNhưng hơi đắt.\nCó nên mua không?"), [
            "Rất đẹp!", "Nhưng hơi đắt.", "Có nên mua không?",
        ])

    def test_no_empty_sentences_from_blank_lines_or_punctuation_clusters(self):
        self.assertEqual(split_sentences("\n\n  Camera đẹp...   \n\n Pin yếu?! \n "), [
            "Camera đẹp...", "Pin yếu?!",
        ])

    def test_whitespace_inside_sentence_is_not_changed(self):
        self.assertEqual(split_sentences("  Camera\t  rất đẹp.  "), ["Camera\t  rất đẹp."])

    def test_boundaries_without_spaces(self):
        self.assertEqual(split_sentences("Đẹp!Pin yếu?Giá cao."), ["Đẹp!", "Pin yếu?", "Giá cao."])

    def test_chained_abbreviations_are_case_insensitive(self):
        self.assertEqual(split_sentences("pgs.ts. Nguyễn Văn A đồng ý. Cảm ơn."), [
            "pgs.ts. Nguyễn Văn A đồng ý.", "Cảm ơn.",
        ])

    def test_sentence_ending_in_a_number_still_splits(self):
        self.assertEqual(split_sentences("Tôi chấm điểm 5. Bạn thì sao?"), [
            "Tôi chấm điểm 5.", "Bạn thì sao?",
        ])


if __name__ == "__main__":
    unittest.main()
