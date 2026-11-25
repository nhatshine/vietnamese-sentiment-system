# Sentence Segmentation – Bước 4

## Mục đích

Sentence Segmentation là tách một đoạn văn thành danh sách câu. Khi phân
tích cảm xúc theo câu ở bước sau, thao tác này giúp giữ riêng từng ý,
ví dụ camera được khen nhưng pin bị chê trong cùng một đoạn.

Module hiện chỉ tách văn bản. Nó không gọi SVM, Transformer hoặc Ensemble,
không dự đoán sentiment và không tổng hợp cảm xúc toàn đoạn.

## Input và output

Hàm nằm trong `backend/src/sentence_splitter.py`:

```python
def split_sentences(text: str) -> list[str]:
```

- Input phải là `str`; kiểu khác gây `TypeError`.
- Chuỗi rỗng hoặc chỉ chứa whitespace trả `[]`.
- Output giữ thứ tự câu, nội dung Unicode tiếng Việt và dấu câu.
- Bỏ whitespace đầu/cuối câu; giữ nguyên whitespace bên trong mỗi câu.
- Không tự thêm dấu kết thúc vào câu chưa có dấu câu và không tạo câu rỗng.
- Kết quả deterministic, không dùng model hay thư viện NLP ngoài.

Ví dụ:

```python
split_sentences("Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!")
# ["Camera rất đẹp.", "Nhưng pin quá yếu.", "Giao hàng rất nhanh!"]
```

## Quy tắc preprocessing bắt buộc

**Không gọi `clean_text()` trước sentence splitting.** Preprocessing model
có thể loại dấu câu, làm mất ranh giới câu. Splitter nhận nguyên đoạn raw,
để các câu raw còn dấu câu phục vụ hiển thị trên giao diện sau này.

Luồng đúng cho phân tích câu trong tương lai:

```text
Raw paragraph
↓
Sentence Splitter
↓
Raw sentence 1
Raw sentence 2
...
↓
Prediction adapter/function thực hiện preprocessing riêng từng câu: clean_text()
↓
Sentiment Model
```

Sau khi tách câu, từng câu mới được đưa qua preprocessing chung của hai
model đúng một lần. Nếu wrapper model đã gọi `clean_text()`, lớp tích hợp
không gọi lại. Splitter không thực hiện preprocessing hoặc prediction.
Analyzer Bước 5 chuyển nguyên raw sentence cho prediction function, không
trực tiếp gọi `clean_text()`. Xem
[tài liệu Sentence-level Analysis](SENTENCE_LEVEL_ANALYSIS.md).

## Boundary được hỗ trợ

- Các dấu `.`, `!`, `?` và cụm liên tiếp như `...`, `?!`, `!?`, `!!!`, `???`.
- Cụm dấu câu liên tiếp được giữ cùng câu trước, không chia thành nhiều câu rỗng.
- Newline, gồm `\n`, `\r\n` và các ký tự xuống dòng do `str.splitlines()` nhận diện.
- Dòng trống được bỏ qua; một dòng không có dấu câu vẫn là một câu.
- Có thể tách khi sau dấu kết thúc không có khoảng trắng, ví dụ `Đẹp!Pin yếu?`.

Hai ngoại lệ dấu chấm cơ bản:

1. Dấu chấm nằm ngay giữa hai chữ số không phải boundary, như `1.5`, `2.0`, `3.14`.
   Dấu chấm kết thúc sau một số vẫn tách câu: `Tôi chấm điểm 5. Bạn thì sao?`.
2. Bảo vệ dấu chấm của danh sách viết tắt nhỏ, không phân biệt hoa/thường:
   `TS.`, `ThS.`, `PGS.`, `GS.`, `TP.`, `Mr.`, `Mrs.`, `Dr.`.

Ví dụ:

```python
split_sentences("TS. Nguyễn Văn A đánh giá sản phẩm tốt. Tôi đồng ý.")
# ["TS. Nguyễn Văn A đánh giá sản phẩm tốt.", "Tôi đồng ý."]

split_sentences("Giá khoảng 1.5 triệu. Tôi thấy khá hợp lý.")
# ["Giá khoảng 1.5 triệu.", "Tôi thấy khá hợp lý."]
```

Code dùng regex để xác định các dấu chấm được bảo vệ và các cụm kết thúc
câu, rồi lấy nguyên lát cắt từ văn bản gốc; không xóa dấu câu hoặc thay thế
bằng các token tạm có thể xung đột với nội dung.

## Hạn chế của phương pháp dùng luật

- Không có mô hình hiểu ngữ cảnh để nhận biết mọi ranh giới câu tiếng Việt.
- Danh sách viết tắt hữu hạn; từ viết tắt ngoài danh sách có thể bị tách sai.
- Dấu chấm của viết tắt được bảo vệ trong cùng dòng; nếu nó đồng thời kết
  thúc một câu, hai câu cùng dòng có thể bị gộp. Newline vẫn là boundary.
- Chưa xử lý riêng URL, email, tên miền, dấu ngoặc kép/ngoặc đóng hoặc các
  trường hợp dấu chấm mang nghĩa đặc biệt khác.
- Mỗi newline được coi là boundary, kể cả khi văn bản chỉ xuống dòng vì dàn trang.
- Cụm `...` được coi là kết thúc câu theo yêu cầu hiện tại dù đôi khi người
  viết dùng để ngập ngừng giữa câu.

## Chạy test và demo

Từ thư mục gốc project:

```powershell
python -m unittest discover backend/tests
python -m backend.demo_sentence_splitter
```

Demo in đoạn gốc, số câu và danh sách câu được đánh số cho bốn đoạn mẫu.
Chỉ dùng Python standard library; không cài underthesea, spaCy, nltk hoặc transformers.

Phạm vi module Bước 4 chỉ là Sentence Segmentation. Bước 5 đã có analyzer
theo câu với mock models; Bước 6 đã có
[Document Aggregation](DOCUMENT_LEVEL_AGGREGATION.md) riêng. Bước 7 có
[Real SVM](REAL_SVM_INTEGRATION.md) tùy chọn; splitter vẫn độc lập với model.
Bước 8 có [Real Transformer](REAL_TRANSFORMER_INTEGRATION.md); preprocessing
nằm trong tokenizer của predictor, không nằm trong splitter. Chưa alpha tuning,
BEST_METHOD, FastAPI hoặc React.
