# Sentence-level Sentiment Analysis – Bước 5

## Mục đích

Phân tích sentiment riêng cho từng câu trong một đoạn văn. Module
`backend/src/sentence_analyzer.py` kết nối các thành phần đã có, sử dụng
mock SVM, mock Transformer và Weighted Ensemble Engine. Không viết lại
luật của mock model, logic splitter hoặc công thức Ensemble.
Bước 7 bổ sung keyword-only `svm_predictor` để dùng SVM thật; không truyền
tham số này thì hành vi cũ với mock được giữ nguyên. Xem
[tài liệu Real SVM](REAL_SVM_INTEGRATION.md).
Bước 8 bổ sung `transformer_predictor` theo cùng cơ chế, cho phép sử dụng
cả hai model thật. Xem [tài liệu Real Transformer](REAL_TRANSFORMER_INTEGRATION.md).

## Pipeline

```text
Raw Paragraph
↓
Sentence Segmentation
↓
Raw Sentence
↓
SVM                 Transformer
  └── probabilities ───┘
            ↓
     Weighted Ensemble
            ↓
Sentence-level Results: SVM + Transformer + Ensemble
```

SVM và Transformer là hai phương án cơ sở; Ensemble là tầng kết hợp hai
bộ xác suất cho cùng câu, không phải model train thứ ba hoặc kết quả cuối
bắt buộc của hệ thống.

## Interface

```python
analyze_sentence(sentence: str, alpha: float = 0.5) -> dict
analyze_sentences(text: str, alpha: float = 0.5) -> list[dict]
```

Hai hàm còn nhận `*, svm_predictor=None, transformer_predictor=None`. Truyền
`backend.src.real_svm.predict_svm` để chọn Real SVM; predictor nhận nguyên raw
sentence và chịu trách nhiệm preprocessing. Nếu không truyền predictor
Transformer thì phương án đó vẫn mock và Ensemble chỉ là hybrid integration
test. Truyền cả hai predictor thật để Ensemble nhận đúng hai output thật.

### Phân tích một câu

`analyze_sentence()` nhận một **raw sentence**. Không tự tách câu ở hàm này.
Input khác `str` gây `TypeError`; câu rỗng hoặc chỉ chứa whitespace gây
`ValueError`. Output giữ nguyên chuỗi truyền vào, gồm dấu câu và whitespace.

Hàm gọi hai mock model rồi truyền probability cho `weighted_ensemble()`.
Các utility contract hiện có xác nhận probability hợp lệ. Alpha được engine
Ensemble kiểm tra: chỉ nhận `int` hoặc `float` hữu hạn trong `[0, 1]`, không
nhận `bool`, chuỗi hoặc `None`.

Ví dụ output cho `analyze_sentence("Camera rất đẹp.")` (số thập phân được
trình bày cho dễ đọc; code không round probability):

```python
{
    "sentence": "Camera rất đẹp.",
    "svm": {
        "method": "svm",
        "label": "positive",
        "confidence": 0.75,
        "probabilities": {"negative": 0.08, "neutral": 0.17, "positive": 0.75},
    },
    "transformer": {
        "method": "transformer",
        "label": "positive",
        "confidence": 0.85,
        "probabilities": {"negative": 0.05, "neutral": 0.10, "positive": 0.85},
    },
    "ensemble": {
        "method": "ensemble",
        "label": "positive",
        "confidence": 0.80,
        "probabilities": {"negative": 0.065, "neutral": 0.135, "positive": 0.80},
        "weights": {"svm": 0.5, "transformer": 0.5},
    },
}
```

Mỗi nhãn được chọn bằng utility `get_predicted_label()` đã có. Confidence
là xác suất của nhãn đó. Output Ensemble được sử dụng trực tiếp từ engine,
không tính lại. Probability SVM/Transformer được copy sang output mới,
không mutate dictionary do prediction function trả về.

### Phân tích các câu trong đoạn văn

`analyze_sentences()` gọi `split_sentences(text)` trên đoạn raw, sau đó gọi
`analyze_sentence(sentence, alpha)` cho từng câu theo đúng thứ tự.

Output là `list[dict]`. Mỗi phần tử có đúng các trường:

```text
sentence_index: 0, 1, 2, ...
sentence: câu raw do splitter trả về
svm: kết quả SVM của câu đó
transformer: kết quả Transformer của câu đó
ensemble: kết quả Ensemble của câu đó
```

Backend dùng index từ 0; demo có thể hiển thị CÂU 1, CÂU 2, ... bằng cách cộng 1.
Splitter chỉ trim boundary whitespace, giữ punctuation và whitespace bên
trong câu. Đoạn rỗng hoặc whitespace trả `[]`, không gọi prediction.
Kiểu input khác `str` được splitter chặn bằng `TypeError`. Alpha chỉ được
Ensemble validate khi có câu để phân tích.

## Preprocessing và trách nhiệm adapter

**Sentence Analyzer không trực tiếp gọi `clean_text()`.** Không clean đoạn
văn trước khi split vì có thể làm mất punctuation dùng để xác định boundary.

Khi tích hợp model thật, luồng phải là:

```text
Raw Paragraph
↓
Sentence Splitter
↓
Raw Sentence
↓
Model Prediction Adapter / Function
↓
Preprocessing phù hợp model, dùng quy tắc chung đã bàn giao
↓
Model prediction
↓
Probability lowercase
```

Prediction adapter/function của từng model chịu trách nhiệm preprocessing
đúng một lần, không yêu cầu analyzer clean lại. Hai model thật phải tuân
theo preprocessing đầu vào chung đã chốt với Phần 1. Dữ liệu CSV đã được
preprocessing dùng train Transformer không gọi `clean_text()` lại.

`normalize_svm_output()` trong `adapters.py` chỉ là adapter **output xác suất**:
đổi key viết hoa đầu từ về lowercase. Không thêm preprocessing văn bản vào
hàm đó. Mock SVM hiện đã mô phỏng output sau adapter nên dùng trực tiếp.

## Tại sao giữ cả ba output?

Nghiên cứu cần so sánh SVM, Transformer và Ensemble trên cùng đầu vào.
Giữ riêng ba kết quả giúp kiểm tra luồng và chuẩn bị cho đánh giá thực
nghiệm sau này, không mặc định Ensemble luôn là phương án tốt nhất.

## Tại sao chưa chọn BEST_METHOD?

Hai model thật đã tích hợp tùy chọn ở Bước 8, nhưng chưa có kết quả thực
nghiệm so sánh model trên Validation/Test. Alpha 0.5
hiện chỉ là giá trị demo, chưa được tuning. Sau này dùng Validation để tuning
alpha và lựa chọn phương án theo Macro-F1, khóa cấu hình trước khi đánh giá
cuối trên Test. Không dùng confidence lớn nhất giữa ba phương án trên từng
câu làm tiêu chí chọn BEST_METHOD.

## Ranh giới với Document Aggregation

Document Aggregation là bước riêng. Bước 5 chỉ trả kết quả từng câu, không
thêm `document_label`, không majority voting, không average probability
giữa các câu và không tổng hợp confidence. Ensemble hiện chỉ kết hợp hai
model cho **cùng một câu**, không kết hợp các câu trong đoạn.

Bước 6 đã bổ sung `document_aggregator.py` để đọc output Bước 5 và tổng hợp
riêng bằng ba chiến lược. Không thêm document result vào sentence analyzer;
wrapper `analyze_document()` mới trả cả sentence results và document results.
Xem [tài liệu Document Aggregation](DOCUMENT_LEVEL_AGGREGATION.md).

## Demo và giới hạn mock

Chạy từ thư mục gốc project:

```powershell
python -m unittest discover backend/tests
python -m backend.demo_sentence_analyzer
```

Demo dùng `Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!` với
alpha 0.5, in riêng ba kết quả cho mỗi câu. Theo luật mock hiện tại, câu
`Nhưng pin quá yếu.` trả neutral vì từ `yếu` không nằm trong danh sách từ
khóa. Không thay đổi mock chỉ để làm kết quả demo giống model NLP thật.
Các nhãn demo chỉ kiểm tra luồng, không phải bằng chứng về chất lượng NLP.

Chưa tuning alpha, chạy Validation/Test để lựa
chọn model, chọn BEST_METHOD, FastAPI hoặc React. Phạm vi module Bước 5 dừng ở Sentence-level
Analysis; aggregation đã được triển khai riêng ở Bước 6, chưa đánh giá hoặc
lựa chọn chiến lược tối ưu.
