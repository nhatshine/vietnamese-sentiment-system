# Document-level Sentiment Aggregation – Bước 6

## Mục đích và kiến trúc

Tổng hợp kết quả nhiều câu thành sentiment toàn đoạn, thực hiện riêng cho
SVM, Transformer và Weighted Ensemble. Mỗi phương án model có đủ ba chiến
lược: Majority Voting, Mean Probability và Confidence-Weighted Probability.

```text
Raw Text
   ↓
Sentence Segmentation
   ↓
Sentence-level Analysis
   ↓
 ┌───────────────────────────────┐
 │ SVM                           │
 │ Transformer                   │
 │ Weighted Ensemble             │
 └───────────────────────────────┘
   ↓
Document Aggregation
   ↓
 ┌───────────────────────────────┐
 │ Majority Voting               │
 │ Mean Probability              │
 │ Confidence Weighted           │
 └───────────────────────────────┘
```

`sentence_analyzer.py` giữ trách nhiệm phân tích từng câu. Logic tổng hợp
nằm riêng trong `backend/src/document_aggregator.py`. Các hàm aggregation
chỉ đọc sentence results đã có, không gọi trực tiếp model hoặc preprocessing.
Chỉ wrapper `analyze_document()` gọi pipeline `analyze_sentences()` của Bước 5.

## Input và API

Input aggregation là `list[dict]` từ `analyze_sentences()`. Mỗi phần tử có
sentence metadata và các prediction chứa `label`, `confidence`, `probabilities`.
`model_key` chỉ nhận `svm`, `transformer`, `ensemble` theo `MODEL_KEYS`.
Các nhãn probability giữ nguyên lowercase và thứ tự `LABELS` trong `config.py`.
Bước 7 cho phép `analyze_document(..., svm_predictor=real_svm)` truyền Real
SVM xuống pipeline theo câu; không đổi ba công thức aggregation. Mặc định
vẫn dùng mock. Chi tiết ở [tài liệu Real SVM](REAL_SVM_INTEGRATION.md).
Bước 8 thêm `transformer_predictor`; truyền cả hai predictor thật để chạy
Real Ensemble rồi tổng hợp đủ ba chiến lược cho cả ba phương án.

```python
aggregate_majority_vote(sentence_results: list[dict], model_key: str) -> dict
aggregate_mean_probability(sentence_results: list[dict], model_key: str) -> dict
aggregate_confidence_weighted(sentence_results: list[dict], model_key: str) -> dict
aggregate_model_document(sentence_results: list[dict], model_key: str) -> dict
aggregate_all_models(sentence_results: list[dict]) -> dict
analyze_document(text: str, alpha: float = 0.5) -> dict
```

Các hàm tạo output mới, không sửa sentence metadata, probability hoặc
weights gốc. Không round sớm; `math.fsum()` được dùng để cộng xác suất.

## Majority Voting

Mỗi câu bỏ một phiếu cho nhãn dự đoán. Nhãn có nhiều phiếu nhất thắng.
Ví dụ positive, negative, negative cho vote counts:

```python
{"negative": 2, "neutral": 0, "positive": 1}
```

Output gồm `aggregation_method="majority_vote"`, `model`, `label`,
`confidence`, `vote_counts` và `vote_distribution`.

```text
vote_distribution(c) = số phiếu cho c / tổng số câu
confidence = số phiếu cho nhãn thắng / tổng số câu
```

Confidence ở đây là **tỷ lệ phiếu**, không phải neural-network probability
hoặc xác suất dự đoán của một model. Ví dụ trên có confidence `2/3`.

### Tie-break deterministic

1. Chỉ xét các nhãn đang hòa ở số phiếu cao nhất.
2. Với mỗi nhãn đó, lấy mean probability của nhãn trên **tất cả câu** của
   model tương ứng. Chọn nhãn có mean lớn nhất.
3. Nếu mean vẫn hòa chính xác, chọn nhãn đứng trước trong `LABELS`.

Không dùng random, không round mean trước khi so sánh và không cho nhãn
ngoài nhóm hòa phiếu thắng chỉ vì có mean probability cao.

Ưu điểm: đơn giản, dễ giải thích. Nhược điểm: bỏ qua mức confidence khi
bỏ phiếu; câu 51% và câu 99% vẫn có cùng một phiếu.

## Mean Probability

Trung bình xác suất của tất cả câu theo từng lớp:

```text
P_document(c) = (1/N) × Σ P_sentence_i(c)
```

Output gồm `aggregation_method="mean_probability"`, `model`, `label`,
`confidence` và `probabilities`. Nhãn là argmax của xác suất tổng hợp,
confidence là xác suất của nhãn đó. Hòa xác suất dùng utility nhãn hiện có,
ưu tiên thứ tự `LABELS`.

Ưu điểm: sử dụng nhiều thông tin hơn hard label. Nhược điểm: mọi câu đều
có trọng số bằng nhau, dù mức chắc chắn hoặc nội dung khác nhau.

## Confidence-Weighted Probability

Trọng số mỗi câu là xác suất lớn nhất của câu đó:

```text
weight_i = max(P_i(negative), P_i(neutral), P_i(positive))

P_document(c) = Σ weight_i × P_i(c) / Σ weight_i
```

Weight được tính trực tiếp từ probability, tương ứng confidence của
prediction hợp lệ. Không dùng độ dài câu làm weight. Các probability được
chia cho tổng weights khi kết hợp; field `sentence_weights` giữ trọng số
thô theo đúng thứ tự câu, không chia lại cho tổng.

Output gồm `aggregation_method="confidence_weighted"`, `model`, `label`,
`confidence`, `probabilities` và `sentence_weights`. Nhãn/confidence lấy từ
phân phối tổng hợp như Mean Probability.

Ưu điểm: tận dụng confidence, để câu model tự tin hơn có ảnh hưởng lớn hơn.
Nhược điểm: phụ thuộc chất lượng **calibration của probability**;
confidence cao không đồng nghĩa chắc chắn dự đoán đúng. Chưa có thực nghiệm
chứng minh chiến lược này tốt hơn hai chiến lược còn lại.

## Validation

- Aggregation trực tiếp yêu cầu list không rỗng: sai kiểu gây `TypeError`,
  list rỗng gây `ValueError`.
- Model key không hợp lệ, model bị thiếu hoặc prediction thiếu field đều
  có lỗi rõ ràng. Item/prediction phải là dictionary.
- Probability được kiểm tra bằng `validate_prediction_output()` hiện có,
  không sao chép lại logic contract. Tổng được phép lệch `1e-5` so với 1.
- Label phải thuộc `LABELS` và là một argmax của probability. Nếu nhiều
  lớp cùng đạt max, nhãn được cung cấp vẫn hợp lệ khi thuộc nhóm max.
- Confidence phải là int/float hữu hạn trong `[0, 1]`, không nhận bool;
  phải khớp probability của nhãn trong sai số tuyệt đối `1e-5`, dùng hằng
  tolerance đã có. Không dùng metadata confidence sai để làm weight.
- Probability tổng hợp và vote distribution được kiểm tra lại; không
  tự chuẩn hóa các input có tổng sai.

## Output tổng hợp

`aggregate_model_document()` tái sử dụng ba hàm aggregation:

```text
model: tên model
majority_vote: kết quả bỏ phiếu
mean_probability: kết quả trung bình xác suất
confidence_weighted: kết quả có trọng số confidence
```

`aggregate_all_models()` trả dictionary với ba key `svm`, `transformer`,
`ensemble`. Mỗi value là output của `aggregate_model_document()`, giữ
metadata `model` cùng đủ ba kết quả aggregation. Không chọn một chiến lược.

`analyze_document()` giữ nguyên text và toàn bộ kết quả từng câu:

```text
text: đoạn raw ban đầu
sentence_count: số câu
sentences: list kết quả Bước 5
document_results: kết quả riêng ba model, mỗi model ba chiến lược
```

Với document rỗng hoặc whitespace:

```python
{
    "text": text,
    "sentence_count": 0,
    "sentences": [],
    "document_results": {},
}
```

Wrapper không gọi aggregation trên list rỗng. Non-string bị splitter chặn
như Bước 5. Alpha được truyền nguyên xuống sentence analyzer/Ensemble;
alpha chỉ được validate khi có câu cần phân tích.

## Ví dụ tính tay độc lập với mock

| Câu | negative | neutral | positive | Nhãn | Weight |
|---|---:|---:|---:|---|---:|
| 1 | 0.05 | 0.10 | 0.85 | positive | 0.85 |
| 2 | 0.80 | 0.15 | 0.05 | negative | 0.80 |
| 3 | 0.70 | 0.20 | 0.10 | negative | 0.70 |

Majority Voting: negative 2 phiếu, neutral 0, positive 1; nhãn negative,
confidence `2/3`.

Mean Probability:

```text
negative = 1.55 / 3 = 0.5166666667...
neutral  = 0.45 / 3 = 0.1500000000...
positive = 1.00 / 3 = 0.3333333333...
```

Confidence Weighted có tổng weights `0.85 + 0.80 + 0.70 = 2.35`:

```text
negative = (0.0425 + 0.64 + 0.49) / 2.35 = 0.4989361702...
neutral  = (0.0850 + 0.12 + 0.14) / 2.35 = 0.1468085106...
positive = (0.7225 + 0.04 + 0.07) / 2.35 = 0.3542553191...
```

Cả hai phân phối đều có tổng xấp xỉ 1 và nhãn negative. Unit test dùng
fixture thủ công này để đối chiếu công thức, không phụ thuộc luật mock.

## Câu hỏi nghiên cứu phụ

Việc tổng hợp sentiment ở mức câu bằng Majority Voting, Mean Probability
và Confidence-Weighted Aggregation ảnh hưởng như thế nào đến kết quả
sentiment ở mức văn bản?

Ba chiến lược chỉ là các phương án nghiên cứu. Chưa kết luận model nào hoặc
aggregation method nào tốt nhất; output hiện tại không có field lựa chọn
phương án tối ưu. Khi có dữ liệu/model thật, cần đánh giá aggregation riêng,
chọn cấu hình trên Validation và khóa trước khi đánh giá cuối trên Test.
Không dùng Test để thử nhiều cấu hình rồi chọn cấu hình tốt nhất.

## Chạy test và demo

```powershell
python -m unittest discover backend/tests
python -m backend.demo_document_aggregation
```

Demo dùng `Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!`,
hiển thị sentence results và đủ chín kết quả document-level.
Mock vẫn trả neutral cho câu có `pin quá yếu` vì chưa có từ khóa `yếu`.
Alpha 0.5 chỉ để demo, không phải alpha đã tối ưu.

Hai model thật đã tích hợp tùy chọn đến Bước 8. Chỉ truyền một predictor
thật vẫn là hybrid integration test; truyền cả hai mới có Real Ensemble.
Chưa tuning alpha,
chạy Validation/Test để lựa chọn model, đánh giá aggregation thật, chọn
BEST_METHOD, FastAPI hoặc React. Bản thân module aggregation và các mock
demo vẫn chỉ dùng Python standard library; dependencies SVM nằm ở predictor.
