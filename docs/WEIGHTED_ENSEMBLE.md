# Weighted Ensemble Engine và kế hoạch nghiên cứu Phần 3

## Phạm vi Bước 3

Hai mô hình cơ sở:

1. TF-IDF + SVM.
2. Transformer tự xây và huấn luyện từ đầu.

Phương án thứ ba là **Weighted Ensemble**, tầng kết hợp prediction probability.
Nó không được train như SVM hoặc Transformer và không phải model train thứ ba.
Engine ban đầu được kiểm tra với mock models và các bộ xác suất tính tay.
Bước 7 đã có SVM thật tùy chọn. Khi kết hợp với Mock Transformer, đó chỉ
là hybrid integration test. Bước 8 đã có cả hai predictor thật để chạy Real
Ensemble, xem [tài liệu Real Transformer](REAL_TRANSFORMER_INTEGRATION.md).
Bước 9 đã sweep alpha trên Validation, chọn 0.8 với Macro-F1 0.8255456672,
xem [báo cáo alpha tuning](ALPHA_TUNING_EXPERIMENT.md). Chưa đánh giá cuối
trên Test hoặc triển khai selector BEST_METHOD; default 0.5 vẫn giữ nguyên.

## Công thức và interface

```text
P_final(c) = alpha * P_svm(c) + (1 - alpha) * P_transformer(c)
```

`alpha` là trọng số SVM; `1 - alpha` là trọng số Transformer. Ví dụ
`alpha=0.3` nghĩa là SVM 30%, Transformer 70%.

```python
weighted_ensemble(svm_probs: dict, transformer_probs: dict, alpha: float = 0.5) -> dict
```

Hàm nằm trong `backend/src/ensemble.py` và chỉ nhận probability lowercase.
SVM thật phải qua `normalize_svm_output()` trước khi đưa vào engine;
logic adapter không nằm trong engine. Hai mock model hiện đã trả lowercase.

Luồng xử lý:

1. Validate cả hai input bằng contract hiện có, kể cả input có trọng số 0.
2. Kiểm tra alpha là `int` hoặc `float`, không nhận `bool`, chuỗi hoặc `None`.
   Alpha phải hữu hạn và thuộc `[0, 1]`; `NaN` và vô cực bị từ chối.
3. Tính từng xác suất theo tên nhãn, duyệt đúng `LABELS` từ `config.py`.
4. Validate xác suất cuối, không round sớm hoặc chuẩn hóa lại tổng.
5. Dùng `get_predicted_label()` để lấy nhãn; confidence là xác suất của nhãn đó.
   Nếu hòa xác suất cao nhất, ưu tiên thứ tự `LABELS` theo utility hiện có.

Hàm không mutate input. Sai kiểu gây `TypeError`; alpha ngoài miền hoặc
probability sai gây `ValueError` theo validator hiện có. Contract probability
gốc vẫn chỉ gồm ba nhãn; metadata của ensemble nằm bên ngoài `probabilities`.

Ví dụ cấu trúc trả về (giá trị trình bày theo dạng thập phân):

```python
{
    "method": "ensemble",
    "label": "positive",
    "confidence": 0.790,
    "probabilities": {
        "negative": 0.065,
        "neutral": 0.145,
        "positive": 0.790,
    },
    "weights": {"svm": 0.3, "transformer": 0.7},
}
```

Do biểu diễn float, giá trị thực tế có thể có thêm chữ số ở cuối. Engine
không round; demo chỉ định dạng khi in. Contract dùng sai số tổng tuyệt đối
`1e-5`, giữ nguyên cấu hình của Bước 2.

Hai trường hợp biên:

```text
alpha = 0 → P_final = P_transformer
alpha = 1 → P_final = P_svm
```

## Ví dụ tính tay bắt buộc: alpha = 0.3

| Nhãn | SVM | Transformer | Phép tính | Kết quả |
|---|---:|---:|---|---:|
| negative | 0.10 | 0.05 | 0.3 × 0.10 + 0.7 × 0.05 | 0.065 |
| neutral | 0.25 | 0.10 | 0.3 × 0.25 + 0.7 × 0.10 | 0.145 |
| positive | 0.65 | 0.85 | 0.3 × 0.65 + 0.7 × 0.85 | 0.790 |

Nhãn cuối là `positive`, confidence xấp xỉ `0.790`. Unit test kiểm tra
từng giá trị với sai số floating-point nhỏ.

## Demo hiện tại

Chạy từ thư mục gốc project:

```powershell
python -m unittest discover backend/tests
python -m backend.demo_ensemble
```

Demo gọi hai mock model và engine cho ba câu:

- `Sản phẩm này rất tốt`.
- `Tôi rất thất vọng vì sản phẩm quá tệ`.
- `Sản phẩm này bình thường`.

`alpha=0.5` hiện chỉ là giá trị demo, **không phải trọng số tối ưu**.
Kết quả mock chỉ kiểm tra luồng và công thức, không phải bằng chứng hiệu quả NLP.

## Câu hỏi nghiên cứu

### RQ1

Việc kết hợp TF-IDF + SVM và Transformer bằng Weighted Ensemble có cải thiện
hiệu quả phân tích cảm xúc tiếng Việt so với từng mô hình riêng lẻ hay không?

### RQ2

Trọng số kết hợp giữa SVM và Transformer ảnh hưởng như thế nào đến Macro-F1?

### RQ3

Hiệu quả của phương án kết hợp có ổn định trên các nguồn dữ liệu `hieu_sa`,
`neu_esc`, `vsfc` hay không?

## Mục tiêu nghiên cứu

Mục tiêu không phải chứng minh trước rằng Ensemble tốt hơn. Nghiên cứu sẽ
so sánh **TF-IDF + SVM vs Transformer vs Weighted Ensemble** để xác định
phương án phân tích sentiment tốt nhất trong phạm vi hệ thống.

Tiêu chí chính là **Macro-F1**, bổ sung Accuracy, Precision và Recall.
Không giả định trước Ensemble sẽ thắng. Chưa tính các metric trên dữ liệu thật.

## Kế hoạch tuning và lựa chọn phương án sau này

Bước 9 đã thực hiện phần tuning alpha trên Validation. Phần chọn phương án
model/aggregation và đánh giá cuối trên Test dưới đây vẫn là kế hoạch:

```text
SVM / Transformer / Ensemble
        ↓
đánh giá trên Validation
        ↓
tuning alpha
        ↓
chọn cấu hình/phương án tốt nhất theo Validation Macro-F1
        ↓
khóa cấu hình, alpha* và BEST_METHOD
        ↓
đánh giá cuối trên Test
```

Khi có model thật, thử trên cùng Validation Set:

```text
alpha = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

Tính Validation Macro-F1 cho từng alpha và chọn `alpha*` có điểm cao nhất.
`alpha=0` tương đương Transformer hoàn toàn, `alpha=1` tương đương SVM hoàn toàn.
So sánh hai phương án cơ sở với Ensemble đã chọn alpha trên Validation.
Quy tắc xử lý hòa điểm cần được xác định trước khi khóa cấu hình.

Validation dùng để tuning trọng số và lựa chọn cấu hình/phương án.
Test chỉ dùng để đánh giá cuối sau khi mọi lựa chọn đã khóa. Không thử đi
thử lại nhiều cấu hình trên Test rồi chọn phương án tốt nhất.

## Kế hoạch đánh giá theo source

Giữ nguyên giá trị `source`; đây chỉ là metadata đánh giá, không phải
input feature của model. Bảng dưới là kế hoạch, chưa có kết quả thực nghiệm:

| Source | SVM Macro-F1 | Transformer Macro-F1 | Ensemble Macro-F1 |
|---|---|---|---|
| hieu_sa | Chưa đo | Chưa đo | Chưa đo |
| neu_esc | Chưa đo | Chưa đo | Chưa đo |
| vsfc | Chưa đo | Chưa đo | Chưa đo |

Có thể tính thêm `max(source_f1) - min(source_f1)` hoặc độ lệch chuẩn F1
giữa ba nguồn để mô tả độ ổn định. Báo cáo theo source trên Test cuối cùng
không được dùng để quay lại chọn hoặc tuning cấu hình trên cùng Test đó.

## Yêu cầu giao diện tương lai

Giao diện sẽ hiển thị kết quả của cả ba phương án: TF-IDF + SVM,
Transformer và Weighted Ensemble. Ngoài ra có khu vực **KẾT QUẢ TỐI ƯU**,
lấy output từ phương án đã được chọn qua thực nghiệm:

```text
BEST_METHOD = ensemble    → output của Weighted Ensemble
BEST_METHOD = transformer → output của Transformer
BEST_METHOD = svm         → output của TF-IDF + SVM
```

Không chọn KẾT QUẢ TỐI ƯU bằng cách lấy confidence cao nhất giữa ba phương
án trên từng câu. Confidence không thay thế kết quả đánh giá phương án.
Hiện chưa chọn `BEST_METHOD`, chưa code selector và chưa code React.
