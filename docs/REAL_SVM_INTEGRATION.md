# Tích hợp Real SVM – Bước 7

## Nguồn artifact và preprocessing

Sử dụng model Phần 1 do người dùng cung cấp, không tải từ mạng:

- Nguồn model: `C:/Users/Admin/data+model_SVM/model_SVM/models/svm_pipeline.joblib`.
- Nguồn preprocessing: `C:/Users/Admin/data+model_SVM/data_final/preprocess.py`.
- Bản sử dụng trong project: `backend/models/svm/svm_pipeline.joblib` và
  `backend/src/preprocess.py`.

Hai file được copy nguyên byte, đối chiếu SHA-256 với nguồn:

```text
svm_pipeline.joblib:
fc5a0bf009f4c58ecb41c32ee5098487155e4d5af9e210a35936d3738f39d2e1

preprocess.py:
20d8e57b3aa014f5c4cab733560971b5dea4aca813c0c7637dee7a0620fa5eae
```

File `svm_meta.json` trong bộ bàn giao xác nhận sklearn 1.9.1 và class IDs
0, 1, 2. Pipeline thực tế đã load thành công, không có cảnh báo phiên bản:

```text
Pipeline
├── features: FeatureUnion
│   ├── word: TfidfVectorizer
│   └── char: TfidfVectorizer
└── clf: CalibratedClassifierCV
```

Không load riêng TF-IDF, không gọi fit và không retrain. Các vectorizer
trong artifact có `preprocessor=None`, `tokenizer=None`; không có lần gọi
`clean_text()` ẩn trong pipeline. Model weights tiếp tục được `.gitignore`
loại khỏi Git. Chỉ load artifact local tin cậy do người dùng bàn giao.

## Môi trường đã kiểm chứng

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.13, Anaconda, Windows |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| NumPy | 2.3.5 |
| SciPy, dependency gián tiếp | 1.18.1 |

Các package này đã có trong môi trường khi tích hợp; không cài hoặc đổi
package toàn máy. `backend/requirements.txt` ghi ba dependency trực tiếp
đã kiểm chứng. Scikit-learn và NumPy được pin yêu cầu Python >=3.11;
SciPy 1.18.1 trong môi trường kiểm chứng yêu cầu Python >=3.12.
Để tái lập đúng môi trường đã chạy, dùng Python 3.13.13 và các phiên bản trên.

Nếu dùng môi trường mới:

```powershell
python -m pip install -r backend/requirements.txt
```

Các mock demo vẫn không cần dependencies Real SVM. Module Real SVM không
import joblib hoặc load artifact cho đến khi gọi inference/load model.

## Load model và cache

`backend/src/real_svm.py` xác định đường dẫn bằng `Path(__file__)` tới
`backend/models/svm/svm_pipeline.joblib`, không hard-code đường dẫn ổ C và
không phụ thuộc working directory để tìm artifact.

`load_model()` dùng `lru_cache(maxsize=1)`: các lần dự đoán liên tiếp trong
cùng process dùng lại model đã deserialize. Sau load, module kiểm tra
`predict_proba()` và `classes_`. Không load lại checkpoint cho từng câu.

## Inference và preprocessing đúng một lần

```text
RAW TEXT
↓
clean_text() gốc
↓
pipeline.predict_proba([cleaned_text])
↓
mapping từng cột theo classes_
↓
prediction probability lowercase
```

`predict_svm(text: str) -> dict[str, float]` nhận raw text. Preprocessing
gốc xử lý Unicode NFC, lowercase, URL/email, emoji/emoticon, số, slang và
dấu câu, giữ dấu tiếng Việt và từ phủ định. Mã preprocessing được giữ nguyên.

Không truyền text đã clean vào predictor này. Không gọi wrapper
`predict_svm.py` bên ngoài vốn cũng tự clean, rồi lại clean thêm trong module.
Sentence/document analyzer chuyển raw sentence vào predictor; không clean
toàn đoạn trước segmentation.

`predict_svm_with_details()` dùng cùng luồng inference và trả thêm `text`,
`cleaned_text`, `probabilities`. Demo dùng helper này để in cleaned text từ
chính lần inference đó, không gọi `clean_text()` lần thứ hai để hiển thị.

## Class mapping

`classes_` thực tế là NumPy array `[0, 1, 2]`, mỗi phần tử kiểu `int64`.
Mapping được đọc theo từng giá trị class, không dựa vào vị trí cột mặc định:

| Class ID | Nhãn nội bộ |
|---:|---|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

Ví dụ nếu classes_ là `[2, 0, 1]`, các cột lần lượt được đưa vào positive,
negative, neutral. Unit test kiểm tra trường hợp đảo thứ tự này.

Ngoài class IDs nguyên, module hỗ trợ bộ nhãn lowercase hoặc bộ nhãn
`Negative/Neutral/Positive`. Với bộ viết hoa đầu từ, module tái sử dụng
`normalize_svm_output()` ở Bước 2. Thiếu/trùng class, class ID lạ, encoding
trộn hoặc số cột không khớp đều gây lỗi rõ ràng, không âm thầm đảo nhãn.

## Prediction contract và lỗi thường gặp

- Output có đúng ba key lowercase theo `LABELS`.
- Value được chuyển sang Python `float`, serialize JSON được.
- Tái sử dụng `validate_prediction_output()`: finite, `[0, 1]`, tổng gần 1
  với `abs_tol=1e-5`, `rel_tol=0.0` đã có.
- Không round probability; chỉ định dạng số lúc in demo.
- Input khác `str`: `TypeError`.
- Input rỗng/whitespace hoặc rỗng sau preprocessing, như `!!!`: `ValueError`,
  không tạo probability giả cho input không còn nội dung.
- Thiếu artifact: `FileNotFoundError` kèm đường dẫn chính xác.
- Thiếu dependency khi load: `ImportError` kèm lệnh cài requirements.
- Artifact không có `predict_proba` hoặc `classes_`: lỗi interface rõ ràng.
- Nếu có cảnh báo phiên bản sklearn khi thay artifact/môi trường, đối chiếu
  metadata và cài đúng môi trường bàn giao trước khi dùng kết quả inference.

## Dùng standalone hoặc injection

```python
from backend.src.real_svm import predict_svm as real_svm

probabilities = real_svm("Sản phẩm này rất tốt.")
```

Các hàm cũ giữ nguyên cách gọi và mặc định mock SVM. Chọn Real SVM bằng
keyword-only argument `svm_predictor`:

```python
from backend.src.real_svm import predict_svm as real_svm
from backend.src.sentence_analyzer import analyze_sentence, analyze_sentences
from backend.src.document_aggregator import analyze_document

sentence = analyze_sentence("Camera rất đẹp.", svm_predictor=real_svm)
sentences = analyze_sentences("Camera rất đẹp. Pin yếu.", svm_predictor=real_svm)
document = analyze_document("Camera rất đẹp. Pin yếu.", alpha=0.5, svm_predictor=real_svm)
```

Trong cách gọi này: kết quả `svm` dùng model thật, `transformer` vẫn là mock,
và `ensemble` kết hợp hai nguồn đó. Đây là **hybrid integration test** để
kiểm tra wiring, không phải kết quả Ensemble thật hoặc kết quả nghiên cứu.
Không thay công thức Ensemble/aggregation, không chọn BEST_METHOD.
Bước 8 đã có Real Transformer: truyền thêm `transformer_predictor` để chạy
cả hai model thật. Xem [tài liệu Real Transformer](REAL_TRANSFORMER_INTEGRATION.md).

## Chạy kiểm thử và demo

```powershell
python -m unittest discover backend/tests
python -m backend.demo_real_svm
```

Test gồm inference trên artifact thật, mapping đảo cột và các encoding,
validation, load cache, một lần preprocessing và hybrid sentence/document.
Các stub chỉ dùng trong test mapping/error; integration test và demo dùng
đúng artifact thật. Không buộc câu minh họa phải được phân loại đúng.

Demo ngày 29/10/2026, số chỉ được định dạng khi hiển thị:

| Input | negative | neutral | positive | Nhãn |
|---|---:|---:|---:|---|
| Sản phẩm này rất tốt. | 0.003743700 | 0.003862124 | 0.992394176 | positive |
| Tôi rất thất vọng vì sản phẩm quá tệ. | 0.981958446 | 0.000969004 | 0.017072550 | negative |
| Sản phẩm này bình thường. | 0.215179979 | 0.603974161 | 0.180845860 | neutral |
| Camera rất đẹp nhưng pin quá yếu. | 0.303592718 | 0.026683528 | 0.669723754 | positive |
| Giao hàng chậm và đóng gói cẩu thả. | 0.895132288 | 0.008683363 | 0.096184349 | negative |

Các xác suất trên đến từ artifact, không phải output hard-code trong demo.
Không tính metric mới hoặc sử dụng Validation/Test để lựa chọn model.
Bước 8 đã tích hợp Transformer thật nhưng chưa tuning alpha, đánh giá
Macro-F1 mới, chọn BEST_METHOD/aggregation tối ưu, FastAPI hoặc React.
Artifact và preprocessing SVM không bị thay đổi.
