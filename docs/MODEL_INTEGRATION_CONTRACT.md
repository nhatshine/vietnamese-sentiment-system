# Chuẩn bàn giao model

## Mục đích

Đây là chuẩn bàn giao giữa Phần 1 (TF-IDF + SVM), Phần 2 (Transformer tự
xây và huấn luyện từ đầu) và Phần 3 (tích hợp, phân tích và demo).
Hiện đã hoàn thành Bước 1, Bước 2 và Bước 3 (Weighted Ensemble Engine với mock).
Bước 4 bổ sung Sentence Segmentation độc lập, chưa kết nối model.
Bước 5 kết nối splitter với hai mock model và Ensemble để trả kết quả từng câu.
Bước 6 tổng hợp sentence results riêng cho mỗi phương án bằng ba chiến lược.
Bước 7 đã load SVM thật và copy preprocessing nguyên gốc từ các file do
người dùng cung cấp. Artifact xác nhận classes_ [0, 1, 2]; metadata và môi
trường load đều dùng scikit-learn 1.9.1, không có cảnh báo phiên bản.
Bước 8 đã tích hợp Transformer gốc từ train_Transformer.zip và Real Ensemble;
mặc định vẫn mock nếu không truyền hai predictor.
Bước 9 thêm cleaned-batch APIs để đánh giá val.csv đã preprocessing mà không
gọi clean_text lại. Raw predictors và alpha mặc định 0.5 không thay đổi.
Đã chọn alpha Validation 0.8 trên lưới 11 điểm; xem
[thực nghiệm alpha tuning](ALPHA_TUNING_EXPERIMENT.md).
Bước 9.1/10 đã phân tích lỗi và đánh giá Test với cấu hình khóa theo Validation.
Bước 11 thêm `PredictionService` mặc định hai model thật và một final result;
comparison đầy đủ chỉ trình bày khi bật research. Analyzer/demo cũ vẫn giữ default mock.

## Input

Mỗi hàm nhận một chuỗi tiếng Việt:

```python
text: str
text = "Sản phẩm này rất tốt"
```

## Label mapping

```text
0 = negative
1 = neutral
2 = positive
```

Thứ tự cố định là `negative`, `neutral`, `positive`. Mã nguồn phải import
`LABEL_TO_ID`, `ID_TO_LABEL`, `LABELS` từ `backend/src/config.py`, không tự
khai báo lại. Dictionary được đọc theo tên key; khi chuyển sang vector,
luôn lấy xác suất theo thứ tự `LABELS`, không dựa vào thứ tự chèn key.

## Chuẩn nội bộ Phần 3

Prediction probability trong Phần 3 giữ nguyên chuẩn lowercase:

```python
{
    "negative": float,
    "neutral": float,
    "positive": float,
}
```

Đây là mô tả kiểu giá trị. Ví dụ output nội bộ:

```python
{
    "negative": 0.12,
    "neutral": 0.18,
    "positive": 0.70,
}
```

## Output SVM thật và adapter

SVM của Phần 1 qua hàm `predict_svm(text)` hiện trả key viết hoa đầu từ:

```python
{
    "Negative": 0.03,
    "Neutral": 0.09,
    "Positive": 0.88,
}
```

Đó là output của wrapper Phần 1. Module `real_svm.py` của Phần 3 đọc trực
tiếp `predict_proba()` và `classes_` trong artifact: class ID 0, 1, 2 được
đổi qua `ID_TO_LABEL` rồi validate. Nếu một artifact dùng nhãn viết hoa đầu
từ, module tái sử dụng `normalize_svm_output()`. Không giả định thứ tự cột.
Interface `backend.src.real_svm.predict_svm` luôn trả lowercase ngay từ đầu.

Luồng tích hợp khi cắm model thật:

```text
SVM thật của Phần 1
        ↓
predict_svm()
        ↓
Negative / Neutral / Positive
        ↓
normalize_svm_output()
        ↓
negative / neutral / positive
        ↓
Phần 3
```

`normalize_svm_output(prediction)` nằm trong `backend/src/adapters.py`:

- Yêu cầu dictionary có đúng `Negative`, `Neutral`, `Positive`.
- Báo `ValueError` với danh sách nhãn thiếu hoặc lạ; không gộp key trùng
  nghĩa như `Positive` và `positive`, không bỏ qua key thừa.
- Suy ra tên key từ `LABELS`, xuất key lowercase theo đúng thứ tự nội bộ.
- Chuyển từng giá trị bằng `float(...)`; báo `TypeError` khi không chuyển
  được hoặc khi giá trị là `bool`.
- Gọi `validate_prediction_output()` trước khi trả kết quả để kiểm tra
  miền giá trị, số hữu hạn và tổng xác suất.
- Không sửa dictionary gốc, không làm tròn, không chia lại xác suất cho tổng.

### Lý do dùng adapter

Không sửa artifact/model/code Phần 1 chỉ để đổi chữ hoa/thường. Phần tích
hợp chịu trách nhiệm chuẩn hóa output tại ranh giới trước khi đưa vào Phần 3.

Mock `predict_svm()` trong `backend/src/mock_models.py` mô phỏng interface
SAU adapter, nên tiếp tục trả lowercase trực tiếp. Không gọi adapter yêu
cầu key viết hoa này lên kết quả mock.

## Output của Transformer

Phần 2 phải bàn giao hàm `predict_transformer(text)` với format hoàn toàn
tương tự chuẩn nội bộ: dictionary chứa đúng ba key lowercase và xác suất
kiểu `float`. Mock Transformer hiện cũng tuân theo chuẩn đó.
`backend.src.real_transformer.predict_transformer` đã có, dùng label_map.json
thực tế 0=negative, 1=neutral, 2=positive. Checkpoint/config/vocab giữ nguyên;
softmax không round trước khi validate hoặc Ensemble.

## Kiểm tra output

- Không được thiếu key, thừa key hoặc dùng tên nhãn khác.
- Mỗi xác suất phải hữu hạn, thuộc đoạn `[0, 1]`.
- Tổng xác suất lệch tối đa `1e-5` so với 1 (sai số tuyệt đối).
  Validator dùng `math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5)`,
  chấp nhận tổng làm tròn như `0.999999` hoặc `1.000001`.
- Model bàn giao giá trị `float`; validator chấp nhận thêm `int` như 0 và 1.
- Không nhận `bool`, chuỗi số, `NaN` hoặc vô cực.
- Không được chỉ trả về nhãn cuối như `"positive"`.

`validate_prediction_output(prediction)` trả `None` khi hợp lệ; sai kiểu
dữ liệu gây `TypeError`, sai nhãn hoặc xác suất gây `ValueError`.
Hàm không sửa dữ liệu và không tự chuẩn hóa tổng xác suất.
`get_predicted_label(prediction)` kiểm tra trước rồi trả tên nhãn có xác suất
cao nhất; nếu hòa, ưu tiên nhãn đứng trước trong `LABELS`.

Ví dụ sử dụng từ thư mục gốc project:

```python
from backend.src.contracts import get_predicted_label, validate_prediction_output

prediction = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
validate_prediction_output(prediction)
label = get_predicted_label(prediction)  # "positive"
```

## Quy tắc preprocessing chung

Phần 1 đã có `clean_text()` dùng chung cho SVM, Transformer và inference.
Hai model phải dùng cùng preprocessing đầu vào khi inference.

```text
TRAIN TRANSFORMER:
train.csv / val.csv / test.csv đã preprocessing
→ KHÔNG gọi clean_text() lại.

INFERENCE:
raw text
→ clean_text()
→ model
```

Đối với luồng phân tích câu từ một đoạn văn, cần tách trên raw text trước:

```text
raw paragraph
→ split_sentences()
→ raw sentences
→ prediction adapter/function thực hiện clean_text() riêng từng câu
→ model
```

Không gọi `clean_text()` trước sentence splitting vì có thể mất dấu câu
và boundary. Splitter Bước 4 giữ dấu câu để hiển thị, chưa gọi preprocessing
hoặc model. Chi tiết nằm trong
[tài liệu Sentence Segmentation](SENTENCE_SEGMENTATION.md).

Analyzer Bước 5 cũng không trực tiếp gọi `clean_text()`: nó chuyển nguyên
raw sentence cho prediction function của từng model. Khi tích hợp model thật,
prediction adapter/function phải chịu trách nhiệm preprocessing phù hợp model
đúng một lần theo quy tắc chung. Chi tiết output ba phương án nằm trong
[tài liệu Sentence-level Analysis](SENTENCE_LEVEL_ANALYSIS.md).

Khi tích hợp thật, mỗi đầu vào inference chỉ được clean một lần bằng
preprocessing chung đã bàn giao. Nếu `predict_svm.py` đã gọi `clean_text()`
thì lớp tích hợp không gọi thêm lần nữa. Adapter chỉ xử lý output xác suất,
không thực hiện preprocessing văn bản.

`backend/src/preprocess.py` đã được copy nguyên byte từ Phần 1. Real SVM
import `clean_text()` từ file này và gọi đúng một lần cho raw input. Analyzer
không tự preprocess; mock vẫn giữ luật từ khóa riêng để bảo toàn test/demo cũ.
Preprocessing trong ZIP Transformer trùng nguyên byte/SHA-256 với bản này.
Tokenizer gốc đã gọi clean_text(); wrapper Transformer truyền raw text vào
encode_text(), không clean thêm trước đó. Không preprocessing hai lần.

## Metadata nguồn dữ liệu

Giữ nguyên các giá trị `source` hiện có trong CSV của Phần 1:

```text
hieu_sa
neu_esc
vsfc
```

Không tự đổi tên các nguồn. `source` không phải input feature của model,
không thêm vào tham số `predict_svm(text)` hoặc `predict_transformer(text)`.
Chỉ giữ metadata này để sau này đánh giá kết quả theo từng nguồn dữ liệu.

## SVM pipeline và môi trường bàn giao

Theo thông tin bàn giao, artifact SVM là `svm_pipeline.joblib` và đã chứa:

- Word TF-IDF;
- Character TF-IDF;
- SVM đã calibration probability.

Vì vậy Phần 3 sau này không cần load riêng `word_tfidf.pkl` hoặc
`char_tfidf.pkl`. Khi tích hợp thật, bộ bàn giao cần gồm:

- `svm_pipeline.joblib`;
- `predict_svm.py`;
- preprocessing chung.

Theo thông tin nhóm cung cấp, SVM được train bằng scikit-learn 1.9.1.
Artifact đã load thành công với môi trường tương ứng:

```text
scikit-learn==1.9.1
```

`backend/requirements.txt` đã pin scikit-learn 1.9.1, joblib 1.6.0 và NumPy
2.3.5 theo môi trường Real SVM kiểm chứng. Không cài thêm/thay package toàn
máy; bản thân SVM không cần pandas/PyTorch. Model được lazy load và cache
một lần mỗi process, không fit lại TF-IDF hoặc retrain.
Bước 8 dùng PyTorch 2.14.0+cpu đã có sẵn, ghi release 2.14.0 vào requirements.
Transformer mặc định CPU, weights_only=True, map_location='cpu', strict load,
model.eval() và torch.inference_mode(); model được cache theo device.

## Tại sao cần probability?

Phần 3 sử dụng xác suất của từng lớp để thực hiện Weighted Ensemble:

```text
P_final = alpha * P_SVM + (1 - alpha) * P_Transformer
```

Các vector xác suất phải cùng thứ tự `LABELS`; `alpha` là trọng số trong
`[0, 1]`. Chỉ trả nhãn cuối sẽ làm mất thông tin cần để phối hợp hai model.
Công thức đã được triển khai trong `backend/src/ensemble.py` và kiểm tra
bằng mock models. Engine chỉ nhận output lowercase đã chuẩn hóa; không
chứa logic adapter hoặc tải model thật.

Output của engine có các trường `method`, `label`, `confidence`,
`probabilities` và `weights`. Chỉ dictionary bên trong `probabilities`
tuân theo contract ba nhãn trên; metadata không được thêm vào probability
dictionary gốc. Alpha mặc định 0.5 chỉ phục vụ demo, chưa được tuning.
Weighted Ensemble là tầng kết hợp xác suất của hai mô hình cơ sở,
không phải model được train thứ ba. Chi tiết nằm trong
[tài liệu Weighted Ensemble](WEIGHTED_ENSEMBLE.md).

## Quy định quan trọng

- Hai model phải dùng cùng label mapping.
- Dùng cùng train/validation/test split khi thực nghiệm.
- Không đảo thứ tự `negative` / `neutral` / `positive`.
- Không chỉ trả nhãn cuối.
- Bàn giao preprocessing của từng model cùng model, để suy luận dùng đúng
  cách xử lý văn bản khi huấn luyện.
- Khi bàn giao, cung cấp hàm dự đoán, các artifact cần thiết, cách tải model
  và thông tin môi trường/phụ thuộc tương ứng.

Hai mock model, adapter và Weighted Ensemble Engine được giữ nguyên. Real
SVM đã có và có thể inject vào analyzer bằng `svm_predictor`; mặc định vẫn
mock. Nếu chỉ truyền một predictor thật thì vẫn là hybrid integration test.
Truyền cả svm_predictor và transformer_predictor thật sẽ cho Real Ensemble.
Bước 9 đã tuning alpha trên Validation; Bước 10 đã đánh giá Test với cấu hình
được khóa trước inference, không chọn lại từ Test. Sentence splitter và sentence-level analysis đã có;
Document Aggregation đã có riêng ba chiến lược cho mỗi phương án, chưa đánh
giá hoặc lựa chọn chiến lược tối ưu. Majority confidence là tỷ lệ phiếu,
còn hai cách tổng hợp probability lấy confidence từ phân phối tổng hợp.
Chi tiết nằm trong [tài liệu Document Aggregation](DOCUMENT_LEVEL_AGGREGATION.md).
FastAPI đã triển khai ở Bước 12; chưa triển khai React.
Chi tiết artifact, cache, mapping và injection nằm trong
[tài liệu Real SVM](REAL_SVM_INTEGRATION.md).
Kiến trúc/tokenizer gốc và numerical consistency CPU được ghi trong
[tài liệu Real Transformer](REAL_TRANSFORMER_INTEGRATION.md).

## Production service và một kết quả chính — Bước 11

`backend/src/prediction_service.py` là tầng nghiệp vụ cho API sau này.
Service inject rõ hai raw predictors thật vào pipeline sentence/document cũ.
Không gọi clean_text ở service/selector/config; không đưa raw input vào cleaned-batch APIs.
Giữ raw input, thứ tự câu; input sai/cleaned-empty báo lỗi, không bỏ qua câu.

`deployment_config.py` đọc Validation JSON/CSV, áp dụng saved tie-break và kiểm tra
model/code fingerprints cùng frozen selection Bước 10. Config thật chọn ensemble,
alpha0.8. Test Macro-F1 chỉ là metadata nghiên cứu, không tham gia lựa chọn.

`select_result` validate cả ba prediction rồi deep-copy prediction của configured method,
không chọn theo confidence. Final một câu trùng chính xác selected sentence prediction;
đoạn nhiều câu dùng Mean Probability của selected method, không dùng câu cuối.
Mean Probability chưa được kiểm chứng tối ưu trên benchmark document-level.

Canonical response có input_text/sentence_count/sentences/document_results/final_result,
selection_info/aggregation_info/comparison/display_info/research_info. Mỗi câu có final_result
và comparison ba phương án. `present_result(response, 'user')` chỉ trình bày final result
và kết quả cuối từng câu; research trình bày toàn bộ dữ liệu đã tính. Frontend giữ canonical
response để chuyển view không inference lại. Macro-F1 dataset không phải confidence input.

Chi tiết schema, lỗi, giới hạn và cách dùng ở [Deployment Selection](DEPLOYMENT_SELECTION.md).
Service Bước 11 giữ nguyên; tầng HTTP Bước 12 mô tả bên dưới. Chưa triển khai React.

## FastAPI HTTP contract — Bước 12

`api.py` dùng lifespan khởi tạo một PredictionService cho mỗi application worker và
xác minh cached models CPU ready. Không checkpoint load mỗi request; inference lock
tuần tự hóa analyze. GET health trả liveness/readiness, GET model-info trả public metadata
đã cache ở startup; không expose paths/fingerprints hoặc mở CSV trong request.

POST /api/analyze nhận duy nhất raw text; schema strict, không preprocessing/truncate và
không chấp nhận override alpha/method/model path. Gọi analyze đúng một lần, trả canonical
response nguyên nghĩa/full precision. React giữ response để đổi view, không inference lại.

Input/schema/Unicode/giới hạn ký tự hoặc cleaned-empty theo contract gốc trả422;
chưa ready503; inference/output lỗi500 an toàn, không prediction giả hay raw-text traceback.
CORS explicit Vite localhost, không credentials. Mean Probability vẫn chưa validated optimal.

API dependencies ở requirements-api.txt, bao gồm base requirements.txt giữ nguyên bytes
vì frozen fingerprint Bước 9. Không thay đổi verifier/artifact/model chỉ để thêm dependency.
Windows run: `python -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000`.
Chi tiết [FastAPI Backend](FASTAPI_BACKEND.md). Dừng Bước 12, chưa có React.
