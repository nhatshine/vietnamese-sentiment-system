# Bước 11 — Final Prediction Pipeline & Deployment Selection

## 1. Mục tiêu và phạm vi

Backend service trả một kết quả chính theo phương án đã chọn trên Validation, đồng thời giữ các kết quả nội bộ để xem so sánh. Không xây FastAPI/React trong bước này; không thay model, preprocessing, vocabulary hay kết quả nghiên cứu.

Production dùng `PredictionService` trong `backend/src/prediction_service.py`. Hai predictor mặc định là `real_svm.predict_svm` và `real_transformer.predict_transformer`. Các analyzer/demo cũ giữ nguyên default mock để tương thích regression; chỉ truyền fake predictors khi test hoặc demo có chủ đích.

## 2. Lựa chọn từ nghiên cứu đã khóa

| Phương án | Validation Macro-F1 | Test Macro-F1 — chỉ để báo cáo |
|---|---:|---:|
| svm | 0.8241494288 | 0.8186049763 |
| transformer | 0.7928055628 | 0.7943039909 |
| ensemble | 0.8255456672 | 0.8221833733 |

`deployment_config.py` đối chiếu alpha_selection.json với alpha_scores.csv: grid, endpoints, score tối đa, tied_alphas, mapping và rule đã lưu. Không chạy lại sweep. Alpha được đọc thật là **0.8**, phương án được chọn là **ensemble**, có Validation Macro-F1 cao nhất.

Tie-break đúng rule Bước 9: khi Macro-F1 bằng nhau chính xác, ưu tiên endpoints; alpha0 trước alpha1, sau đó alpha tăng dần. Nếu endpoint được chọn và tương đương model đơn thì alpha0 tương ứng Transformer, alpha1 tương ứng SVM. Không chọn method bằng confidence của input.

Khi có `evaluation_freeze.json` của Bước 10, kiểm tra selected method/alpha, hash selection, Validation scores và fingerprints khớp cấu hình đã khóa trước Test. Cấu hình hiện tại xác nhận `frozen_selection_verified=true`.

Test không tham gia quyết định. `test_metrics.json` chỉ cung cấp metadata hiển thị sau khi quyết định đã xác định từ Validation; kể cả Test có xếp hạng khác, lựa chọn vẫn giữ nguyên. Loader không mở Test CSV, cache probability hoặc chạy lại evaluation.

## 3. Cấu hình triển khai

Tệp `backend/config/deployment_selection.json` được tạo bằng:

```powershell
python -m backend.src.deployment_config
```

Các policy chính:

```json
{
  "selection_source": "validation",
  "selection_metric": "macro_f1",
  "selected_method": "ensemble",
  "selected_alpha": 0.8,
  "default_view": "user",
  "comparison_view_available": true,
  "document_aggregation_default": "mean_probability",
  "aggregation_validated_optimal": false
}
```

Tệp thật còn có Validation/Test Macro-F1, tie-break, hashes của artifacts và fingerprints model/code. Loader kiểm tra policy, so với artifact nghiên cứu và hash model/code hiện tại trước khi khởi tạo service. Không load checkpoint khi import. Thiếu/corrupt/mismatch artifact thì báo lỗi; không fallback alpha0.5 hoặc model khác. Không chỉnh tay selection trong JSON: loader sẽ từ chối nếu khác nghiên cứu đã khóa.

Service giữ snapshot cấu hình riêng; sửa dictionary mà caller nhận từ `service.config` không đổi alpha nội bộ. Không hot-reload model/cấu hình trong request. Tái khởi tạo service để kiểm tra cấu hình mới trong quy trình triển khai có chủ đích.

## 4. Result Selector

`select_result(predictions, selected_method)` kiểm tra cả ba phương án: method nếu có, label, confidence, probabilities và tie-break nhãn theo contract. Sau đó trả deep copy nguyên prediction đã chọn, không thay probability hoặc metadata.

Không chọn confidence lớn nhất vì confidence của các model không đảm bảo hiệu chuẩn tương đương và không xác định model nào đúng với từng input. Validation chọn một phương án cố định cho pipeline. Ví dụ model đơn có thể confidence cao hơn Ensemble nhưng final vẫn phải là Ensemble. Không triển khai dynamic model selection chưa được thực nghiệm.

## 5. Input contract và preprocessing

- Input phải là `str`; non-string → `TypeError`, chuỗi rỗng/whitespace → `ValueError` trước predictor.
- `input_text` giữ nguyên raw string, gồm Unicode, dấu câu, xuống dòng và khoảng trắng.
- Splitter cũ chỉ trim biên từng câu, bảo vệ viết tắt/số thập phân theo các luật đã có; không thay splitter trong bước này.
- Service/selector/config không gọi clean_text. Raw paragraph → raw sentences → raw predictors. Mỗi predictor tự preprocessing một lần theo pipeline đã huấn luyện; không đưa raw text vào cleaned-batch APIs.
- SVM chạy trước và chặn câu cleaned-empty, ví dụ `!!!`. Service thêm tên model và raw sentence vào lỗi; toàn request thất bại, không bỏ qua câu rồi trả kết quả thiếu.
- Văn bản dài vẫn giữ nguyên trong response. Transformer áp dụng max_len128 đã train (1 CLS, tối đa127 token nội dung); service không tự cắt raw text hay đổi cấu hình.
- Không dùng Test để inference văn bản người dùng; không có label/source dataset làm input feature.

## 6. Pipeline một câu

```text
Raw text → splitter → raw sentence → Real SVM + Real Transformer
         → weighted_ensemble(alpha từ config) → select_result → final_result
```

Tái sử dụng `analyze_document`/sentence analyzer với hai real predictor injection. Dù có tạo document aggregations nội bộ, final của một câu lấy chính xác prediction câu từ selected method; không round hoặc thay confidence bằng tỷ lệ phiếu.

## 7. Pipeline nhiều câu và aggregation

Mỗi raw sentence được gọi đúng một lần qua mỗi predictor. Các kết quả đã tính được dùng để tổng hợp; không inference lại toàn đoạn hoặc inference lần hai cho final.

`document_results` giữ ba chiến lược riêng cho SVM, Transformer, Ensemble: `majority_vote`, `mean_probability`, `confidence_weighted`. Mặc định final lấy Mean Probability của selected method. Không dùng câu cuối hoặc ngầm majority voting.

`comparison` ở cấp response là ba document predictions của aggregation mặc định, bổ sung field `method`; nhãn/confidence/probabilities giữ đúng kết quả aggregation gốc. `document_results` giữ nguyên cấu trúc của aggregator cũ. Với majority vote, confidence là tỷ lệ phiếu và có `vote_distribution`, không giả là model probability.

Mean Probability là **quy ước demo/triển khai**, chưa được kiểm chứng tối ưu bằng benchmark có nhãn document-level. Trường `aggregation_info.validated_optimal` và config `aggregation_validated_optimal` đều false.

## 8. Output contract

Canonical response luôn đầy đủ và JSON serialize được:

| Field | Ý nghĩa |
|---|---|
| input_text | Raw input giữ nguyên |
| sentence_count | Số câu từ splitter |
| sentences | Theo thứ tự, có sentence_index 0-based, sentence, final_result và comparison ba phương án |
| document_results | Tất cả ba phương án × ba aggregation |
| final_result | Một prediction chính từ selected method; đoạn nhiều câu dùng Mean Probability |
| selection_info | selected_method, selected_alpha, selected_by=validation_macro_f1 |
| aggregation_info | method, applied_to_final, validated_optimal=false và policy |
| comparison | Ba prediction đủ method/label/confidence/probabilities |
| display_info | default_view=user, comparison_view_available=true |
| research_info | Validation/Test Macro-F1, Ensemble weights, metric_scope và frozen verification |

`final_result` bằng chính prediction trong `comparison[selected_method]` (deep copy độc lập). Với đoạn nhiều câu, prediction có thêm aggregation_method/model; các câu vẫn giữ kết quả cuối riêng.

Macro-F1 trong research_info là metric trên toàn dataset, **không phải confidence của input hiện tại**. Confidence cuối nằm riêng trong final_result. Không thay đổi probability để tạo demo đẹp; không bảo đảm prediction luôn đúng, nhất là input cảm xúc hỗn hợp.

## 9. User và Research chỉ khác cách trình bày

```python
from backend.src.prediction_service import PredictionService, present_result

service = PredictionService()  # tạo một lần cho ứng dụng
response = service.analyze("Camera đẹp. Nhưng pin yếu.")
user_view = present_result(response)             # mặc định user
research_view = present_result(response, "research")  # không gọi model
```

User view có một final result, selection/aggregation metadata và kết quả cuối từng câu. Không có comparison, document_results hoặc research_info trong phần trình bày này. Giao diện tương lai hiển thị ô nhập, nút phân tích, nhãn/confidence, biểu đồ ba xác suất và lựa chọn xem từng câu/so sánh.

Research view có đủ ba predictions, metadata nghiên cứu và mọi aggregation. Frontend phải giữ canonical response trong state; chuyển view đọc dữ liệu này, không gửi request phân tích lại. Hai view có cùng final_result và không mutate canonical response.

## 10. Demo thật đã chạy

```powershell
python -m unittest discover backend/tests
python -m backend.demo_final_pipeline
python -m backend.demo_final_pipeline --research
python -m backend.demo_final_pipeline --both --save-json backend/outputs/final_pipeline/demo_results.json
```

Lần kiểm chứng dùng `--both`: mỗi input được analyze một lần rồi render user/research từ cùng response. Năm câu và đoạn ba câu tương ứng tổng8 lần gọi raw predictor cho mỗi model; không16 lần chỉ vì hai view. Checkpoint được cache bởi các predictor cũ.

| Input | Final label | Confidence |
|---|---|---:|
| Sản phẩm này rất tốt. | positive | 0.99360003 |
| Tôi rất thất vọng vì sản phẩm quá tệ. | negative | 0.98413348 |
| Sản phẩm này bình thường. | neutral | 0.55867315 |
| Camera rất đẹp nhưng pin quá yếu. | positive | 0.70830317 |
| Giao hàng chậm và đóng gói cẩu thả. | negative | 0.90565771 |

Đoạn `Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!`:

- Final document: positive, confidence0.67243147; Mean Probability của Ensemble.
- Probabilities negative/neutral/positive: 0.2919011383 / 0.0356673857 / 0.6724314729.
- Từng câu: positive0.98672463 / negative0.85005972 / positive0.97148594.

Đây là output thật, không phải ground truth đánh giá hay nhãn được bảo đảm đúng. JSON đầy đủ giữ full precision ở `backend/outputs/final_pipeline/demo_results.json`.

## 11. Hợp đồng cho FastAPI/React sau này

FastAPI sẽ tạo/reuse service và trả canonical response, ánh xạ lỗi input thành phản hồi phù hợp trong Bước12. React Bước13 mặc định chỉ hiển thị final_result và có thể mở phân tích câu. Khi bật research, đọc comparison/research_info/document_results của response đã giữ. Không lấy confidence lớn nhất giữa model và không hiển thị Macro-F1 như confidence.

Chưa tạo endpoint, HTTP server hoặc React component. Không tuyên bố aggregation tối ưu. Dừng tại Bước11.

## 12. Kiểm chứng hoàn thành

370/370 unit tests PASS, gồm329 regression tests cũ và41 tests mới. Rà soát độc lập đã kiểm tra core service/config/selector và kiểm tra sửa lỗi frozen label/dataset contract. Instrumented real demo ghi nhận8raw-inference/preprocessing calls mỗi model và1checkpoint load mỗi model cho cả hai view.

`backend/outputs/final_pipeline/demo_console.txt` lưu console hai view; `demo_results.json` giữ canonical full precision; `demo_verification.json` ghi call counts thực. Hash audit105tệp cũ xác nhận chỉ README và ModelIntegrationContract đổi, không xóa tệp; code/models/tests/data và mọi kết quả Bước9/9.1/10 giữ nguyên bytes.
