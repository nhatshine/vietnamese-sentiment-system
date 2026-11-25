# Bước 12 — FastAPI Backend

## 1. Mục tiêu và kiến trúc

FastAPI chỉ là tầng HTTP của `PredictionService` Bước11. Không viết lại model, preprocessing, Ensemble, splitter hay aggregation. API trả canonical response đầy đủ một lần; frontend giữ response và tự chuyển User/Research view.

Ba module mới:

- `backend/src/api.py`: application factory, lifespan, ba routes, CORS, lỗi và inference lock.
- `backend/src/api_dependencies.py`: factory service thật, xác minh readiness và whitelist metadata.
- `backend/src/api_schemas.py`: strict request và response contract tương thích output Bước11.

Không có React, database, authentication, upload, background workers hoặc dịch vụ bên ngoài.

## 2. Môi trường và dependency đã kiểm chứng

Python3.13 trên Windows CPU; FastAPI0.141.1, Starlette1.7.0, Pydantic2.13.2, Uvicorn0.54.0 và httpx0.28.1 đã có sẵn. Không cài/nâng cấp package toàn hệ thống trong bước này. SVM/PyTorch giữ nguyên môi trường đã kiểm chứng trước đó.

**Dependency API nằm ở `backend/requirements-api.txt`, bao gồm `-r requirements.txt`.** Base requirements.txt được Bước9 ghi hash và Bước9.1/10/11 kiểm tra đúng bytes. Thêm dòng vào base sẽ làm deployment/regression bị từ chối; vì vậy giữ nguyên base và thêm manifest API riêng, không chỉnh frozen fingerprint để hợp thức hóa thay đổi.

Nếu cần chuẩn bị môi trường riêng, chạy trong virtual environment của project rồi cài manifest API. Có thể dùng `python -m venv .venv --system-site-packages` để kế thừa các model dependencies đã có trên máy này, sau đó activate `.venv` và `python -m pip install -r backend/requirements-api.txt`. Không dùng pip để nâng cấp môi trường toàn máy. Trong môi trường hiện tại lệnh chạy trực tiếp bên dưới đã được kiểm chứng.

Starlette hiện phát deprecation warning cho backend TestClient dùng httpx0.28.1, nhưng các tests và HTTP thực đều pass với bộ phiên bản đã pin. Không thay stack chỉ để che warning; ứng dụng production Uvicorn không sử dụng TestClient.

## 3. Singleton, CPU và readiness

Import `api:app` không load checkpoint. Lifespan tạo một `PredictionService` cho application worker, kiểm tra config/fingerprints qua service gốc, rồi gọi hai loader đã cache để xác minh SVM và Transformer CPU sẵn sàng. Không warmup prediction, không preprocessing khi startup. Chỉ sau khi startup hoàn tất mới đặt `model_ready=true`.

Service và metadata được giữ trong `app.state`. Không tạo service hoặc đọc file metadata theo từng HTTP request. Các raw predictors vẫn chịu trách nhiệm preprocessing đúng một lần và cache weights/vocab. Không fallback mock nếu thiếu/corrupt model/config: startup fail với thông báo rõ ràng.

`threading.Lock` bảo vệ một lần `service.analyze(text)` trong mỗi request. Route sync chạy ở threadpool; nhiều request có thể đến nhưng inference được xử lý tuần tự để giữ trạng thái ổn định. Health/model-info không chạy inference. Dùng một worker cho demo CPU; nhiều worker tạo một bộ model mỗi process và tăng RAM, không phải cách tối ưu được đánh giá ở đây.

Health là liveness có readiness metadata. Nếu dependency chưa ready trong runtime thì analyze/model-info trả503; health vẫn có status=ok nhưng model_ready=false. Trong startup production lỗi nghiêm trọng thì Uvicorn không phục vụ application.

## 4. Cấu hình đã khóa

Selection lấy từ deployment JSON đã đối chiếu Validation JSON/CSV và freeze Bước10: ensemble, alpha0.8. Không nhận alpha/method/model path/view từ request. Test scores chỉ là metadata báo cáo; không chọn lại từ Test hoặc confidence lớn nhất.

`GET /api/model-info` chỉ trả whitelist: tên model, method/alpha/source/metric, Validation/Test Macro-F1, CPU và aggregation policy. Không công khai paths/fingerprints/checkpoint/env. Metadata lấy từ `service.config` đúng một lần lúc startup.

## 5. Endpoints

| Method | Path | Chức năng |
|---|---|---|
| GET | /api/health | Liveness và readiness thực tế |
| GET | /api/model-info | Frozen selection và metric dataset đã lưu |
| POST | /api/analyze | Phân tích raw text đúng một lần, trả canonical response |

Health production đã quan sát:

```json
{"status":"ok","service":"sentiment-analysis-api","version":"1.0.0","model_ready":true}
```

Model-info có selected_method=ensemble, selected_alpha=0.8, selection_source=validation, selection_metric=macro_f1. Validation Macro-F1 SVM/Transformer/Ensemble là 0.8241494288/0.7928055628/0.8255456672; Test tương ứng 0.8186049763/0.7943039909/0.8221833733. Giá trị lấy từ artifacts, không hard-code trong API.

## 6. Request và response

```json
{"text":"Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"}
```

Text phải là string Unicode hợp lệ, không rỗng/whitespace. Giữ raw text, dấu câu và xuống dòng; schema/API không clean_text hoặc truncate. Mặc định tối đa20.000 ký tự, cấu hình bằng `SENTIMENT_MAX_TEXT_CHARS` là số nguyên dương. Vượt giới hạn trả422, không cắt bớt. Giới hạn128 token trong Transformer giữ nguyên và là cơ chế riêng bên trong predictor.

Request chỉ có text; thêm alpha, selected_method, model_path hoặc view đều bị422. Không có endpoint inference riêng cho Research Mode.

Response giữ toàn bộ field Bước11: input_text, sentence_count, sentences, document_results, final_result, selection_info, aggregation_info, comparison, display_info, research_info. Mỗi câu có index/raw sentence/final/comparison. Document giữ ba phương án × ba aggregation. Majority Vote vẫn có vote_distribution, không bị ép vào schema probability model.

Schema kiểm tra method/label/probabilities/confidence bằng contract cũ, consistency final/selected method và số thứ tự câu. Trước trả HTTP, kiểm tra toàn bộ object serialize với allow_nan=false: không Tensor, file handle, callable hay NaN/Infinity. Không làm tròn xác suất. Metadata bổ sung hợp lệ được giữ nguyên.

## 7. User và Research Mode

React Bước13 mặc định chỉ trình bày final_result: nhãn, confidence và ba xác suất. Có thể mở kết quả cuối từng câu. Không mặc định hiển thị SVM/Transformer riêng.

Khi bật Research, đọc comparison/research_info/document_results từ response đang giữ trong state; không POST lại. Hai view dùng cùng prediction. Validation Macro-F1 dùng chọn mô hình, Test Macro-F1 báo cáo đánh giá cuối, confidence thuộc input mới; không đồng nhất ba khái niệm này.

Mean Probability là mặc định triển khai, `validated_optimal=false`; chưa có benchmark document-level để tuyên bố tối ưu.

## 8. Lỗi HTTP

| Status | Trường hợp |
|---:|---|
| 422 | JSON/schema sai, non-string, rỗng, Unicode sai, vượt giới hạn, request override policy |
| 422 | Câu trở thành rỗng sau preprocessing theo đúng error contract SVM hiện tại |
| 503 | Service/model/config chưa ready trong runtime |
| 500 | Inference hoặc canonical output lỗi; không có prediction thay thế |

ValueError bất kỳ trong model không tự coi là lỗi input. Chỉ nhận diện chính xác error cleaned-empty gốc qua cause chain. Không bỏ câu, không trả neutral giả hoặc HTTP200 khi lỗi. HTTP error không chứa paths hay stack trace.

Default log chỉ có loại lỗi, không raw text, exception string hoặc traceback của inference. Access log Uvicorn có method/path/status, không body. API không ghi lịch sử/log text vào file/database. Các tệp verification của bước này chỉ lưu public demo input do đề bài chỉ định, không phải tính năng lưu request runtime.

## 9. CORS

Allowlist mặc định: `http://localhost:5173`, `http://127.0.0.1:5173`. Không credentials/cookie/session; không wildcard. Có thể bổ sung origins HTTP(S) cụ thể bằng biến `SENTIMENT_CORS_ORIGINS`, phân tách dấu phẩy. Đây phải là origins thuần, không path/query.

```powershell
$env:SENTIMENT_MAX_TEXT_CHARS = "20000"
$env:SENTIMENT_CORS_ORIGINS = "http://localhost:3000"
```

CORS không phải authentication: client ngoài browser vẫn gọi HTTP được. Unknown origin không nhận Access-Control-Allow-Origin; preflight hợp lệ cho Vite đã kiểm tra200.

## 10. Chạy trên Windows

```powershell
cd C:\sentiment_system
python -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`. OpenAPI: `http://127.0.0.1:8000/openapi.json`. Swagger HTML và OpenAPI JSON đã trả200 qua HTTP thật; không custom Swagger UI. Frontend dự kiến `http://localhost:5173`.

Chỉ dùng `--reload` khi phát triển: mỗi reload tạo process/service/model mới. Không cần Docker. Dừng server bằng Ctrl+C.

Trong sandbox Windows của phiên kiểm chứng, asyncio bị chặn khi tạo loopback socket nội bộ, trước lifespan. TestClient và Uvicorn đã chạy thành công khi được phép loopback ngoài sandbox; không thay pipeline hoặc thư viện để né môi trường. Trên PowerShell Windows bình thường dùng các lệnh trên.

## 11. Gửi request bằng PowerShell

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/model-info

$body = @{ text = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!" } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/analyze" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

React dùng fetch POST JSON một lần rồi lưu canonical object trong state; không xây frontend ở bước này.

## 12. Kiểm chứng thật

396/396 tests PASS, gồm370 tests cũ và26 API tests mới. Uvicorn thật trên127.0.0.1:8000 đã startup với model thật, kiểm tra HTTP health/model-info/docs/OpenAPI đều200; lỗi empty/non-string/override/cleaned-empty đều422. Localhost preflight200; unknown origin không có allow-origin header.

POST đoạn ba câu dùng đúng SVM và Transformer thật, alpha0.8, final Ensemble Mean Probability:

```json
{
  "method": "ensemble",
  "label": "positive",
  "confidence": 0.6724314728874431,
  "probabilities": {
    "negative": 0.29190113827252445,
    "neutral": 0.03566738565413307,
    "positive": 0.6724314728874431
  },
  "aggregation_method": "mean_probability",
  "model": "ensemble"
}
```

Từng câu: positive0.9867246318 / negative0.8500597222 / positive0.9714859374. So với saved demo Bước11, max absolute probability difference của ba model theo câu và document là **0.0**. Không inference lại để tạo expected output.

`backend/outputs/api_verification/http_analyze.json` lưu canonical response của public demo; `http_verification.json` lưu các status, metadata và đối chiếu. Không chỉnh các artifacts Bước9–11.

## 13. Hạn chế

Inference được tuần tự hóa cho demo CPU, chưa benchmark throughput/production load. Không auth/rate-limit/persistence/upload. Swagger UI dùng các assets mặc định của FastAPI. Confidence không bảo đảm prediction đúng; truncation theo model vẫn tồn tại. Mean Probability chưa được chứng minh tối ưu. Dừng tại Bước12, chưa triển khai React/Bước13.

## 14. Bàn giao

Các Uvicorn process kiểm thử đã dừng sau xác minh; chạy lại bằng lệnh mục10 để dùng Swagger/API. Hash audit118tệp cũ: không xóa, chỉ README và ModelIntegrationContract thay đổi; toàn bộ oldPython/tests/checkpoint/data/deployment và artifacts Bước9–11 giữ nguyên. Rà soát độc lập đã hoàn tất; các ca Unicode biên được thêm regression tests và sửa, full396suitePASS.
