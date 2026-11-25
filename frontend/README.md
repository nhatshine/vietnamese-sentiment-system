# React Frontend — Phân tích cảm xúc tiếng Việt

React + Vite + TypeScript, CSS thuần và Lucide. Dữ liệu prediction/metrics đến từ FastAPI thật; production không import fixtures hoặc mock. Tài liệu đầy đủ: [REACT_FRONTEND](../docs/REACT_FRONTEND.md).

## Chạy local

Bản demo mới nhất: Transformer epoch 21, SVM 70% + Transformer 30%. Nháy đúp `C:\Nhom3_NLP\CHAY_HE_THONG.cmd` để mở cả backend và frontend. Chi tiết phiên bản: [EPOCH21_DEMO](../docs/EPOCH21_DEMO.md).

Terminal1 tại project root:

```powershell
cd C:\Nhom3_NLP\Kethoptrongso
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -X utf8 -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000
```

Terminal2:

```powershell
cd C:\Nhom3_NLP\Kethoptrongso\frontend
npm ci
npm run dev
```

Mở http://localhost:5173. Dependency đã cài trong lượt kiểm chứng; lần sau không cần install lại nếu lockfile không thay đổi. Không cài package global.

## Cấu hình

Copy `.env.example` thành `.env.local` nếu cần thay mặc định; restart Vite sau khi đổi env:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000/api
VITE_MAX_TEXT_CHARS=20000
VITE_REQUEST_TIMEOUT_MS=120000
```

Giới hạn frontend phải khớp SENTIMENT_MAX_TEXT_CHARS của Backend nếu đổi default. Không có secret trong VITE_ variables. Input không bị truncate.

## Kiểm tra

```powershell
npm run typecheck
npm test
npm run build
$env:E2E_RESULTS_DIR='e2e-results-epoch21'
npm run test:e2e
```

50 Vitest/RTL tests, 1 Playwright E2E test với 5 input thật. E2E dùng Chrome có sẵn (`channel=chrome`), cần quyền chạy browser và loopback; tự reuse/start các server local. Không tải browser hay model từ internet khi chạy tests. Chrome đã có trên máy kiểm chứng. Test backend độc lập: `python -X utf8 -m unittest discover -s backend/tests -q` từ project root (449 kiểm thử).

`npm run preview` phục vụ build ở http://localhost:4173. Nếu muốn gọi API từ preview, thêm origin đó vào SENTIMENT_CORS_ORIGINS rồi restart Backend; mặc định CORS chỉ cho Vite5173. Không dùng dist/index.html bằng file:// để kiểm tra API.

## Cấu trúc

- `src/services`: fetch client, safe errors/timeout/cancellation và runtime contract validation.
- `src/types`: canonical types lấy từ FastAPI Bước12.
- `src/hooks/useAnalysis.ts`: input/response/loading/error và request generation guard.
- `src/components`: nhập liệu, kết quả, probabilities, từng câu, research comparisons/metrics/aggregation.
- `src/test/fixtures`: bản sao JSON thật chỉ dành cho tests, không vào production imports.
- `e2e`: test browser với FastAPI/model thật; `e2e-results-epoch21`: ảnh và response hiện tại; `e2e-results`: kết quả lịch sử.

Mặc định User chỉ hiển thị final_result. Research đọc cùng canonical object trong state, không POST lại. Sửa input xóa kết quả cũ và invalidate request; response đến muộn không ghi đè. Không tính Ensemble/aggregation hoặc chọn confidence cao nhất trong frontend. Hệ thống chạy local, chưa hosting/cloud/database.
