# Bước 13 — React Frontend hoàn chỉnh

## 1. Giới thiệu

Giao diện tiếng Việt cho demo NLP, kết nối FastAPI thật của Bước12. Người dùng nhập văn bản và nhận một kết quả cuối, có thể xem từng câu hoặc mở góc nhìn nghiên cứu. Không sửa NLP/model hoặc biến ứng dụng thành dashboard quản trị/notebook.

Đã kiểm tra project trước khi xây: frontend chỉ có `.gitkeep`, backend entrypoint `backend.src.api:app`, ba endpoints `/api/health`, `/api/model-info`, `/api/analyze`; canonical schema và JSON thực tế được đọc từ các files/API Bước11–12. `.gitkeep` được giữ nguyên.

## 2. Công nghệ

React19.3.1, Vite8.3.5, TypeScript7.0.2, Lucide1.64.0; CSS thuần, Fetch, hooks. Vitest5.0.5 + React Testing Library16.3.3 + jsdom30.1.3; Playwright1.64.1 chỉ dùng E2E. Node24.15.0/npm11.12.1 được kiểm chứng. package-lock.json khóa dependency thực; không cài global, không nâng cấp môi trường Python.

Production build khoảng256kB JS (~80kB gzip), không thư viện UI/chart nặng. Không tải font, ảnh hay model từ dịch vụ ngoài khi sử dụng app; system sans-serif hỗ trợ tiếng Việt, icons SVG.

## 3. Cấu trúc component

```text
frontend/
  src/
    App.tsx, App.css, main.tsx
    components/
      Header, TextInput, SentimentResult, ProbabilityBars
      SentenceAnalysis, ModelComparison, ModelMetrics
      EnsembleExplanation, DocumentAggregation, ErrorMessage
    hooks/useAnalysis.ts
    services/api.ts, contracts.ts
    types/sentiment.ts
    utils/formatters.ts
    test/fixtures/                 # chỉ test
    App.test.tsx, services/api.test.ts
  e2e/real-backend.spec.ts
  e2e-results/                     # kiểm chứng public demo, không runtime history
  package.json, package-lock.json
  vite.config.ts, playwright.config.ts, tsconfig.json
  .env.example
```

App tổ chức view/metadata; hook quản lý inference state; mỗi component có trách nhiệm giới hạn. Không có Redux, login, database, upload hoặc chức năng train.

## 4. Luồng người dùng và canonical state

Nhập text → validate → một POST → parse response → lưu nguyên canonical object → render final_result. `inputText`, `canonicalResponse`, `viewMode`, `isAnalyzing`, `error`, `modelInfo` tách biệt.

Textarea giữ raw text/dấu câu/xuống dòng; counter đếm Unicode code points giống Python len(), không dùng maxLength để âm thầm cắt paste. Giới hạn mặc định20.000 có thể cấu hình. Nút ví dụ chỉ điền text; không gán prediction và không auto-submit. Ctrl+Enter hỗ trợ phân tích, có focus visible và nút Xóa/Hủy.

Sửa text hoặc Xóa sẽ xóa kết quả cũ, abort request và tăng generation ID. Dù fetch cũ không tuân thủ abort và trả response muộn, generation/input echo check vẫn chặn kết quả sai. In-flight ref chặn double submit ngay trong cùng event batch. Thay mode không liên quan đến submit/effects inference.

## 5. API integration và boundary

`services/api.ts` là nơi duy nhất chứa base URL, mặc định `http://127.0.0.1:8000/api`. Dùng `analyzeText`, `getModelInfo`, `checkHealth`, credentials=omit; POST chỉ gửi `{text}`. Không gửi alpha/method/model path/view. GET health/info lúc mount, không mỗi render/tab.

Types đối chiếu OpenAPI/canonical JSON thật. Runtime boundary kiểm tra nhãn, method, finite probability/confidence, đủ ba phương án, selected/final consistency, sentence count/order và document aggregation contract. Giữ object gốc/full precision/extra metadata; validation không normalize, không chọn nhãn, không tính Ensemble.

Metadata thiếu không tạo số liệu giả: Research có thể dùng research_info của response thật nếu GET model-info lỗi. Nếu cả hai chưa có thì hiện thông báo không có số liệu. Health badge phản ánh kết nối; POST thành công xác nhận readiness và chặn health response startup đến muộn ghi sai trạng thái.

## 6. User Mode

Mặc định `user`. Card chính chỉ đọc final_result.label/confidence/probabilities và mapping negative=Tiêu cực, neutral=Trung tính, positive=Tích cực. Không chọn confidence lớn nhất của ba model, không lặp ba prediction, không đưa Macro-F1 vào confidence card.

ProbabilityBars đọc ba key theo LABELS cố định, không dựa thứ tự JSON. Chuyển probability sang phần trăm chỉ khi render; ví dụ0.8542→85,42%. Không chỉnh tổng phân phối khi sai khác rất nhỏ do floating point. Nhãn có chữ và icon/màu, không dùng màu đơn độc.

## 7. Research Mode

Giữ nguyên input và final_result, bổ sung ba model cards và các phần nghiên cứu. Comparison đọc response.comparison; phương án selected từ selection_info được đánh dấu, không chọn lại theo confidence. Trọng số lấy research_info/selected_alpha từ Backend, không cho sửa alpha.

Mode switch hỗ trợ Tab và ArrowLeft/Right/Home/End. Canonical response không bị mutate hoặc fetch lại khi chuyển chế độ. Không endpoint inference riêng cho Research.

## 8. Sentence-level UI

Chỉ hiện khi sentence_count>1. Danh sách theo backend sentence_index/sentence, mỗi card lấy final_result riêng và có details/summary mở probabilities. Không split câu bằng JS, không keyword sentiment, không gộp hoặc bỏ câu.

Ca `Giá sản phẩm là 1.5 triệu đồng. TS. Nam đánh giá chất lượng tốt.` hiển thị đúng hai câu backend: `Giá sản phẩm là 1.5 triệu đồng.` và `TS. Nam đánh giá chất lượng tốt.` Không đặt kỳ vọng sentiment cho ca này.

## 9. Document-level UI

Card chính vẫn đọc Backend final_result của toàn đoạn, không lấy câu cuối hoặc tự majority vote. Tên phương án và aggregation đọc selection_info/aggregation_info; Mean Probability được giải thích là tổng hợp câu theo policy hiện tại, không hard-code một quyết định NLP khác.

Research DocumentAggregation có model tabs và ba hàng Majority Voting/Mean Probability/Confidence-Weighted Aggregation. Tất cả giá trị từ document_results. Mean Probability được đánh dấu đang áp dụng theo metadata, chưa được chứng minh tối ưu trên document benchmark. Majority confidence ghi rõ `% phiếu`, không gọi là xác suất mô hình; vote_distribution không bị nhầm với probabilities.

## 10. Weighted Ensemble

Research hiển thị công thức typography `P_Ensemble(c)=αP_SVM(c)+(1−α)P_Transformer(c)`, trọng số từ Backend, thanh tỷ lệ và giải thích Weighted Soft Voting. Đây là giải thích dữ liệu cấu hình, không có hàm tính prediction mới trong React. Không nhầm Hard Voting/Stacking, không điều chỉnh alpha và không dynamic model selection.

## 11. Metrics thực nghiệm

Grouped bars Validation/Test theo ba model, trục0–1, kèm bảng Macro-F1 chính xác sáu chữ số sau dấu chấm. Data source là `/api/model-info` hoặc canonical research_info, không constants chứa các F1 nghiên cứu trong production.

Kiểm chứng UI đọc đúng:

| Model | Validation | Test |
|---|---:|---:|
| SVM | 0.8241494288 | 0.8186049763 |
| Transformer | 0.7928055628 | 0.7943039909 |
| Ensemble | 0.8255456672 | 0.8221833733 |

Đây là số liệu đối chiếu trong tài liệu, không phải source frontend. UI ghi rõ Validation chọn trọng số, Test chỉ đánh giá cuối; chưa tuyên bố ý nghĩa thống kê, không gọi F1 là confidence input.

## 12. Loading/error/cancellation

Trạng thái ban đầu có hướng dẫn, loading có spinner/aria-busy và không prediction giả. Submit disabled khi đang chờ, rỗng hoặc vượt giới hạn; lỗi vẫn được guard ở hook nếu submit qua keyboard/programmatic form.

Client dùng AbortController với timeout mặc định120s. Thông báo tiếng Việt riêng cho network,422,503,500, timeout, cancellation và contract mismatch. Không hiển thị stack trace/detail nội bộ của server; không fallback sample response. Request lỗi xóa prediction thay vì giả neutral. Aborting fetch chỉ hủy chờ ở browser; không bảo đảm Backend dừng phép inference CPU đã bắt đầu.

## 13. Responsive và accessibility

Đã render Chrome thật ở1366×768, FullHD1920, tablet768 và mobile390/360. Không có page horizontal overflow; bảng aggregation có vùng cuộn riêng. Laptop thấp được thu gọn hero/textarea để nút phân tích nằm trọn viewport; mobile xếp dọc, chart thu gọn và bảng vẫn đọc bằng text.

Đã xem ảnh thực desktop, mobile và tablet; tăng contrast/chữ phụ sau lượt QA đầu. Textarea có label/description, button/tab có accessible names, keyboard navigation/focus, status/error live regions, meter có nhãn/giá trị. Reduced-motion được tôn trọng. Không tuyên bố đã audit WCAG đầy đủ.

## 14. Testing đã thực hiện

- 49/49 Vitest/RTL tests PASS: contract/client/formatting, labels/bars, one POST, no mode inference, sentences/document, global metrics, errors/loading, double submit, stale/abort/race, keyboard và health response đến muộn.
- TypeScript và production build PASS.
- Playwright Chrome E2E:1/1 PASS, năm input thật → đúng5POST. Mỗi lần chuyển User/Research và model aggregation tab:0POST thêm. CORS thực đúng origin localhost5173, không pageerror.
- 396/396 backend regression tests PASS, không chỉnh tests/backend cũ.

Hai công cụ browser tương tác của phiên làm việc bị lỗi khởi tạo kernel assets. Đã thay bằng test runner Playwright cục bộ dùng Chrome đã có, không tải browser mới. Screenshots được kiểm tra trực tiếp từ render thật, không phải mockup. Test runner là dev dependency, không vào bundle production.

`e2e-results/integration.json` giữ actual responses của năm public input, metrics từ API và POST counters. Các PNG giữ desktop-empty/user/research, mobile-top/result/research, FullHD và tablet. Fixtures chỉ nằm trong test imports, không production. API-response reference cho E2E lấy từ response thật; không inference thêm chỉ để tạo expected.

## 15. Chạy local và build

Terminal1:

```powershell
cd C:\sentiment_system
python -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000
```

Terminal2:

```powershell
cd C:\sentiment_system\frontend
npm ci
npm run dev
```

Frontend http://localhost:5173, Backend http://127.0.0.1:8000, Swagger http://127.0.0.1:8000/docs. CORS Backend hiện có đã cho đúng origin, không cần sửa backend.

```powershell
npm run typecheck
npm test
npm run build
npm run test:e2e
```

E2E cần Chrome local và quyền process/loopback; config tự reuse/start hai servers. Trong Codex Windows sandbox, các lệnh server/HTTP/test backend cần quyền loopback ngoài sandbox; không có thay đổi model vì giới hạn môi trường.

`.env.example` có VITE_API_BASE_URL, VITE_MAX_TEXT_CHARS, VITE_REQUEST_TIMEOUT_MS. Giới hạn frontend phải khớp Backend nếu đổi SENTIMENT_MAX_TEXT_CHARS. Không đặt secret vào VITE_ variables. Vite preview dùng4173: muốn gọi API ở preview thì thêm origin4173 qua SENTIMENT_CORS_ORIGINS trước khi chạy Backend, hoặc dùng dev5173. Không mở dist bằng file://.

## 16. Demo cho giảng viên và kết quả thật

1. Mở User Mode, chọn ví dụ hoặc nhập văn bản, nhấn phân tích; giới thiệu một final result và ba probabilities.
2. Nhập đoạn ba câu, mở từng câu để thấy sắc thái hỗn hợp và final toàn đoạn.
3. Chuyển Research mà không submit: xem ba model, frozen weights, F1 datasets và aggregation policy.
4. Giải thích Validation chọn mô hình/alpha, Test chỉ đánh giá cuối, confidence không bảo đảm đúng.

| Input kiểm chứng | Backend/UI label | Confidence | Số câu |
|---|---|---:|---:|
| Sản phẩm này rất tốt. | positive | 0.9936000322 |1|
| Tôi rất thất vọng vì sản phẩm quá tệ. | negative | 0.9841334794 |1|
| Sản phẩm này bình thường. | neutral | 0.5586731473 |1|
| Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh! | positive | 0.6724314729 |3|
| Giá sản phẩm là 1.5 triệu đồng. TS. Nam đánh giá chất lượng tốt. | positive | 0.5735625047 |2|

Ca cuối chỉ đối chiếu UI/API, không khẳng định ground-truth sentiment. Bốn input đầu khớp kết quả đã lưu Bước11/12, không chỉnh nhãn. Đoạn ba câu hiển thị positive/negative/positive đúng thứ tự, probabilities cuối0.2919011383/0.0356673857/0.6724314729. Full precision vẫn giữ trong canonical state.

## 17. Giới hạn và phạm vi

Không cloud/VPS, auth, database, upload hay train web. Các giới hạn của model, truncation, confidence và aggregation vẫn như backend. Metadata/health là snapshot, không polling liên tục; lỗi request hiển thị tại chỗ. Frontend không hủy được computation đã chạy ở server. Test layout là browser tự động/screenshot, không thay thế toàn bộ manual/assistive-device audit. Không thay backend để khớp UI; dừng ở Bước13.

Kiểm tra bàn giao:127tệp gốc được đối chiếu SHA256, không xóa tệp, chỉ README thay đổi. Toàn bộ backend/config/model/data/tests và kết quả Bước9–12 byte-identical; frontend/.gitkeep giữ nguyên. Hai server kiểm thử được dừng sau QA; dùng hai terminal ở mục15 để mở ứng dụng.
