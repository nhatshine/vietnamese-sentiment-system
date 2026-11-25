# Hệ thống demo Transformer epoch 21

Phiên bản hiện tại nạp trực tiếp checkpoint epoch 21 từ ZIP bàn giao, giữ nguyên SVM và dùng Weighted Ensemble với 70% SVM + 30% Transformer. Alpha 0,7 được chọn bằng Macro-F1 trên Validation trong dải 0–1, bước 0,1. Không dùng Test để chọn trọng số và không huấn luyện lại khi tích hợp.

## Mở hệ thống

Nháy đúp `C:\Nhom3_NLP\CHAY_HE_THONG.cmd`. Đợi backend nạp mô hình rồi trình duyệt sẽ mở tại http://localhost:5173. File chạy bên trong project là `C:\Nhom3_NLP\Kethoptrongso\CHAY_HE_THONG.cmd`.

Nếu hệ thống đang chạy, chỉ cần mở http://localhost:5173. Chuyển sang **Nghiên cứu**, kéo xuống phần phương pháp kết hợp để xem phiên bản Transformer epoch 21 và trọng số 70/30. Điểm Macro-F1 hiển thị là điểm toàn tập, không phải confidence của văn bản vừa nhập.

Backend: http://127.0.0.1:8000; trạng thái: http://127.0.0.1:8000/api/health; cấu hình đang phục vụ: http://127.0.0.1:8000/api/model-info. Nếu cổng 8000 đang chạy mô hình khác, launcher dừng và báo lỗi thay vì âm thầm demo sai phiên bản.

## Kết quả của phiên bản này

| Phương pháp | Validation Macro-F1 | Test Macro-F1 |
|---|---:|---:|
| SVM | 0,824149 | 0,818605 |
| Transformer epoch 21 | 0,807684 | 0,804910 |
| Max Confidence | 0,820965 | 0,821084 |
| Weighted Ensemble 70/30 | 0,824976 | 0,825271 |

Tổng hợp toàn đoạn vẫn mặc định Mean Probability: lấy trung bình xác suất từng câu của phương pháp được chọn. Chưa có bộ nhãn chuẩn toàn đoạn để khẳng định cách tổng hợp này tối ưu. Ví dụ “Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!” cho nhãn tích cực, confidence 67,59%; đây là kết quả ví dụ, không phải độ chính xác toàn tập.

## Bằng chứng và kiểm thử

- Checkpoint và các tệp gốc: `backend/models/transformer_epoch21`. Kiến trúc, từ vựng và tiền xử lý tương thích với bộ nạp CPU đã tích hợp; kiểm tra trọng số bằng hàm băm, nạp strict và weights_only.
- Cấu hình hiện hành: `backend/config/deployment_selection.json`. Bản khóa và số liệu phiên bản mới: `backend/outputs/epoch21_deployment`. Bản khóa là ảnh chụp cấu hình triển khai, không được trình bày như bằng chứng đăng ký thí nghiệm từ trước.
- `cpu_verification.json`: suy luận thật trên 64 mẫu Validation và 64 mẫu Test khớp nhãn với dự đoán bàn giao, sai lệch xác suất tối đa dưới 4,2×10⁻⁷. `bang9_inference_epoch21.csv`: ba ví dụ suy luận cho Bảng 9.
- `backend-regression.log`: 449 kiểm thử backend đạt. Frontend: 50 kiểm thử đạt, typecheck và production build đạt.
- `frontend/e2e-results-epoch21/integration.json`: năm đầu vào qua backend thật, không lỗi trình duyệt; đổi chế độ không gọi lại mô hình. Kiểm tra chiều rộng 360, 390, 768, 1366 và 1920 px; ảnh thực tế ở cùng thư mục.
- Mô hình cũ, dữ liệu và kết quả thực nghiệm lịch sử được giữ nguyên. Bản sao sáu tệp trước khi chỉnh sửa nằm trong `backend/outputs/epoch21_deployment/backup`.
- Nhật ký dịch vụ: `backend/outputs/epoch21_deployment/runtime`. Đóng trang web không dừng hai dịch vụ; khởi động lại máy thì chạy lại file CMD.

## Chạy lại kiểm tra

Từ thư mục project, dùng Python của môi trường có torch và scikit-learn:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -X utf8 -m unittest discover -s backend/tests -q
python -X utf8 -m backend.verify_epoch21_demo
```

Trong thư mục frontend:

```powershell
npm run typecheck
npm test
npm run build
$env:E2E_RESULTS_DIR='e2e-results-epoch21'
npm run test:e2e
```

Kiểm thử trình duyệt cần Chrome và quyền truy cập dịch vụ local. Các số liệu đánh giá đầy đủ vẫn ở `C:\Nhom3_NLP\Ket_qua_ket_hop_Transformer_epoch21`.
