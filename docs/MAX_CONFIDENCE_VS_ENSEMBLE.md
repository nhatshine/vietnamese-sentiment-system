# Bước 14.1 — Max Confidence so với Weighted Ensemble

## 1. Mục tiêu nghiên cứu

Kiểm tra liệu việc kết hợp xác suất của hai mô hình có mang lại kết quả tốt hơn cách chọn mô hình tự tin hơn cho mỗi văn bản. Chỉ đánh giá trên cùng 8.238 mẫu Validation, không thay đổi cấu hình triển khai.

## 2. Nguyên lý hai phương pháp

**Max Confidence:** lấy giá trị lớn nhất trong ba xác suất của từng mô hình, so sánh hai giá trị và chọn **toàn bộ phân phối xác suất** của mô hình có confidence cao hơn. Không trộn từng cột và không chỉ so nhãn thắng. Nếu hai confidence bằng nhau chính xác, ưu tiên SVM. Không làm tròn và không dùng ngưỡng gần bằng. Nếu các nhãn trong một phân phối hòa xác suất, chọn nhãn đầu tiên theo thứ tự negative, neutral, positive.

**Weighted Ensemble:** giữ nguyên P = 0,8 × P_SVM + 0,2 × P_Transformer cho từng nhãn, sau đó chọn argmax. Alpha cố định 0,8 từ Bước 9, không sweep lại và không dùng Test để chọn. Hai phương pháp đều dùng đủ cùng hai ma trận đầu vào.

## 3. Kiểm tra project và quy trình thực nghiệm

- SVM cung cấp `predict_proba`, ánh xạ cột bằng `classes_` trong `backend/src/real_svm.py`. Transformer dùng softmax logits và ánh xạ `label_map.json` trong `backend/src/real_transformer.py`. Cả hai adapter đưa về negative=0, neutral=1, positive=2.
- Validation đã làm sạch, không làm sạch lại, không tách câu lại. Mỗi hàng CSV là một đơn vị đánh giá, không phải bài kiểm chứng riêng cho tổng hợp đoạn nhiều câu.
- Xác minh SHA-256 của Validation, provenance của hai bộ bàn giao và hash Validation trong cấu hình huấn luyện Transformer. Bộ bàn giao SVM không cung cấp hash dữ liệu huấn luyện trong metadata, nên không suy diễn có bằng chứng đó.
- Xác minh fingerprint mô hình/dữ liệu/code Bước 9, metadata nhãn, shape (8238,3), từng hàng nhãn thật/nguồn và các Macro-F1 baseline.
- Tái sử dụng `backend/outputs/alpha_tuning/val_probabilities.npz` sau khi đối chiếu hash cache với bản ghi Bước 9.1. Không chạy inference mới trong thực nghiệm này, không train/fine-tune.
- Tính hai quy tắc trên cùng thứ tự hàng và nhãn thật. Macro là trung bình không trọng số của đủ ba lớp, dùng zero_division=0. Confusion Matrix: hàng là nhãn thật, cột là nhãn dự đoán.
- Xuất dự đoán từng mẫu, mã hàng từ index/text/label/source và toàn bộ trường hợp khác nhãn. Hash các tệp được bảo vệ trước/sau thực nghiệm.

Validation SHA-256: `e4f5ed62299115d4f726331c47482d3195bd006adcf58223ddb64718c50156c7`. Cache SHA-256: `8fbf4fdd8bb9f1f8ad45da1fc20198b0123b54a5c91e9fe9ac7db8c3a01f3df3`.

**Giới hạn alignment:** cache Bước 9 không chứa hash văn bản cho từng hàng. Liên kết xác suất với văn bản gốc dựa vào thứ tự producer đã khóa, hash toàn CSV, nhãn/nguồn từng hàng và kiểm tra baseline. Hash lịch sử giúp phát hiện cache bị sửa sau Bước 9.1 nhưng không phải chữ ký độc lập chứng minh inference gốc. Mã hàng mới không chứng minh ngược lại alignment gốc.

## 4. Kết quả toàn bộ Validation

| Phương pháp | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | --- | --- | --- | --- |
| Max Confidence | 0.8162175285 | 0.8180495575 | 0.8163962165 | 0.8170001870 |
| Weighted Ensemble | 0.8248361253 | 0.8266227895 | 0.8249823201 | 0.8255456672 |

**Chênh lệch Macro-F1 (Weighted − Max): +0.0085454802, tương đương +0.854548 điểm phần trăm.**

Weighted Ensemble có Macro-F1 cao hơn trên Validation. Đây là chênh lệch tuyệt đối, không phải phần trăm cải thiện tương đối.

### Đối chiếu thêm với từng mô hình riêng

| Mô hình | Accuracy | Macro-F1 |
| --- | --- | --- |
| svm | 0.8235008497 | 0.8241494288 |
| transformer | 0.7920611799 | 0.7928055628 |

So với SVM riêng lẻ, Weighted Ensemble tăng Macro-F1 0.0013962384, tương đương 0.139624 điểm phần trăm. Mức cải thiện so với SVM nhỏ hơn mức cải thiện so với Max Confidence, nên không diễn giải rằng kết hợp là bắt buộc hoặc luôn vượt trội.

### F1 từng nhãn

| Nhãn | Số mẫu | F1 Max Confidence | F1 Weighted Ensemble |
| --- | --- | --- | --- |
| negative | 2775 | 0.8050923436 | 0.8172781493 |
| neutral | 2785 | 0.7961850936 | 0.8047752809 |
| positive | 2678 | 0.8497231239 | 0.8545835716 |

### Confusion Matrix

**Max Confidence**

| Thật / Dự đoán | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 2245 | 387 | 143 |
| neutral | 340 | 2254 | 191 |
| positive | 217 | 236 | 2225 |

**Weighted Ensemble**

| Thật / Dự đoán | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 2261 | 368 | 146 |
| neutral | 312 | 2292 | 181 |
| positive | 185 | 251 | 2242 |

### Kết quả từng nguồn

| Nguồn | N | Phương pháp | Accuracy | Macro-P | Macro-R | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| hieu_sa | 3372 | Max Confidence | 0.80367734 | 0.74586902 | 0.75264274 | 0.74859270 |
| hieu_sa | 3372 | Weighted Ensemble | 0.81494662 | 0.75796558 | 0.76791044 | 0.76217551 |
| neu_esc | 3270 | Max Confidence | 0.78868502 | 0.71375948 | 0.68945097 | 0.70066070 |
| neu_esc | 3270 | Weighted Ensemble | 0.79755352 | 0.72705031 | 0.69463023 | 0.70939957 |
| vsfc | 1596 | Max Confidence | 0.89912281 | 0.69953032 | 0.68887974 | 0.69340644 |
| vsfc | 1596 | Weighted Ensemble | 0.90162907 | 0.70639042 | 0.69059962 | 0.69693417 |

Chi tiết precision/recall/F1/support từng lớp và confusion matrix theo từng nguồn có trong `summary.json`; các bảng CSV giữ số thực đầy đủ.

## 5. Phân tích sai khác

| Thống kê | Số mẫu |
| --- | --- |
| Max Confidence chọn SVM | 2483 |
| Max Confidence chọn Transformer | 5755 |
| Hòa confidence chính xác (đã tính trong SVM) | 0 |
| Cả hai phương pháp đúng | 6598 |
| Chỉ Max Confidence đúng | 126 |
| Chỉ Weighted Ensemble đúng | 197 |
| Cả hai sai | 1317 |
| Hai phương pháp khác nhãn | 362 |
| Khác nhãn nhưng cả hai sai | 39 |

Tổng bốn nhóm đúng/sai là 8,238 mẫu. Weighted Ensemble dự đoán đúng 6795 mẫu, Max Confidence đúng 6724 mẫu. Chênh lệch số đúng bằng số mẫu chỉ Weighted đúng trừ số mẫu chỉ Max đúng. Macro-F1 còn phụ thuộc phân bố lỗi giữa ba nhãn.

### Ví dụ khác nhãn

| Index | Nguồn | Văn bản (rút gọn) | Nhãn thật | SVM conf. | TF conf. | MC chọn | MC nhãn | WE nhãn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 119 | vsfc | trong khi đề thi cho phép sinh viên dùng mạng | neutral | 0.55501 | 0.55792 | transformer | neutral | negative |
| 231 | neu_esc | ớt cay đấy cay nhảy mũ bảo hiểm hôm trường bao trời mùa hè hà nội xanh nắng cưỡi giấc mơ trung hoa tấp xe ulis chẳng hôm dại dội mũ bảo hiểm yên xe thay balo cú hôm mũ thời trang k | neutral | 0.52717 | 0.58533 | transformer | neutral | negative |
| 102 | neu_esc | thế còn đời trai củ toi thì saohic | negative | 0.50433 | 0.55298 | transformer | neutral | negative |
| 116 | hieu_sa | còn đặt thì lấy lên size | neutral | 0.54194 | 0.54748 | transformer | negative | neutral |
| 312 | neu_esc | còn không biết bước vào con đường thích phú bà bao ấm | neutral | 0.59004 | 0.64061 | transformer | negative | positive |
| 335 | neu_esc | báo cáo học thuật toàn cầu thú vị báo cáo việt nam nằm top laos campuchia châu phi bắc tt china | positive | 0.49284 | 0.52108 | transformer | neutral | negative |

Các ví dụ lấy theo index tăng dần, tối đa hai ví dụ ở mỗi nhóm khác nhãn (chỉ Max đúng, chỉ Weighted đúng, cả hai sai). Đây là ví dụ giải thích, không thay thế thống kê toàn bộ. Văn bản đầy đủ, cả ba xác suất của từng mô hình và hai phương pháp nằm trong `disagreements.csv`. Index bắt đầu từ 0, csv_line = index + 2.

## 6. Confidence và hiệu chuẩn

| Đầu ra | Confidence trung bình | Accuracy | ECE (10 bins) | Sai nhưng confidence ≥ 0,9 |
| --- | --- | --- | --- | --- |
| svm | 0.79927592 | 0.82350085 | 0.02890225 | 79 |
| transformer | 0.83884560 | 0.79206118 | 0.04678442 | 228 |
| max_confidence | 0.86922717 | 0.81621753 | 0.05330179 | 256 |
| weighted_ensemble | 0.79965569 | 0.82483613 | 0.02583217 | 74 |

Transformer có confidence trung bình 83.88% nhưng accuracy 79.21%. SVM có confidence trung bình 79.93% và accuracy 82.35%. Như vậy, trên Validation, Transformer tự tin trung bình cao hơn dù accuracy toàn tập thấp hơn SVM. Max Confidence chọn Transformer 5,755/8,238 lần (69.86%). Đây là dấu hiệu cần thận trọng khi so trực tiếp hai thang confidence.

ECE của Max Confidence là 0.05330179, cao hơn ECE của Weighted Ensemble (0.02583217) trong phép đo này. Max Confidence có 256 dự đoán sai với confidence từ 0,9 trở lên, còn Weighted Ensemble có 74. Các số đếm này dùng toàn bộ Validation, không phải tỷ lệ chính xác của một câu nhập mới.

SVM sử dụng xác suất từ bộ phân loại có hiệu chuẩn sigmoid trong artifact, còn Transformer trả softmax. Giá trị 0,9 từ hai đầu ra không mặc nhiên đại diện cho cùng một tỷ lệ đúng thực tế. Softmax hoặc hiệu chuẩn riêng từng mô hình không bảo đảm hai thang confidence tương đương trên dữ liệu đa nguồn.

Max Confidence có thể ưu tiên mô hình cho phân phối sắc hơn, kể cả khi mô hình đó sai. ECE dùng 10 khoảng đều nhau trên [0,1], cộng chênh lệch tuyệt đối giữa confidence trung bình và accuracy trong mỗi khoảng, có trọng số theo số mẫu. ECE nhỏ hơn chỉ cho thấy mức khớp confidence/accuracy tốt hơn theo cách chia khoảng này, không chứng minh ưu thế mọi mẫu hoặc mọi nguồn. Không hiệu chuẩn lại hoặc thay đổi xác suất trong Bước 14.1.

Các chẩn đoán bao gồm confidence trung bình trên dự đoán đúng/sai, số dự đoán sai confidence cao, accuracy trên phần mẫu mà Max chọn từng mô hình và reliability bins trong `summary.json`. Chúng mô tả rủi ro chọn theo confidence. Chưa có can thiệp hiệu chuẩn đối chứng nên không thể quy toàn bộ chênh lệch kết quả cho nguyên nhân hiệu chuẩn.

## 7. Ưu điểm và nhược điểm

**Max Confidence:** quy tắc đơn giản, có thể chuyển sang Transformer ở từng văn bản khi Transformer tự tin hơn. Tuy nhiên, phương pháp loại bỏ thông tin của mô hình còn lại, nhạy với sự khác biệt thang confidence và không bảo đảm mô hình tự tin hơn là mô hình đúng hơn. Để chọn được mô hình vẫn phải chạy cả hai, nên không tự động giảm chi phí suy luận so với Ensemble.

**Weighted Ensemble:** dùng đầy đủ ba xác suất của cả hai mô hình, cho phép bổ sung thông tin ngay cả khi hai mô hình khác nhãn. Tuy nhiên, cần chọn trọng số bằng dữ liệu xác thực, vẫn cần suy luận hai mô hình, trọng số cố định có thể chưa phù hợp từng nguồn và không bảo đảm mọi văn bản đều tốt hơn. Ensemble cũng có thể bị ảnh hưởng bởi xác suất chưa được hiệu chuẩn.

## 8. Kết luận và giới hạn

Weighted Ensemble có Macro-F1 cao hơn trên Validation. Max Confidence có Macro-F1 0.8170001870; Weighted Ensemble có Macro-F1 0.8255456672. Độ lớn chênh lệch tuyệt đối là 0.0085454802 (0.854548 điểm phần trăm).

Kết quả này là bằng chứng thực nghiệm cho lựa chọn trong phạm vi Validation hiện tại, không chứng minh kết hợp là bắt buộc hoặc luôn tốt hơn trên dữ liệu mới. Alpha 0,8 đã được chọn trên chính Validation từ trước, còn Max Confidence không có tham số điều chỉnh ở đây. Vì vậy đây là so sánh mô tả trên tập phát triển, không phải ước lượng tổng quát hóa độc lập và chưa kiểm định ý nghĩa thống kê.

Giữ nguyên BEST_METHOD, mặc định Weighted Ensemble, trọng số hai mô hình, alpha, FastAPI/React và kết quả Test hiện có. Dừng tại Bước 14.1. Đề xuất sau này: nếu được phê duyệt, dùng dữ liệu phát triển độc lập để nghiên cứu hiệu chuẩn và độ ổn định chênh lệch, rồi đánh giá theo kế hoạch khóa trước. Không tự triển khai bước tiếp theo.

## 9. Tái lập và tệp kết quả

Chạy tại thư mục gốc project:

```powershell
python -m backend.run_max_confidence_comparison
python -m unittest discover backend/tests
```

Frontend regression giữ nguyên: tại `frontend`, chạy `npm test`, `npm run typecheck`, `npm run test:e2e`.

Đầu ra trong `backend/outputs/max_confidence_vs_ensemble/`: `summary.json`, `metrics.csv`, `per_class_metrics.csv`, `source_metrics.csv`, `predictions.csv` (8.238 mẫu), `disagreements.csv` (toàn bộ mẫu khác nhãn), `protected_files_before.json`, `invariants.json`, `artifact_manifest.json`. Kết quả kiểm thử thực tế được lưu riêng trong `verification.json` và log tại cùng thư mục sau khi chạy.

Không chạy lại `run_alpha_tuning`, `run_error_analysis` hoặc `run_final_evaluation` để thực hiện bước này. Nếu kiểm tra cache/provenance thất bại, runner dừng rõ lỗi, không tự ghi đè cache hay suy luận lại.

## 10. Kết quả kiểm thử của lần thực hiện Bước 14.1

| Kiểm tra | Kết quả |
| --- | --- |
| Unit tests mới | 25/25 pass, đã kiểm tra thất bại trước khi triển khai |
| Toàn bộ backend regression | 421/421 pass, bao gồm 396 test cũ và 25 test mới |
| Frontend Vitest/RTL | 49/49 pass |
| E2E với backend thật | 1/1 pass, kiểm tra 5 đầu vào thật |
| TypeScript typecheck | Pass |
| Production build | Pass |
| Đối chiếu CSV với cache | Đủ 8.238 hàng, xác suất khớp chính xác |
| Đối chiếu độc lập metrics và confusion matrix | Khớp kết quả sklearn |
| Kiểm tra khác nhãn | Đủ 362 mẫu, giữ nguyên thứ tự index |
| Hash tệp bảo vệ sau regression | 153/153 tệp giữ nguyên nội dung |

Test mới bao phủ chọn toàn bộ phân phối, confidence cao hơn, hòa ưu tiên SVM, gần hòa không làm tròn, thứ tự nhãn/mẫu, input không hợp lệ, không sửa đầu vào, metrics tính tay, các nhóm đúng/sai, kết quả từng nguồn, baseline và từ chối cache/provenance sai. E2E dùng ứng dụng hiện có, không tích hợp Max Confidence vào FastAPI hoặc React.

Lần regression backend trong sandbox treo ở socket nội bộ Windows khi TestClient khởi tạo event loop. Chẩn đoán xác định đúng vị trí, sau đó chạy toàn bộ bộ test ngoài sandbox đã pass. Không sửa API hoặc dependency để xử lý việc này. Cảnh báo deprecation Starlette/httpx hiện có vẫn xuất hiện. Chi tiết, lệnh và log nằm trong `backend/outputs/max_confidence_vs_ensemble/verification.json`.
