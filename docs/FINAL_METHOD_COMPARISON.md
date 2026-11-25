# Bước 14.2 — Kiểm chứng Max Confidence và Weighted Ensemble trên Test

## 1. Mục tiêu nghiên cứu

Trả lời câu hỏi: kết hợp xác suất SVM và Transformer có hiệu quả tốt hơn chọn dự đoán của mô hình có confidence cao nhất hay không? Phép kiểm chứng giữ nguyên hai quy tắc sau Bước 14.1 và dùng cùng Test 8.238 mẫu. Không train, fine-tune, hiệu chuẩn lại hoặc chọn trọng số bằng Test.

## 2. Hai phương pháp được so sánh

**Max Confidence:** lấy xác suất lớn nhất của từng mô hình, chọn toàn bộ phân phối của mô hình tự tin hơn. Bằng nhau chính xác thì ưu tiên SVM, không làm tròn và không dùng ngưỡng gần bằng. Tái sử dụng nguyên hàm Bước 14.1. Nếu các nhãn hòa trong một phân phối, argmax ưu tiên thứ tự negative, neutral, positive.

**Weighted Ensemble:** P = 0,8 × P_SVM + 0,2 × P_Transformer cho từng nhãn, rồi chọn argmax. Alpha 0,8 và phương án ensemble đã được chọn bằng Validation từ trước. Test chỉ kiểm chứng và mô tả kết quả, không thay đổi BEST_METHOD hoặc mặc định ứng dụng.

## 3. Thiết lập thực nghiệm

Train 65.899 mẫu theo bàn giao, Validation 8.238 mẫu, Test 8.238 mẫu. Chỉ đọc Test đã cleaned, không làm sạch lại, không tách câu hoặc loại/shuffle mẫu. Nhãn negative=0, neutral=1, positive=2. Mỗi hàng CSV là một đơn vị đánh giá, không phải đánh giá riêng phương pháp tổng hợp đoạn nhiều câu.

Test SHA-256: `60f9a200f025c8001d72e83280dc471f6103c8dae0d76e4a058577ed5a9625de`. Cache: `backend/outputs/final_evaluation/test_probabilities.npz`, SHA-256 `1d7587c7cf355d616cdfc2a76bf1a819b2a231ded13bc7b5a7994e348f50cc42`.

Provenance Test khớp hai CSV bàn giao và hash trong training config Transformer. Kiểm tra fingerprint model/data/code, selection Validation, evaluation_freeze, inference_record, hash toàn cache và manifest. So khớp từng row ID từ index/text/label/source, nhãn thật và nguồn theo đúng thứ tự. Cache thiếu hoặc sai thì dừng, không inference lại tự động. SVM không có training-dataset SHA trong metadata, không suy diễn bằng chứng này.

Cả hai phương pháp sử dụng nguyên hai ma trận xác suất (8238,3), không sửa input. Lần này không chạy suy luận mô hình mới. Precision/Recall/F1 macro là trung bình không trọng số trên đủ ba lớp với zero_division=0. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán. ECE dùng 10 khoảng độ rộng bằng nhau trên [0,1], khoảng cuối chứa 1. Ngưỡng confidence cao là ≥ 0,9.

Trước khi xuất kết quả, kiểm tra Ensemble khớp toàn bộ bốn metric tổng, confusion matrix và per-class của Bước 10 với sai số tuyệt đối 1e-12 cho số thực. Macro-F1 cũ đầy đủ là 0,8221833733152949 (công bố 0,8221833733). Các nguồn cũng phải khớp kết quả cũ. Không đánh giá lại SVM riêng lẻ, chỉ tham chiếu các metric đã công bố khi thảo luận.

## 4. Kết quả Validation và đối chiếu Test

| Phương pháp | Validation Macro-F1 | Test Macro-F1 | Test − Validation |
| --- | --- | --- | --- |
| Max Confidence | 0.8170001870 | 0.8165981610 | -0.0004020260 |
| Weighted Ensemble | 0.8255456672 | 0.8221833733 | -0.0033622939 |

Validation: Weighted Ensemble cao hơn 0.0085454802 Macro-F1. Test: Weighted − Max = +0.0055852123, tương đương +0.558521 điểm phần trăm. Weighted Ensemble có Macro-F1 cao hơn trên Test.

Validation đã dùng để chọn alpha và phương pháp nên không phải ước lượng độc lập. Test kiểm chứng hai quy tắc cố định. So sánh độ giảm/tăng giữa hai tập giúp xem tính nhất quán, không chứng minh hiệu quả trên mọi miền dữ liệu mới. Test đã được công bố trước đây và cùng hệ dữ liệu ba nguồn, nên đây là kiểm chứng bổ sung, không phải một bộ benchmark hoàn toàn mới chưa từng xem.

## 5. Kết quả Test

| Phương pháp | Accuracy | Macro-P | Macro-R | Macro-F1 | ECE |
| --- | --- | --- | --- | --- | --- |
| Max Confidence | 0.8157319738 | 0.8176443672 | 0.8159665393 | 0.8165981610 | 0.0561372001 |
| Weighted Ensemble | 0.8213158534 | 0.8232012659 | 0.8215699493 | 0.8221833733 | 0.0230784561 |

### Precision, Recall, F1 từng nhãn

| Phương pháp | Nhãn | N | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Max Confidence | negative | 2775 | 0.7990802971 | 0.8140540541 | 0.8064976794 |
| Max Confidence | neutral | 2785 | 0.7808555400 | 0.7996409336 | 0.7901365975 |
| Max Confidence | positive | 2678 | 0.8729972646 | 0.8342046303 | 0.8531602062 |
| Weighted Ensemble | negative | 2775 | 0.8046462513 | 0.8237837838 | 0.8141025641 |
| Weighted Ensemble | neutral | 2785 | 0.7856135402 | 0.8000000000 | 0.7927415051 |
| Weighted Ensemble | positive | 2678 | 0.8793440062 | 0.8409260642 | 0.8597060508 |

### Confusion matrix

**Max Confidence**

| Thật / Dự đoán | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 2259 | 381 | 135 |
| neutral | 368 | 2227 | 190 |
| positive | 200 | 244 | 2234 |

**Weighted Ensemble**

| Thật / Dự đoán | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 2286 | 365 | 124 |
| neutral | 372 | 2228 | 185 |
| positive | 183 | 243 | 2252 |

### Kết quả từng nguồn

| Nguồn | N | Phương pháp | Accuracy | Macro-P | Macro-R | Macro-F1 | ECE | Sai conf ≥ 0,9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hieu_sa | 3372 | Max Confidence | 0.81494662 | 0.75089728 | 0.75514203 | 0.75276824 | 0.05687991 | 99 |
| hieu_sa | 3372 | Weighted Ensemble | 0.82028470 | 0.75461070 | 0.75872764 | 0.75641119 | 0.03024582 | 26 |
| neu_esc | 3270 | Max Confidence | 0.78287462 | 0.70814353 | 0.68441533 | 0.69498532 | 0.06191301 | 136 |
| neu_esc | 3270 | Weighted Ensemble | 0.78807339 | 0.71136965 | 0.69372234 | 0.70144550 | 0.01448530 | 44 |
| vsfc | 1596 | Max Confidence | 0.88471178 | 0.69270501 | 0.68366399 | 0.68761795 | 0.04273412 | 49 |
| vsfc | 1596 | Weighted Ensemble | 0.89160401 | 0.70333544 | 0.69747756 | 0.70017710 | 0.04423662 | 9 |

Weighted Ensemble có Macro-F1 cao hơn trên cả ba nguồn. Tuy nhiên, ECE không cải thiện đồng đều: tại vsfc, ECE Ensemble 0,04423662 hơi cao hơn Max Confidence 0,04273412, dù F1 Ensemble cao hơn. Hiệu quả phân lớp và hiệu chuẩn là hai tiêu chí khác nhau.

Macro-F1 toàn Test được tính từ toàn bộ dự đoán, không lấy trung bình Macro-F1 của ba nguồn. Per-class metrics, confusion matrix và reliability bins theo nguồn được lưu đầy đủ trong summary.json.

## 6. Phân tích sai khác

| Thống kê | Số mẫu |
| --- | --- |
| Max Confidence chọn SVM | 2484 |
| Max Confidence chọn Transformer | 5754 |
| Hòa chính xác, đã tính trong SVM | 0 |
| Cả hai đúng | 6573 |
| Chỉ Max đúng | 147 |
| Chỉ Weighted đúng | 193 |
| Cả hai sai | 1325 |
| Khác nhãn | 377 |
| Khác nhãn nhưng cả hai sai | 37 |

Bốn nhóm đúng/sai cộng lại 8238 mẫu. Số dự đoán đúng Weighted − Max = +46. Chênh lệch accuracy và Macro-F1 không phải cùng một đại lượng.

### A. Max Confidence đúng, Weighted Ensemble sai

| Index | Nguồn | Văn bản | Nhãn thật | SVM conf | TF conf | Max chọn | Max nhãn | Weighted nhãn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | hieu_sa | bạn nào chân dài nên chọn size kỹ nhé size m mình cao 1m65 mặc bị ngắn ống | negative | 0.37699 | 0.66857 | transformer | negative | neutral |
| 214 | neu_esc | lớp toeic online thầy nha mình thấy học rất ổn áp nè bạn tham khảo nha | positive | 0.59615 | 0.75500 | transformer | positive | neutral |

Mẫu 9: Max chọn transformer và dự đoán negative, Weighted dự đoán neutral, nhãn thật negative. Confidence cao hơn dẫn đến lựa chọn đúng ở mẫu này, còn phép trộn làm đổi nhãn sang sai. Cụ thể, Transformer cho tiêu cực 66,86%, nhưng Ensemble cho trung tính 35,74%, cao hơn tiêu cực 33,85%. Trọng số SVM 80% có thể làm mất lợi thế của một dự đoán Transformer đúng ở từng mẫu.

### B. Weighted Ensemble đúng, Max Confidence sai

| Index | Nguồn | Văn bản | Nhãn thật | SVM conf | TF conf | Max chọn | Max nhãn | Weighted nhãn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | vsfc | có nội dung học rõ ràng hơn | negative | 0.63578 | 0.89448 | transformer | positive | negative |
| 37 | neu_esc | quyết định nghỉ việc xác nhận thời gian hành nghề | neutral | 0.47284 | 0.65271 | transformer | negative | neutral |

Mẫu 12: Max chọn transformer và dự đoán positive, Weighted dự đoán negative, nhãn thật negative. Phép trộn đưa nhãn về đúng dù mô hình tự tin hơn dự đoán sai ở mẫu này. Transformer cho tích cực 89,45% và SVM cho tiêu cực 63,58%. Sau kết hợp, tiêu cực đạt 52,08%, cao hơn tích cực 41,12%, nên Ensemble khớp nhãn thật tiêu cực.

### C. Cả hai phương pháp đều sai

| Index | Nguồn | Văn bản | Nhãn thật | SVM conf | TF conf | Max chọn | Max nhãn | Weighted nhãn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | neu_esc | bác hỏi những thứ rất cần tính đến pb la ts thì có thể theo quy trình phản biện kín được nơi đào tạo có thể đưa ra gợi ý hoặc gửi cả ra nước ngoài tôi cũng chỉ đang suy nghĩ còn chức danh giáo sư psg thì hội đồng có thể chỉ định người gần c | neutral | 0.55765 | 0.53290 | svm | negative | negative |
| 5 | vsfc | sử dụng kiến thức có tính thực tiễn cao hơn kiến thức có thể áp dụng vào trong thực tiễn | negative | 0.75693 | 0.97755 | transformer | positive | positive |

Mẫu 1: Max chọn svm và dự đoán negative, Weighted dự đoán negative, nhãn thật neutral. Cả chọn theo confidence lẫn kết hợp xác suất đều chưa khắc phục được lỗi ở mẫu này. Không thể xác định chắc nguyên nhân ngôn ngữ chỉ từ một ví dụ.

Ví dụ lấy tối đa hai mẫu đầu theo index của từng nhóm, không chọn theo hướng làm đẹp kết luận. Văn bản trong bảng rút gọn tối đa 240 ký tự. Predictions.csv chứa mọi mẫu và đầy đủ xác suất, disagreements.csv chứa toàn bộ mẫu khác nhãn, kể cả khi cả hai sai. Nhóm cả hai sai có thể cùng nhãn hoặc khác nhãn. Index bắt đầu từ 0. Không dùng các ví dụ để điều chỉnh thuật toán.

## 7. Đánh giá độ tin cậy và độ bất định

| Phương pháp | Confidence trung bình | Accuracy | ECE 10 bins | Sai confidence ≥ 0,9 |
| --- | --- | --- | --- | --- |
| Max Confidence | 0.8715072023 | 0.8157319738 | 0.0561372001 | 284 |
| Weighted Ensemble | 0.8009862677 | 0.8213158534 | 0.0230784561 | 79 |

Max Confidence chọn Transformer 5754/8238 lần (69.85%). Trên Validation tỷ lệ này là 69.86%. Đây là mức ưu tiên quan sát được, không tự nó chứng minh lựa chọn quá nhiều hay sai.

Confidence trung bình của SVM là 0.80085777, Transformer là 0.84076359. Accuracy riêng lẻ đã công bố từ Bước 10 tương ứng 0.81779558 và 0.79351784. Đối chiếu này mô tả thang confidence với hiệu quả thực tế, không suy luận lại SVM.

SVM dùng bộ phân loại có hiệu chuẩn sigmoid, Transformer dùng softmax. Hiệu chuẩn riêng hoặc softmax không đảm bảo confidence của hai mô hình tương đương. Max Confidence có thể ưu tiên phân phối sắc nhưng sai, trong khi Weighted giữ cả ba xác suất của cả hai mô hình. ECE đo chênh lệch confidence–accuracy theo các khoảng, không phải accuracy và không bảo đảm hiệu chuẩn cho mọi miền. Không có can thiệp hiệu chuẩn đối chứng nên không quy toàn bộ chênh lệch cho hiệu chuẩn.

### Kiểm định McNemar và paired bootstrap

McNemar chính xác hai phía: chỉ Max đúng b=147, chỉ Weighted đúng c=193, p=0.01454760229. Kiểm định nhị thức b trên b+c với H0: p=0,5, mức ý nghĩa 0,05. Bác bỏ giả thuyết xác suất dự đoán đúng bằng nhau theo kiểm định này.

Paired bootstrap: 10.000 lần, seed=42, mỗi lần lấy N chỉ số mẫu có hoàn lại và dùng cùng chỉ số cho hai phương pháp. Tính lại Macro-F1 đủ ba lớp rồi lấy Weighted − Max. Khoảng tin cậy percentile 95%: [0.0012067823; 0.0098660902], tương đương [0.120678; 0.986609] điểm phần trăm.

Khoảng bootstrap nằm hoàn toàn trên 0, hỗ trợ chênh lệch Macro-F1 dương ở mức tin cậy danh nghĩa 95% theo giả định của phép bootstrap.

McNemar kiểm tra khác biệt đúng/sai (accuracy), không phải kiểm định trực tiếp Macro-F1. Khoảng bootstrap mô tả độ bất định của Macro-F1 cho hai mô hình đã cố định. Giả định các hàng Test độc lập cùng phân phối trong hỗn hợp nguồn hiện tại. Chưa xét phụ thuộc theo người dùng/chủ đề/hội thoại, biến thiên huấn luyện hoặc thay đổi miền. Tham số thống kê được ghi trước khi đánh giá; không thay đổi để tìm kết quả có ý nghĩa. Phân tích nguồn chỉ mô tả, không có kiểm định nhiều nguồn.

## 8. Thảo luận tính cần thiết của Ensemble

Weighted Ensemble có thể tốt hơn vì tận dụng thông tin của cả hai mô hình, giảm ảnh hưởng của một dự đoán tự tin nhưng sai. Max Confidence loại bỏ thông tin mô hình còn lại. Tuy nhiên, Ensemble vẫn có các mẫu làm đúng thành sai và trọng số cố định không bảo đảm tối ưu từng nguồn. Cả hai phương pháp đều cần suy luận cả hai mô hình, nên Max Confidence không tự động tiết kiệm một lượt suy luận.

SVM riêng lẻ đã có Test Macro-F1 0.8186049763. Weighted Ensemble tăng 0.0035783970 (khoảng 0.357840 điểm phần trăm) so với SVM. Đây chỉ là tham chiếu kết quả cũ, không đánh giá lại/chỉnh sửa SVM và không thực hiện kiểm định Weighted–SVM trong bước này.

Kết hợp có thể không cần thiết nếu mô hình riêng lẻ đã đáp ứng yêu cầu, mô hình thứ hai không đóng góp đáng kể, hoặc chi phí độ trễ/bộ nhớ vượt lợi ích. Chênh lệch có ý nghĩa thống kê không đồng nghĩa lợi ích thực tế lớn. Chưa có ngưỡng cải thiện tối thiểu, ngân sách độ trễ hoặc đánh giá tác động nghiệp vụ để khẳng định mức tăng là lớn hay đủ cho mọi ứng dụng.

## 9. Kết luận nghiên cứu

**Weighted Ensemble có thực sự tốt hơn Max Confidence trên Test không?** Weighted Ensemble có Macro-F1 cao hơn trên Test. Chênh lệch quan sát được là +0.0055852123 Macro-F1 (+0.558521 điểm phần trăm). Độ bất định và kết quả kiểm định phải đọc cùng các số liệu ở mục 7.

**Nhóm có đủ cơ sở tiếp tục sử dụng Weighted Ensemble không?** Có cơ sở thực nghiệm để tiếp tục dùng Weighted Ensemble theo lựa chọn Validation đã khóa. Mức cải thiện cần đặt cạnh chi phí chạy hai mô hình và yêu cầu ứng dụng. Giữ nguyên lựa chọn theo Validation, không thay đổi phương pháp vì kết quả Test.

## 10. Hạn chế, tái lập và hướng phát triển

Kết luận giới hạn ở Test ba nguồn hiện có, mô hình/checkpoint cố định và đơn vị văn bản theo CSV. Không chứng minh ưu thế mọi nguồn mới, mọi câu hoặc cách tổng hợp đoạn văn. Giới hạn provenance SVM được giữ nguyên. Các kiểm tra hash xác nhận tính toàn vẹn của chuỗi artifact, không thay thế đánh giá chất lượng gán nhãn hoặc kiểm chứng độc lập quá trình thu thập dữ liệu.

Hướng phát triển có thể là kiểm chứng ngoài miền, hiệu chuẩn trên dữ liệu phát triển độc lập và đánh giá chi phí–chất lượng theo yêu cầu ứng dụng. Đây là đề xuất, chưa triển khai Bước 15 hoặc bất kỳ thay đổi mô hình/ứng dụng nào.

Chạy tại thư mục gốc: `python -m backend.run_final_method_comparison`. Runner từ chối ghi đè một lần chạy cũ. Tái lập vào thư mục mới: `python -m backend.run_final_method_comparison --run-name reproduce_seed42`. Chạy regression: `python -m unittest discover backend/tests`.

Tệp kết quả trong `backend/outputs/final_method_comparison/`: summary.json, metrics.csv, per_class_metrics.csv, source_metrics.csv, predictions.csv, disagreements.csv, statistical_tests.json và verification.json. Kèm statistical_plan.json, protected_files_before.json và artifact_manifest.json. Các số thực lưu đầy đủ, bảng báo cáo làm tròn. Verification ghi trạng thái kiểm thử thực tế và kiểm tra nội dung tệp cũ không đổi.

### Kiểm chứng hoàn tất

- 24/24 unit test mới pass, bao gồm quy tắc cố định, tie-break, baseline, xác suất, cache/row IDs và thống kê paired.
- Toàn bộ backend regression: 445/445 pass (421 test hiện có và 24 test mới), không sửa test cũ.
- Kiểm tra độc lập CSV: đủ 8.238 hàng, 377 mẫu khác nhãn, xác suất khớp chính xác cache Test, metric tổng/theo nguồn, ECE và số sai confidence cao khớp.
- McNemar được đối chiếu thêm bằng tổng xác suất nhị thức trực tiếp. Bootstrap có unit test đối chiếu từng lần lấy mẫu với sklearn, dùng cùng seed/chỉ số.
- 223 tệp được bảo vệ giữ nguyên nội dung sau thực nghiệm và regression, gồm artifact, dữ liệu, code cũ, cấu hình và kết quả các bước trước.

Chi tiết và log: `backend/outputs/final_method_comparison/verification.json` và `backend_regression.log`. Regression chạy ngoài Windows sandbox để TestClient khởi tạo socket nội bộ như lần kiểm chứng Bước 14.1. Không sửa FastAPI, React hoặc dependency. Dừng sau Bước 14.2.
