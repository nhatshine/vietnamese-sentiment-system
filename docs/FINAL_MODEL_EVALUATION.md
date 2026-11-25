# Bước 10 — Final Model Evaluation on Test



## 1. Mục tiêu thực nghiệm



Đánh giá cuối ba phương án trên cùng Test nguyên gốc. Lựa chọn mô hình và alpha chỉ dựa trên Validation, trước Test inference; không tối ưu theo Test.



## 2. Dataset



Train: 65899 mẫu theo metadata, đã dùng huấn luyện trong các bộ bàn giao; không train lại hoặc đánh giá lại Train ở đây. Validation Bước 9: 8.238 mẫu. Test: **8,238 mẫu**.

Test SHA-256: `60f9a200f025c8001d72e83280dc471f6103c8dae0d76e4a058577ed5a9625de`. Hai Test CSV bàn giao, bản project và hash training config được đối chiếu trước inference. SVM không cung cấp training-dataset SHA trong metadata; không suy diễn bằng chứng này.

Dữ liệu text đã cleaned, dùng nguyên văn/thứ tự, không clean_text lại, không loại/shuffle mẫu. Nhãn 0=negative, 1=neutral, 2=positive.

| Label | N |
|---|---|
| negative | 2775 |
| neutral | 2785 |
| positive | 2678 |

| Source | N |
|---|---|
| neu_esc | 3270 |
| vsfc | 1596 |
| hieu_sa | 3372 |



## 3. Cấu hình đã khóa



SVM đã huấn luyện và Transformer tự xây giữ nguyên checkpoint/vocab/tokenizer/preprocessing. Alpha = **0.8**; selected_method = **ensemble**, xác định từ Validation Macro-F1 trước Test inference. Không đổi lựa chọn theo thứ hạng Test.

Transformer chạy CPU, batch64, đường inference cũ dùng eval()/torch.inference_mode(); mỗi model gọi cleaned-batch một lượt. Hash model/config/vocab/code, mapping và môi trường được ghi trong evaluation_freeze.json trước inference.

test_probabilities.npz lưu probability full precision, ordered row IDs (SHA256 của index/text/label/source), manifest và hash integrity ngoài cache. Chạy lại script chỉ tái xuất từ cache hợp lệ; cache/freeze sai thì dừng.



## 4. Phương pháp đánh giá



Dùng sklearn.metrics; macro Precision/Recall/F1 tính trực tiếp trên toàn tập, labels=[0,1,2], average='macro', zero_division=0. Weighted-F1 tính theo support. Không trung bình source F1 hoặc F1 hai model để thay metric tổng. Ensemble dùng hàm vector hóa hiện có, không làm tròn probability.



## 5. Kết quả tổng thể — RQ1



| Metric | SVM | Transformer | Ensemble |
|---|---|---|---|
| accuracy | 0.8177955815 | 0.7935178441 | 0.8213158534 |
| macro_precision | 0.8196898273 | 0.7949903076 | 0.8232012659 |
| macro_recall | 0.8179863957 | 0.7938377326 | 0.8215699493 |
| macro_f1 | 0.8186049763 | 0.7943039909 | 0.8221833733 |
| weighted_f1 | 0.8181749165 | 0.7938238256 | 0.8217058159 |

Ensemble tốt hơn cả hai model trên Test.

| So sánh Macro-F1 | Chênh lệch tuyệt đối | Điểm phần trăm |
|---|---|---|
| vs_svm | +0.0035783970 | +0.357840 |
| vs_transformer | +0.0278793825 | +2.787938 |

Các delta là quan sát trên Test, chưa kiểm định ý nghĩa thống kê. Không thay alpha/selected_method nếu kết quả không như kỳ vọng.

![Test Macro-F1](C:/sentiment_system/backend/outputs/final_evaluation/test_macro_f1_comparison.png)

![Test Accuracy](C:/sentiment_system/backend/outputs/final_evaluation/test_accuracy_comparison.png)



## 6. Theo class và confusion — RQ2



| Model | Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| svm | negative | 0.801406 | 0.821622 | 0.811388 | 2775 |
| svm | neutral | 0.784231 | 0.800000 | 0.792037 | 2785 |
| svm | positive | 0.873433 | 0.832338 | 0.852390 | 2678 |
| transformer | negative | 0.776783 | 0.781261 | 0.779015 | 2775 |
| transformer | neutral | 0.762005 | 0.780610 | 0.771195 | 2785 |
| transformer | positive | 0.846184 | 0.819642 | 0.832701 | 2678 |
| ensemble | negative | 0.804646 | 0.823784 | 0.814103 | 2775 |
| ensemble | neutral | 0.785614 | 0.800000 | 0.792742 | 2785 |
| ensemble | positive | 0.879344 | 0.840926 | 0.859706 | 2678 |

Class F1 thấp nhất trên Test: svm=neutral; transformer=neutral; ensemble=neutral.

So với SVM trên Test, Ensemble: negative tăng F1 (+0.00271466); neutral tăng F1 (+0.00070453); positive tăng F1 (+0.00731599).

Bước 9.1: neutral thấp nhất, negative→neutral nhầm nhiều nhất; Ensemble tăng positive và giảm nhẹ neutral. Bảng dưới đối chiếu trực tiếp Test, không giả định các hiện tượng lặp lại.

| Label | Test Ensemble F1 − SVM F1 |
|---|---|
| negative | +0.00271466 |
| neutral | +0.00070453 |
| positive | +0.00731599 |

Confusion matrices: hàng ground truth, cột predicted; thứ tự negative/neutral/positive. Cả ba ảnh dùng cùng thang counts và normalized [0,1].

| Model | Cặp nhầm nhiều nhất | N |
|---|---|---|
| svm | neutral → negative | 369 |
| transformer | negative → neutral | 430 |
| ensemble | neutral → negative | 372 |

![Per-class F1](C:/sentiment_system/backend/outputs/final_evaluation/test_per_class_f1.png)

![Confusion svm](C:/sentiment_system/backend/outputs/final_evaluation/test_confusion_matrix_svm.png)

![Confusion transformer](C:/sentiment_system/backend/outputs/final_evaluation/test_confusion_matrix_transformer.png)

![Confusion ensemble](C:/sentiment_system/backend/outputs/final_evaluation/test_confusion_matrix_ensemble.png)



## 7. Theo source — RQ3



| Source | N | negative/neutral/positive | SVM Macro-F1 | Transformer Macro-F1 | Ensemble Macro-F1 |
|---|---|---|---|---|---|
| hieu_sa | 3372 | 1440/452/1480 | 0.75896534 | 0.72571983 | 0.75641119 |
| neu_esc | 3270 | 593/2266/411 | 0.69351018 | 0.66545697 | 0.70144550 |
| vsfc | 1596 | 742/67/787 | 0.69575194 | 0.68954690 | 0.70017710 |

Nguồn có Macro-F1 cao nhất/thấp nhất theo từng phương án: svm=hieu_sa/neu_esc; transformer=hieu_sa/neu_esc; ensemble=hieu_sa/vsfc.

Ensemble cải thiện Macro-F1 so SVM trên 2/3 nguồn. hieu_sa: -0.00255415; neu_esc: +0.00793532; vsfc: +0.00442517. Không chọn alpha theo source.

test_source_metrics.csv chứa Accuracy, Macro-F1, F1 từng class và support. Macro-F1 cố định ba lớp; cần xem support khi diễn giải subgroup.

![Source F1](C:/sentiment_system/backend/outputs/final_evaluation/test_source_f1_comparison.png)



## 8. Validation vs Test — RQ4



| Model | Validation Macro-F1 | Test Macro-F1 | Test − Validation |
|---|---|---|---|
| svm | 0.8241494288 | 0.8186049763 | -0.0055444525 |
| transformer | 0.7928055628 | 0.7943039909 | +0.0014984280 |
| ensemble | 0.8255456672 | 0.8221833733 | -0.0033622939 |

Mức chênh lệch tuyệt đối lớn nhất = 0.00554445. Delta âm biểu thị giảm F1 ở Test; delta dương biểu thị tăng. Một chênh lệch nhỏ không đủ kết luận chắc chắn overfitting hoặc cơ chế nhân quả.

![Validation vs Test](C:/sentiment_system/backend/outputs/final_evaluation/validation_vs_test_f1.png)



## 9. Thảo luận



RQ1: Ensemble tốt hơn cả hai model trên Test. Alpha và selected_method vẫn giữ theo Validation. RQ2: class yếu nhất và hướng confusion lấy từ Test ở mục 6; hướng thay đổi F1 neutral của Ensemble so SVM là tăng, cần phân biệt với giảm nhẹ ở Validation. RQ3: Ensemble so SVM tăng trên 2/3 nguồn; không coi một tăng tổng thể là tăng đồng đều mọi source.

Không chọn BEST_METHOD mới bằng Test. Bước 11 sau này chỉ đóng gói cấu hình lựa chọn từ Validation. Không dùng kết quả Test đã thấy để thêm rule hoặc điều chỉnh mô hình.



## 10. Hạn chế và tái lập



Transformer tự xây với kiến trúc/vocabulary cố định có thể còn giới hạn năng lực; thực nghiệm này không xác định lợi ích của tăng layer/head. Weighted Ensemble có thể chỉ thay đổi nhẹ; chưa kiểm định thống kê. Khác biệt source chưa chứng minh label noise/domain shift. Confidence không bảo đảm đúng.

Document Aggregation chưa có benchmark document-level độc lập phù hợp; không gộp các dòng Test hoặc chọn Majority Voting/Mean Probability/Confidence Weighted theo nhãn câu.

Các bộ bàn giao đã có báo cáo Test riêng từ trước. Chúng không tham gia chọn alpha/method; đánh giá này là lần đánh giá cấu hình tích hợp đã khóa, không tuyên bố Test chưa từng được chủ model xem. Mọi đối chiếu báo cáo bàn giao chỉ thực hiện sau khi hoàn tất inference/metrics.

Đối chiếu sau thực nghiệm với Accuracy/Macro-F1 Test trong hai bộ bàn giao (tolerance 1e-12): svm=khớp; transformer=khớp. Kết quả chi tiết/hash nằm trong handoff_consistency của test_metrics.json; không dùng để đổi cấu hình.

Không retrain/fine-tune, không sửa tokenizer/vocab/preprocessing, không tune alpha, không chọn aggregation; chưa xây FastAPI/React. Dừng tại Bước 10.

```powershell
python -m unittest discover backend/tests
python -m backend.run_final_evaluation
```
