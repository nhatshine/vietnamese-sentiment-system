"""Step 14.1 only: python -m backend.run_max_confidence_comparison.

Uses verified Validation cache, never imports predictors, fits or opens Test
for evaluation. Existing files are read-only, outputs have their own directory.
"""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from .run_alpha_tuning import BACKEND, VALIDATION_PATH, collect_fingerprints, load_validation
from .src.alpha_tuning import compute_ensemble_probabilities, evaluate_probabilities
from .src.config import LABELS, LABEL_TO_ID
from .src.final_evaluation import ordered_row_ids
from .src.max_confidence import FIXED_ALPHA, compare_methods, select_max_confidence


ROOT = BACKEND.parent
ALPHA_DIR = BACKEND / "outputs" / "alpha_tuning"
OUTPUT_DIR = BACKEND / "outputs" / "max_confidence_vs_ensemble"
REPORT_PATH = ROOT / "docs" / "MAX_CONFIDENCE_VS_ENSEMBLE.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cache(cache_path, data, selection, current_fingerprints, expected_sha256):
    """Fail closed on integrity, metadata, row alignment or baseline mismatch."""
    if (selection.get("selection_dataset") != "validation" or
            selection.get("test_dataset_used") is not False or selection.get("selected_alpha") != FIXED_ALPHA):
        raise ValueError("Only Validation with the existing frozen alpha=0.8 is allowed.")
    if selection.get("probability_columns") != LABELS or selection.get("label_mapping") != LABEL_TO_ID:
        raise ValueError("Selection label order/mapping is inconsistent.")
    if current_fingerprints != selection.get("fingerprints"):
        raise ValueError("Frozen Step-9 model/data/code fingerprints changed.")
    if sha(cache_path) != expected_sha256:
        raise ValueError("Cache bytes no longer match the historical Step-9.1 integrity record.")
    with np.load(cache_path, allow_pickle=False) as cached:
        if json.loads(cached["fingerprint_json"].item()) != current_fingerprints:
            raise ValueError("Cache fingerprints do not match the producer.")
        if (not np.array_equal(cached["y_true"], data["labels"]) or
                not np.array_equal(cached["sources"], np.asarray(data["sources"]))):
            raise ValueError("Ordered cache truth/source rows do not match Validation.")
        svm, transformer = cached["svm_probs"].copy(), cached["transformer_probs"].copy()
    if svm.shape != (len(data["labels"]), 3) or transformer.shape != svm.shape:
        raise ValueError("Cache probability shape does not match Validation.")
    ensemble = compute_ensemble_probabilities(svm, transformer, FIXED_ALPHA)
    for name, values in (("svm", svm), ("transformer", transformer), ("ensemble", ensemble)):
        actual = evaluate_probabilities(data["labels"], values)["macro_f1"]
        if not np.isclose(actual, selection[f"{name}_validation_macro_f1"], rtol=0, atol=1e-12):
            raise ValueError(f"{name} Validation Macro-F1 no longer matches Step 9.")
    return svm, transformer


def load_inputs():
    selection_path, cache_path = ALPHA_DIR / "alpha_selection.json", ALPHA_DIR / "val_probabilities.npz"
    anchor_path = BACKEND / "outputs" / "error_analysis" / "summary.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    provenance = json.loads((BACKEND / "data" / "validation_provenance.json").read_text(encoding="utf-8"))
    config = json.loads((BACKEND / "models" / "transformer" / "config.json").read_text(encoding="utf-8"))
    expected_sha = config["dataset_sha256"]["val"]
    fields = ("sha256", "svm_handoff_validation_sha256", "transformer_archive_validation_sha256",
              "transformer_training_config_validation_sha256")
    if (provenance.get("dataset") != "validation" or provenance.get("preprocessed") is not True or
            any(provenance.get(field) != expected_sha for field in fields) or
            selection.get("dataset_sha256") != expected_sha):
        raise ValueError("Validation provenance/config/selection are inconsistent.")
    if selection.get("svm_classes") != [0, 1, 2] or selection.get("transformer_logits_labels") != LABELS:
        raise ValueError("Frozen model class metadata is inconsistent.")
    data = load_validation(VALIDATION_PATH, expected_sha)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))["provenance"]
    if (anchor.get("dataset") != "validation" or anchor.get("dataset_sha256") != expected_sha or
            anchor.get("selection_sha256") != sha(selection_path)):
        raise ValueError("Historical cache integrity anchor is inconsistent.")
    fingerprints = collect_fingerprints()
    svm, transformer = validate_cache(cache_path, data, selection, fingerprints, anchor["cache_sha256"])
    return data, svm, transformer, {
        "dataset": "validation", "dataset_sha256": expected_sha,
        "dataset_provenance": provenance, "source_counts": data["source_counts"],
        "cache_path": str(cache_path), "cache_sha256": sha(cache_path), "cache_status": "reused",
        "integrity_anchor_path": str(anchor_path), "integrity_anchor_sha256": sha(anchor_path),
        "selection_sha256": sha(selection_path), "frozen_step9_fingerprints": fingerprints,
        "row_alignment_checks": "Frozen CSV hash, frozen producer code/order, exact ordered y_true/source arrays, shapes, class mapping, baseline Macro-F1, historical cache SHA-256.",
        "row_alignment_limitations": "Original Step-9 cache has no per-row text hashes. Its original probability/text correspondence relies on frozen producer ordering. The historical SHA protects against later cache changes but is not an independent signed proof of original inference correctness. New row IDs do not retroactively prove original alignment.",
        "test_used_for_evaluation": False, "alpha_retuned": False,
        "inference_calls_this_experiment": 0, "retrained": False,
    }


def protected_manifest():
    """Byte guards only; Test files are hashed, never evaluated by this runner."""
    files = set()
    for directory in (BACKEND / "models", BACKEND / "data", BACKEND / "config", BACKEND / "src",
                      ROOT / "frontend" / "src", ROOT / "frontend" / "e2e"):
        files.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for directory in (BACKEND / "outputs").iterdir():
        if directory.is_dir() and directory != OUTPUT_DIR:
            files.update(p for p in directory.rglob("*") if p.is_file())
    for name in ("package.json", "package-lock.json", "playwright.config.ts", "vite.config.ts"):
        file = ROOT / "frontend" / name
        if file.is_file():
            files.add(file)
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def write_csv(path, rows, fieldnames=None):
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(data, svm, transformer):
    selected = select_max_confidence(svm, transformer)
    ensemble = compute_ensemble_probabilities(svm, transformer, FIXED_ALPHA)
    matrices = {"svm": svm, "transformer": transformer,
                "max_confidence": selected["probabilities"], "weighted_ensemble": ensemble}
    ids = ordered_row_ids(data)
    rows = []
    for i, text in enumerate(data["texts"]):
        truth = int(data["labels"][i])
        row = {"sample_index": i, "csv_line": i + 2, "row_id": ids[i], "source": data["sources"][i],
               "text": text, "true_label": LABELS[truth],
               "selected_model": str(selected["selected_models"][i]), "exact_confidence_tie": bool(selected["ties"][i])}
        for name, matrix in matrices.items():
            label_id = int(matrix[i].argmax())
            row.update({f"{name}_label": LABELS[label_id], f"{name}_confidence": float(matrix[i, label_id]),
                        f"{name}_correct": label_id == truth})
            row.update({f"{name}_p_{label}": float(matrix[i, j]) for j, label in enumerate(LABELS)})
        row["different_labels"] = row["max_confidence_label"] != row["weighted_ensemble_label"]
        rows.append(row)
    return rows


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in rows])


def write_report(result, rows):
    methods = ("max_confidence", "weighted_ensemble")
    names = {"max_confidence": "Max Confidence", "weighted_ensemble": "Weighted Ensemble"}
    models, groups = result["models"], result["correctness_groups"]
    delta = result["macro_f1_delta_weighted_minus_max"]
    winner = result["winner_on_validation"]
    conclusion = (f"{names[winner]} có Macro-F1 cao hơn trên Validation." if winner != "tie" else
                  "Hai phương pháp có Macro-F1 bằng nhau trên Validation.")
    metrics = markdown_table(["Phương pháp", "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"],
                            [[names[name]] + [f"{models[name][key]:.10f}" for key in
                             ("accuracy", "macro_precision", "macro_recall", "macro_f1")] for name in methods])
    per_class = markdown_table(["Nhãn", "Số mẫu", "F1 Max Confidence", "F1 Weighted Ensemble"],
                              [[LABELS[i], models[methods[0]]["per_class"][i]["support"]] +
                               [f"{models[name]['per_class'][i]['f1']:.10f}" for name in methods] for i in range(3)])
    source_table = markdown_table(["Nguồn", "N", "Phương pháp", "Accuracy", "Macro-P", "Macro-R", "Macro-F1"],
                                 [[r["source"], r["sample_count"], names[name]] +
                                  [f"{r['models'][name][key]:.8f}" for key in
                                   ("accuracy", "macro_precision", "macro_recall", "macro_f1")]
                                  for r in result["by_source"] for name in methods])
    calibration = markdown_table(["Đầu ra", "Confidence trung bình", "Accuracy", "ECE (10 bins)", "Sai nhưng confidence ≥ 0,9"],
                                [[name, f"{v['mean_confidence']:.8f}", f"{v['accuracy']:.8f}", f"{v['ece']:.8f}",
                                  v["wrong_high_confidence_counts"]["0.9"]] for name, v in result["confidence_diagnostics"].items()])
    examples = []
    for mc_ok, we_ok in ((True, False), (False, True), (False, False)):
        matching = [r for r in rows if r["different_labels"] and r["max_confidence_correct"] == mc_ok and r["weighted_ensemble_correct"] == we_ok]
        examples.extend(matching[:2])
    examples_table = markdown_table(["Index", "Nguồn", "Văn bản (rút gọn)", "Nhãn thật", "SVM conf.", "TF conf.", "MC chọn", "MC nhãn", "WE nhãn"],
                                   [[r["sample_index"], r["source"], r["text"][:180], r["true_label"],
                                     f"{r['svm_confidence']:.5f}", f"{r['transformer_confidence']:.5f}",
                                     r["selected_model"], r["max_confidence_label"], r["weighted_ensemble_label"]] for r in examples])
    sections = [
        "# Bước 14.1 — Max Confidence so với Weighted Ensemble",
        "## 1. Mục tiêu nghiên cứu",
        "Kiểm tra liệu việc kết hợp xác suất của hai mô hình có mang lại kết quả tốt hơn cách chọn mô hình tự tin hơn cho mỗi văn bản. Chỉ đánh giá trên cùng 8.238 mẫu Validation, không thay đổi cấu hình triển khai.",
        "## 2. Nguyên lý hai phương pháp",
        "**Max Confidence:** lấy giá trị lớn nhất trong ba xác suất của từng mô hình, so sánh hai giá trị và chọn **toàn bộ phân phối xác suất** của mô hình có confidence cao hơn. Không trộn từng cột và không chỉ so nhãn thắng. Nếu hai confidence bằng nhau chính xác, ưu tiên SVM. Không làm tròn và không dùng ngưỡng gần bằng. Nếu các nhãn trong một phân phối hòa xác suất, chọn nhãn đầu tiên theo thứ tự negative, neutral, positive.",
        "**Weighted Ensemble:** giữ nguyên P = 0,8 × P_SVM + 0,2 × P_Transformer cho từng nhãn, sau đó chọn argmax. Alpha cố định 0,8 từ Bước 9, không sweep lại và không dùng Test để chọn. Hai phương pháp đều dùng đủ cùng hai ma trận đầu vào.",
        "## 3. Kiểm tra project và quy trình thực nghiệm",
        "- SVM cung cấp `predict_proba`, ánh xạ cột bằng `classes_` trong `backend/src/real_svm.py`. Transformer dùng softmax logits và ánh xạ `label_map.json` trong `backend/src/real_transformer.py`. Cả hai adapter đưa về negative=0, neutral=1, positive=2.\n"
        "- Validation đã làm sạch, không làm sạch lại, không tách câu lại. Mỗi hàng CSV là một đơn vị đánh giá, không phải bài kiểm chứng riêng cho tổng hợp đoạn nhiều câu.\n"
        "- Xác minh SHA-256 của Validation, provenance của hai bộ bàn giao và hash Validation trong cấu hình huấn luyện Transformer. Bộ bàn giao SVM không cung cấp hash dữ liệu huấn luyện trong metadata, nên không suy diễn có bằng chứng đó.\n"
        "- Xác minh fingerprint mô hình/dữ liệu/code Bước 9, metadata nhãn, shape (8238,3), từng hàng nhãn thật/nguồn và các Macro-F1 baseline.\n"
        "- Tái sử dụng `backend/outputs/alpha_tuning/val_probabilities.npz` sau khi đối chiếu hash cache với bản ghi Bước 9.1. Không chạy inference mới trong thực nghiệm này, không train/fine-tune.\n"
        "- Tính hai quy tắc trên cùng thứ tự hàng và nhãn thật. Macro là trung bình không trọng số của đủ ba lớp, dùng zero_division=0. Confusion Matrix: hàng là nhãn thật, cột là nhãn dự đoán.\n"
        "- Xuất dự đoán từng mẫu, mã hàng từ index/text/label/source và toàn bộ trường hợp khác nhãn. Hash các tệp được bảo vệ trước/sau thực nghiệm.",
        f"Validation SHA-256: `{result['provenance']['dataset_sha256']}`. Cache SHA-256: `{result['provenance']['cache_sha256']}`.",
        "**Giới hạn alignment:** cache Bước 9 không chứa hash văn bản cho từng hàng. Liên kết xác suất với văn bản gốc dựa vào thứ tự producer đã khóa, hash toàn CSV, nhãn/nguồn từng hàng và kiểm tra baseline. Hash lịch sử giúp phát hiện cache bị sửa sau Bước 9.1 nhưng không phải chữ ký độc lập chứng minh inference gốc. Mã hàng mới không chứng minh ngược lại alignment gốc.",
        "## 4. Kết quả toàn bộ Validation", metrics,
        f"**Chênh lệch Macro-F1 (Weighted − Max): {delta:+.10f}, tương đương {100 * delta:+.6f} điểm phần trăm.**\n\n{conclusion} Đây là chênh lệch tuyệt đối, không phải phần trăm cải thiện tương đối.",
        "### Đối chiếu thêm với từng mô hình riêng",
        markdown_table(["Mô hình", "Accuracy", "Macro-F1"], [[name, f"{m['accuracy']:.10f}", f"{m['macro_f1']:.10f}"]
                        for name, m in result["baselines"].items()]),
        f"So với SVM riêng lẻ, Weighted Ensemble tăng Macro-F1 {models['weighted_ensemble']['macro_f1'] - result['baselines']['svm']['macro_f1']:.10f}, tương đương {100 * (models['weighted_ensemble']['macro_f1'] - result['baselines']['svm']['macro_f1']):.6f} điểm phần trăm. Mức cải thiện so với SVM nhỏ hơn mức cải thiện so với Max Confidence, nên không diễn giải rằng kết hợp là bắt buộc hoặc luôn vượt trội.",
        "### F1 từng nhãn", per_class,
        "### Confusion Matrix",
    ]
    for name in methods:
        sections.extend([f"**{names[name]}**", markdown_table(["Thật / Dự đoán", *LABELS],
                         [[LABELS[i], *models[name]["confusion_matrix"][i]] for i in range(3)])])
    sections += ["### Kết quả từng nguồn", source_table,
        "Chi tiết precision/recall/F1/support từng lớp và confusion matrix theo từng nguồn có trong `summary.json`; các bảng CSV giữ số thực đầy đủ.",
        "## 5. Phân tích sai khác",
        markdown_table(["Thống kê", "Số mẫu"], [["Max Confidence chọn SVM", result["selection_counts"]["svm"]],
             ["Max Confidence chọn Transformer", result["selection_counts"]["transformer"]],
             ["Hòa confidence chính xác (đã tính trong SVM)", result["selection_counts"]["exact_ties"]],
             ["Cả hai phương pháp đúng", groups["both_correct"]], ["Chỉ Max Confidence đúng", groups["max_confidence_only"]],
             ["Chỉ Weighted Ensemble đúng", groups["weighted_ensemble_only"]], ["Cả hai sai", groups["both_wrong"]],
             ["Hai phương pháp khác nhãn", result["disagreements"]["count"]],
             ["Khác nhãn nhưng cả hai sai", result["disagreements"]["both_wrong_count"]]]),
        f"Tổng bốn nhóm đúng/sai là {sum(groups.values()):,} mẫu. Weighted Ensemble dự đoán đúng {models['weighted_ensemble']['correct_count']} mẫu, Max Confidence đúng {models['max_confidence']['correct_count']} mẫu. Chênh lệch số đúng bằng số mẫu chỉ Weighted đúng trừ số mẫu chỉ Max đúng. Macro-F1 còn phụ thuộc phân bố lỗi giữa ba nhãn.",
        "### Ví dụ khác nhãn", examples_table,
        "Các ví dụ lấy theo index tăng dần, tối đa hai ví dụ ở mỗi nhóm khác nhãn (chỉ Max đúng, chỉ Weighted đúng, cả hai sai). Đây là ví dụ giải thích, không thay thế thống kê toàn bộ. Văn bản đầy đủ, cả ba xác suất của từng mô hình và hai phương pháp nằm trong `disagreements.csv`. Index bắt đầu từ 0, csv_line = index + 2.",
        "## 6. Confidence và hiệu chuẩn", calibration,
        f"Transformer có confidence trung bình {100 * result['confidence_diagnostics']['transformer']['mean_confidence']:.2f}% nhưng accuracy {100 * result['confidence_diagnostics']['transformer']['accuracy']:.2f}%. SVM có confidence trung bình {100 * result['confidence_diagnostics']['svm']['mean_confidence']:.2f}% và accuracy {100 * result['confidence_diagnostics']['svm']['accuracy']:.2f}%. Như vậy, trên Validation, Transformer tự tin trung bình cao hơn dù accuracy toàn tập thấp hơn SVM. Max Confidence chọn Transformer {result['selection_counts']['transformer']:,}/{result['sample_count']:,} lần ({100 * result['selection_counts']['transformer'] / result['sample_count']:.2f}%). Đây là dấu hiệu cần thận trọng khi so trực tiếp hai thang confidence.",
        f"ECE của Max Confidence là {result['confidence_diagnostics']['max_confidence']['ece']:.8f}, cao hơn ECE của Weighted Ensemble ({result['confidence_diagnostics']['weighted_ensemble']['ece']:.8f}) trong phép đo này. Max Confidence có {result['confidence_diagnostics']['max_confidence']['wrong_high_confidence_counts']['0.9']} dự đoán sai với confidence từ 0,9 trở lên, còn Weighted Ensemble có {result['confidence_diagnostics']['weighted_ensemble']['wrong_high_confidence_counts']['0.9']}. Các số đếm này dùng toàn bộ Validation, không phải tỷ lệ chính xác của một câu nhập mới.",
        "SVM sử dụng xác suất từ bộ phân loại có hiệu chuẩn sigmoid trong artifact, còn Transformer trả softmax. Giá trị 0,9 từ hai đầu ra không mặc nhiên đại diện cho cùng một tỷ lệ đúng thực tế. Softmax hoặc hiệu chuẩn riêng từng mô hình không bảo đảm hai thang confidence tương đương trên dữ liệu đa nguồn.",
        "Max Confidence có thể ưu tiên mô hình cho phân phối sắc hơn, kể cả khi mô hình đó sai. ECE dùng 10 khoảng đều nhau trên [0,1], cộng chênh lệch tuyệt đối giữa confidence trung bình và accuracy trong mỗi khoảng, có trọng số theo số mẫu. ECE nhỏ hơn chỉ cho thấy mức khớp confidence/accuracy tốt hơn theo cách chia khoảng này, không chứng minh ưu thế mọi mẫu hoặc mọi nguồn. Không hiệu chuẩn lại hoặc thay đổi xác suất trong Bước 14.1.",
        "Các chẩn đoán bao gồm confidence trung bình trên dự đoán đúng/sai, số dự đoán sai confidence cao, accuracy trên phần mẫu mà Max chọn từng mô hình và reliability bins trong `summary.json`. Chúng mô tả rủi ro chọn theo confidence. Chưa có can thiệp hiệu chuẩn đối chứng nên không thể quy toàn bộ chênh lệch kết quả cho nguyên nhân hiệu chuẩn.",
        "## 7. Ưu điểm và nhược điểm",
        "**Max Confidence:** quy tắc đơn giản, có thể chuyển sang Transformer ở từng văn bản khi Transformer tự tin hơn. Tuy nhiên, phương pháp loại bỏ thông tin của mô hình còn lại, nhạy với sự khác biệt thang confidence và không bảo đảm mô hình tự tin hơn là mô hình đúng hơn. Để chọn được mô hình vẫn phải chạy cả hai, nên không tự động giảm chi phí suy luận so với Ensemble.",
        "**Weighted Ensemble:** dùng đầy đủ ba xác suất của cả hai mô hình, cho phép bổ sung thông tin ngay cả khi hai mô hình khác nhãn. Tuy nhiên, cần chọn trọng số bằng dữ liệu xác thực, vẫn cần suy luận hai mô hình, trọng số cố định có thể chưa phù hợp từng nguồn và không bảo đảm mọi văn bản đều tốt hơn. Ensemble cũng có thể bị ảnh hưởng bởi xác suất chưa được hiệu chuẩn.",
        "## 8. Kết luận và giới hạn",
        f"{conclusion} Max Confidence có Macro-F1 {models['max_confidence']['macro_f1']:.10f}; Weighted Ensemble có Macro-F1 {models['weighted_ensemble']['macro_f1']:.10f}. Độ lớn chênh lệch tuyệt đối là {abs(delta):.10f} ({100 * abs(delta):.6f} điểm phần trăm).",
        "Kết quả này là bằng chứng thực nghiệm cho lựa chọn trong phạm vi Validation hiện tại, không chứng minh kết hợp là bắt buộc hoặc luôn tốt hơn trên dữ liệu mới. Alpha 0,8 đã được chọn trên chính Validation từ trước, còn Max Confidence không có tham số điều chỉnh ở đây. Vì vậy đây là so sánh mô tả trên tập phát triển, không phải ước lượng tổng quát hóa độc lập và chưa kiểm định ý nghĩa thống kê.",
        "Giữ nguyên BEST_METHOD, mặc định Weighted Ensemble, trọng số hai mô hình, alpha, FastAPI/React và kết quả Test hiện có. Dừng tại Bước 14.1. Đề xuất sau này: nếu được phê duyệt, dùng dữ liệu phát triển độc lập để nghiên cứu hiệu chuẩn và độ ổn định chênh lệch, rồi đánh giá theo kế hoạch khóa trước. Không tự triển khai bước tiếp theo.",
        "## 9. Tái lập và tệp kết quả",
        "Chạy tại thư mục gốc project:\n\n```powershell\npython -m backend.run_max_confidence_comparison\npython -m unittest discover backend/tests\n```\n\nFrontend regression giữ nguyên: tại `frontend`, chạy `npm test`, `npm run typecheck`, `npm run test:e2e`.",
        "Đầu ra trong `backend/outputs/max_confidence_vs_ensemble/`: `summary.json`, `metrics.csv`, `per_class_metrics.csv`, `source_metrics.csv`, `predictions.csv` (8.238 mẫu), `disagreements.csv` (toàn bộ mẫu khác nhãn), `protected_files_before.json`, `invariants.json`, `artifact_manifest.json`. Kết quả kiểm thử thực tế được lưu riêng trong `verification.json` và log tại cùng thư mục sau khi chạy.",
        "Không chạy lại `run_alpha_tuning`, `run_error_analysis` hoặc `run_final_evaluation` để thực hiện bước này. Nếu kiểm tra cache/provenance thất bại, runner dừng rõ lỗi, không tự ghi đè cache hay suy luận lại.",
    ]
    REPORT_PATH.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main():
    before = protected_manifest()
    data, svm, transformer, provenance = load_inputs()
    result = compare_methods(data["labels"], svm, transformer, data["sources"])
    result["provenance"] = {**provenance, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "python_version": sys.version,
                            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
                            "experiment_code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
                                                       (Path(__file__), BACKEND / "src" / "max_confidence.py")}}
    rows = prediction_rows(data, svm, transformer)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "predictions.csv", rows)
    write_csv(OUTPUT_DIR / "disagreements.csv", [r for r in rows if r["different_labels"]], list(rows[0]))
    metric_keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    write_csv(OUTPUT_DIR / "metrics.csv", [{"method": name, **{key: m[key] for key in metric_keys}}
                                         for name, m in result["models"].items()])
    write_csv(OUTPUT_DIR / "per_class_metrics.csv", [{"method": name, **row}
                                                   for name, m in result["models"].items() for row in m["per_class"]])
    write_csv(OUTPUT_DIR / "source_metrics.csv", [{"source": r["source"], "n": r["sample_count"], "method": name,
                                                 **{key: m[key] for key in metric_keys}}
                                                for r in result["by_source"] for name, m in r["models"].items()])
    write_report(result, rows)
    after = protected_manifest()
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    invariants = {"protected_file_count": len(before), "unchanged": not changed, "changed_files": changed,
                  "scope": "Models/data/config/old outputs/backend src/frontend src and configuration. Test data/results hashed only, never evaluated by this experiment."}
    (OUTPUT_DIR / "protected_files_before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "invariants.json").write_text(json.dumps(invariants, indent=2) + "\n", encoding="utf-8")
    if changed:
        raise ValueError(f"Protected files changed: {changed}; do not accept these results.")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts = [REPORT_PATH] + [OUTPUT_DIR / name for name in ("summary.json", "metrics.csv", "per_class_metrics.csv",
                 "source_metrics.csv", "predictions.csv", "disagreements.csv", "protected_files_before.json", "invariants.json")]
    (OUTPUT_DIR / "artifact_manifest.json").write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": {name: {k: v[k] for k in metric_keys} for name, v in result["models"].items()},
                      "winner": result["winner_on_validation"], "macro_f1_delta_weighted_minus_max": result["macro_f1_delta_weighted_minus_max"],
                      "selection_counts": result["selection_counts"], "correctness_groups": result["correctness_groups"],
                      "different_labels": result["disagreements"]["count"], "cache_status": provenance["cache_status"],
                      "protected_files_unchanged": invariants["unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
