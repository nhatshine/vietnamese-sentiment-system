"""Validation diagnostics, fixed Step-9 alpha: python -m backend.run_error_analysis."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .run_alpha_tuning import BACKEND, VALIDATION_PATH, collect_fingerprints, load_validation
from .src.alpha_tuning import compute_ensemble_probabilities, evaluate_probabilities
from .src.config import LABELS, LABEL_TO_ID
from .src.error_analysis import MODEL_NAMES, analyze_validation_errors


ALPHA_OUTPUT = BACKEND / "outputs" / "alpha_tuning"
SELECTION_PATH = ALPHA_OUTPUT / "alpha_selection.json"
CACHE_PATH = ALPHA_OUTPUT / "val_probabilities.npz"
OUTPUT_DIR = BACKEND / "outputs" / "error_analysis"
REPORT_PATH = BACKEND.parent / "docs" / "ERROR_ANALYSIS_REPORT.md"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recompute_probabilities(texts):
    # Chỉ dùng nếu cache lỗi, trong khi frozen inputs/model vẫn đúng Bước 9.
    from .src.real_svm import predict_svm_cleaned_batch
    from .src.real_transformer import predict_transformer_cleaned_batch
    print("Cache invalid/unavailable: recomputing cleaned Validation once, no alpha sweep.", flush=True)
    svm = predict_svm_cleaned_batch(texts)
    transformer = predict_transformer_cleaned_batch(texts, batch_size=64, device="cpu")
    return (np.array([[row[label] for label in LABELS] for row in svm]),
            np.array([[row[label] for label in LABELS] for row in transformer]))


def load_analysis_inputs() -> dict:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("selection_dataset") != "validation" or selection.get("test_dataset_used") is not False:
        raise ValueError("Selection phải là Validation-only của Bước 9.")
    if selection.get("probability_columns") != LABELS or selection.get("label_mapping") != LABEL_TO_ID:
        raise ValueError("Class mapping không khớp Bước 9.")
    data = load_validation(VALIDATION_PATH, selection["dataset_sha256"])
    if collect_fingerprints() != selection["fingerprints"]:
        raise ValueError("Frozen model/data/code fingerprint không khớp Bước 9; dừng, không đổi model.")
    with (ALPHA_OUTPUT / "alpha_scores.csv").open(encoding="utf-8", newline="") as stream:
        scores = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    selected_rows = [row for row in scores if row["alpha"] == selection["selected_alpha"]]
    if len(selected_rows) != 1 or not np.isclose(selected_rows[0]["macro_f1"], selection["selected_validation_macro_f1"], atol=1e-12, rtol=0):
        raise ValueError("alpha_selection không khớp bảng điểm Bước 9.")
    if selected_rows[0]["macro_f1"] != max(row["macro_f1"] for row in scores):
        raise ValueError("Selected alpha không nhất quán với kết quả đã khóa Bước 9.")
    status, reason = "reused", None
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as cached:
            if json.loads(cached["fingerprint_json"].item()) != selection["fingerprints"]:
                raise ValueError("Cache fingerprint không khớp.")
            if not np.array_equal(cached["y_true"], data["labels"]) or not np.array_equal(cached["sources"], np.array(data["sources"])):
                raise ValueError("Cache sample alignment/ground truth/source không khớp.")
            svm, transformer = cached["svm_probs"].copy(), cached["transformer_probs"].copy()
            compute_ensemble_probabilities(svm, transformer, selection["selected_alpha"])
            if svm.shape != (len(data["labels"]), len(LABELS)):
                raise ValueError("Cache có số mẫu sai.")
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        status, reason = "recomputed", str(error)
        svm, transformer = _recompute_probabilities(data["texts"])
    ensemble = compute_ensemble_probabilities(svm, transformer, selection["selected_alpha"])
    for name, values in (("svm", svm), ("transformer", transformer), ("ensemble", ensemble)):
        actual = evaluate_probabilities(data["labels"], values)["macro_f1"]
        if not np.isclose(actual, selection[f"{name}_validation_macro_f1"], atol=1e-12, rtol=0):
            raise ValueError(f"{name} Macro-F1 không khớp Bước 9; không xuất báo cáo.")
    return {"data": data, "svm_probs": svm, "transformer_probs": transformer,
            "alpha": selection["selected_alpha"], "selection": selection,
            "cache_status": status, "cache_reason": reason}


def _csv(path, rows, default_fields=None):
    fields = list(rows[0]) if rows else default_fields or []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _plot_all(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#2166ac", "#d97706", "#16836b")

    def save(figure, filename):
        figure.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(figure)

    groups = summary["complementarity"]["groups"]
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    bars = ax.bar(["A\nBoth correct", "B\nSVM only correct", "C\nTransformer only correct", "D\nBoth wrong"],
                  [groups[key]["count"] for key in ("A", "B", "C", "D")], color=[colors[2], colors[0], colors[1], "#a33f45"])
    for bar, key in zip(bars, ("A", "B", "C", "D")):
        ax.annotate(f"{groups[key]['count']:,}\n{100 * groups[key]['rate']:.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.set(title=f"Validation complementary errors | N={summary['sample_count']:,}", ylabel="Samples")
    ax.set_ylim(0, max(bar.get_height() for bar in bars) * 1.18)
    save(fig, "model_correctness_groups.png")

    for model in MODEL_NAMES:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), layout="constrained")
        for ax, key, title in zip(axes, ("confusion_matrix", "confusion_matrix_row_normalized"), ("Counts", "Row-normalized")):
            values = np.array(summary["models"][model][key])
            image = ax.imshow(values, cmap="Blues", vmin=0, vmax=None if key == "confusion_matrix" else 1)
            for i in range(3):
                for j in range(3):
                    text = str(int(values[i, j])) if key == "confusion_matrix" else f"{values[i, j]:.3f}"
                    ax.text(j, i, text, ha="center", va="center", color="white" if values[i, j] > values.max() / 2 else "black")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                   xlabel="Predicted label", ylabel="Ground truth", title=title)
            fig.colorbar(image, ax=ax, shrink=.8)
        fig.suptitle(f"Validation | {model.upper()}" + (f" | alpha={summary['alpha']}" if model == "ensemble" else ""))
        save(fig, f"confusion_matrix_{model}.png")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    positions = np.arange(3)
    for i, model in enumerate(MODEL_NAMES):
        values = [row["f1"] for row in summary["models"][model]["per_class"]]
        bars = ax.bar(positions + (i - 1) * .24, values, .24, label=model, color=colors[i])
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set(xticks=positions, xticklabels=LABELS, ylim=(0, 1.08), ylabel="F1", title="Validation per-class F1")
    ax.legend()
    save(fig, "per_class_f1_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for i, model in enumerate(MODEL_NAMES):
        bars = ax.bar(positions + (i - 1) * .24, [row[f"{model}_macro_f1"] for row in summary["sources"]], .24, label=model, color=colors[i])
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set(xticks=positions, xticklabels=[f"{row['source']}\nN={row['n']:,}" for row in summary["sources"]],
           ylim=(0, 1.08), ylabel="Macro-F1", title="Validation Macro-F1 by source")
    ax.legend()
    save(fig, "source_macro_f1_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for offset, key, label in ((-.17, "mean_confidence_correct", "Correct"), (.17, "mean_confidence_wrong", "Wrong")):
        axes[0].bar(positions + offset, [summary["confidence"][model][key] if summary["confidence"][model][key] is not None else np.nan for model in MODEL_NAMES], .34, label=label)
    axes[0].set(xticks=positions, xticklabels=MODEL_NAMES, ylim=(0, 1), title="Mean top-1 confidence", ylabel="Confidence")
    axes[0].legend()
    for i, threshold in enumerate(("0.8", "0.9", "0.95")):
        axes[1].bar(positions + (i - 1) * .24,
                    [100 * summary["confidence"][model]["wrong_high_confidence_rates"][threshold]
                     if summary["confidence"][model]["wrong_high_confidence_rates"][threshold] is not None else np.nan for model in MODEL_NAMES],
                    .24, label=f">={threshold}")
    axes[1].set(xticks=positions, xticklabels=MODEL_NAMES, ylabel="% of model errors", title="High-confidence errors")
    axes[1].legend()
    bars = axes[2].bar(MODEL_NAMES, [summary["confidence"][model]["ece"] for model in MODEL_NAMES], color=colors)
    axes[2].bar_label(bars, fmt="%.4f", padding=3)
    axes[2].set(title="ECE | 10 equal-width bins", ylabel="ECE")
    axes[2].margins(y=.25)
    fig.suptitle("Validation confidence analysis; threshold denominator = each model's errors")
    save(fig, "confidence_error_analysis.png")

    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    for i, model in enumerate(MODEL_NAMES):
        ax.plot(range(5), [row[f"{model}_macro_f1"] for row in summary["length_groups"]], "o-", label=model, color=colors[i])
    ax.set(xticks=range(5), xticklabels=[f"{row['length_group']}\nN={row['n']:,}" for row in summary["length_groups"]],
           ylim=(0, 1), ylabel="Macro-F1", xlabel="Whitespace token count (cleaned text)", title="Validation performance by text length")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "length_group_macro_f1.png")

    for model in MODEL_NAMES:
        rows = [row for row in summary["confidence"][model]["bins"] if row["count"]]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        axes[0].plot([row["mean_confidence"] for row in rows], [row["accuracy"] for row in rows], "o-", label=model)
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Reliability | nonempty bins")
        axes[0].legend(fontsize=8)
        all_bins = summary["confidence"][model]["bins"]
        axes[1].bar([row["lower"] for row in all_bins], [row["count"] for row in all_bins], width=.09, align="edge", color=colors[MODEL_NAMES.index(model)])
        axes[1].set(xlim=(0, 1), xlabel="Confidence bin", ylabel="Samples", title="Bin support")
        fig.suptitle(f"Validation calibration | {model.upper()}")
        save(fig, f"reliability_diagram_{model}.png")


def _number(value, digits=6):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                      ["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in rows])


def _write_report(summary):
    complement, effects = summary["complementarity"], summary["ensemble_effects"]
    models, confidence = summary["models"], summary["confidence"]
    lines = ["# Error Analysis & Model Complementarity – Bước 9.1", "", "## A. Dataset và phạm vi",
             f"Validation: **{summary['sample_count']:,} mẫu**, alpha cố định **{summary['alpha']}** đọc từ alpha_selection.json. "
             f"Cache **{summary['provenance']['cache_status']}**, không sweep alpha, không sửa kết quả Bước 9.",
             "Dữ liệu đã cleaned; không gọi clean_text(). Hash, ground truth và source từng dòng, mapping ba nhãn và toàn bộ "
             "fingerprints model/config/vocab/code đã đối chiếu. Sample index từ 0, giữ thứ tự CSV. Macro-F1 khớp Bước 9 với tolerance 1e-12.",
             "", _table(["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro-F1"],
                         [[model] + [_number(models[model][key], 10) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")] for model in MODEL_NAMES]),
             "", "## B. Complementary errors – RQ-A", "",
             _table(["Nhóm", "Mô tả", "N", "Tỷ lệ"], [[key, row["name"], row["count"], f"{row['rate']:.4%}"] for key, row in complement["groups"].items()]),
             f"\nTổng bốn nhóm = {sum(row['count'] for row in complement['groups'].values())}. "
             f"SVM đúng/sai = {complement['svm_correct_count']}/{complement['svm_wrong_count']}; "
             f"Transformer đúng/sai = {complement['transformer_correct_count']}/{complement['transformer_wrong_count']}.",
             f"Transformer rescue rate = C / SVM errors = {_number(complement['transformer_rescue_rate'])}; "
             f"SVM rescue rate = B / Transformer errors = {_number(complement['svm_rescue_rate'])}. Mẫu số 0 trả null/N/A, không coi là tỷ lệ 0.",
             "\n![Correctness groups](../backend/outputs/error_analysis/model_correctness_groups.png)",
             "", "## C. Lỗi Ensemble và Accuracy khác Macro-F1", "",
             _table(["Chuyển đổi", "N"], [[key, effects[key]] for key in (
                 "svm_wrong_to_ensemble_correct", "svm_correct_to_ensemble_wrong",
                 "transformer_wrong_to_ensemble_correct", "transformer_correct_to_ensemble_wrong",
                 "both_wrong_to_ensemble_correct", "both_correct_to_ensemble_wrong")]),
             f"\nEnsemble tăng ròng {effects['net_correct_vs_svm']} mẫu đúng so SVM: "
             f"delta Accuracy={effects['accuracy_difference_vs_svm']:.10f}, delta Macro-F1={effects['macro_f1_difference_vs_svm']:.10f}. "
             "Đây là hai đại lượng khác nhau, không dùng số mẫu đúng thay cho Macro-F1.",
             "Cả hai model sai nhưng Ensemble đúng được đếm trực tiếp, không giả định bằng 0. Nếu hai model có các đỉnh sai "
             "ở hai class khác nhau, phép cộng probability có thể làm class đúng thắng. Unit test có counterexample synthetic; "
             "đó không phải kết quả quan sát thật. Nếu cả hai đều ưu tiên cùng một class sai hơn class đúng, convex average không sửa được tình huống đó.",
             "", "## D. Confusion matrices và per-class – RQ-B", "",
             "Hàng = ground truth, cột = predicted label; thứ tự negative, neutral, positive. Ảnh có cả counts và row-normalized."]
    for model in MODEL_NAMES:
        pair = models[model]["top_confusion"]
        if pair:
            lines.append(f"- {model}: nhầm nhiều nhất **{pair['true_label']} → {pair['predicted_label']}**, {pair['count']} mẫu.")
        lines.append(f"\n![{model} confusion](../backend/outputs/error_analysis/confusion_matrix_{model}.png)\n")
    lines += [_table(["Model", "Label", "Precision", "Recall", "F1", "Support"],
                     [[model, row["label"], _number(row["precision"]), _number(row["recall"]), _number(row["f1"]), row["support"]]
                      for model in MODEL_NAMES for row in models[model]["per_class"]]),
              "", _table(["Class", "Ensemble F1 - SVM F1"], [[row["label"], f"{row['difference_vs_svm']:+.8f}"] for row in effects["class_f1_changes"]]),
              "", "Class có F1 thấp nhất của từng model: " + "; ".join(
                  f"{model}={min(models[model]['per_class'], key=lambda row: row['f1'])['label']}" for model in MODEL_NAMES) + ".",
              "\n![Per-class F1](../backend/outputs/error_analysis/per_class_f1_comparison.png)",
              "", "## E. Phân tích source – RQ-D", "",
              _table(["Source", "N", "SVM F1", "Transformer F1", "Ensemble F1", "TF cứu SVM", "Both wrong", "Ensemble cứu/mất so SVM"],
                     [[row["source"], row["n"], _number(row["svm_macro_f1"]), _number(row["transformer_macro_f1"]),
                       _number(row["ensemble_macro_f1"]), row["svm_wrong_transformer_correct"], row["both_wrong_count"],
                       f"{row['ensemble_rescues_svm']}/{row['ensemble_hurts_svm']}"] for row in summary["sources"]]),
              "\nNguồn khó nhất theo Macro-F1 của từng model: " + "; ".join(
                  f"{model}={min(summary['sources'], key=lambda row: row[model+'_macro_f1'])['source']}" for model in MODEL_NAMES) + ".",
              "Nguồn có nhiều mẫu SVM sai/Transformer đúng nhất: " + max(summary["sources"], key=lambda row: row["svm_wrong_transformer_correct"])["source"] + ".",
              "Source_error_analysis.csv chứa cả Accuracy, F1 từng class, class support, correct/wrong counts và rescue/loss. "
              "Không suy ra label noise hoặc domain shift chỉ từ khác biệt source. Macro-F1 toàn tập không phải trung bình F1 các source.",
              "Thay đổi Ensemble so SVM theo từng nguồn: " + "; ".join(
                  f"{row['source']}: delta Accuracy={row['ensemble_accuracy']-row['svm_accuracy']:+.8f}, "
                  f"delta Macro-F1={row['ensemble_macro_f1']-row['svm_macro_f1']:+.8f}"
                  for row in summary["sources"]) + ". Accuracy và Macro-F1 có thể chuyển động ngược chiều.",
              "\n![Source F1](../backend/outputs/error_analysis/source_macro_f1_comparison.png)",
              "", "## F. Confidence và calibration – RQ-C", "",
              _table(["Model", "Mean correct", "Mean wrong", "Median wrong", "Wrong ≥.8", "Wrong ≥.9", "Wrong ≥.95", "ECE"],
                     [[model] + [_number(confidence[model][key]) for key in ("mean_confidence_correct", "mean_confidence_wrong", "median_confidence_wrong")] +
                      [_number(confidence[model]["wrong_high_confidence_rates"][threshold]) for threshold in ("0.8", "0.9", "0.95")] + [_number(confidence[model]["ece"])]
                      for model in MODEL_NAMES]),
              "\nCác tỷ lệ threshold dùng mẫu số **số prediction sai của chính model**, không phải N toàn tập. So sánh >= trên probability full precision.",
              "ECE = Σ_b (n_b/N) × |accuracy_b - mean_confidence_b|. Dùng 10 equal-width bins [0,.1),...,[.9,1], "
              "bin cuối gồm 1; bin rỗng đóng góp 0. ECE là top-label calibration error phụ thuộc cách chia bin, không phải metric class-wise đầy đủ.",
              _table(["Model", "Mean confidence toàn tập", "Accuracy", "Confidence - Accuracy"],
                     [[model, _number(confidence[model]["mean_confidence"]), _number(confidence[model]["accuracy"]),
                       f"{confidence[model]['confidence_minus_accuracy']:+.8f}"] for model in MODEL_NAMES]),
              "Confidence cao không bảo đảm prediction đúng. Dấu của confidence-accuracy và reliability diagrams mới giúp phân biệt "
              "overconfidence/underconfidence; ECE tuyệt đối một mình không cho biết hướng.",
              "Trên Validation này, độ lệch mean confidence - Accuracy: " + "; ".join(
                  f"{model}={confidence[model]['confidence_minus_accuracy']:+.8f} "
                  f"({'overconfidence ở mức trung bình' if confidence[model]['confidence_minus_accuracy'] > 0 else 'underconfidence ở mức trung bình' if confidence[model]['confidence_minus_accuracy'] < 0 else 'khớp trung bình'})"
                  for model in MODEL_NAMES) + ". Đây là chẩn đoán top-label trên tập hiện tại, không phải kết luận cho mọi miền dữ liệu.",
              "\n![Confidence](../backend/outputs/error_analysis/confidence_error_analysis.png)",
              "", "## G. Oracle potential", "",
              f"Oracle accuracy = {complement['oracle_accuracy']:.10f}; oracle Macro-F1 = {complement['oracle_macro_f1']:.10f}. "
              "Quy tắc: nếu ít nhất một model đúng thì dùng ground truth, nếu cả hai sai dùng prediction SVM. "
              "Oracle dùng ground truth, không deploy được và không phải mô hình thật. Accuracy này là tiềm năng chọn giữa hai "
              "hard labels, không phải upper bound cho mọi probability ensemble (ensemble có thể cứu nhóm cả hai hard label sai).",
              "", "## H. Độ dài văn bản", "",
              _table(["Tokens", "N", "SVM F1", "Transformer F1", "Ensemble F1", "Class counts", "Vượt limit nội dung"],
                     [[row["length_group"], row["n"], _number(row["svm_macro_f1"]), _number(row["transformer_macro_f1"]),
                       _number(row["ensemble_macro_f1"]), json.dumps(row["class_counts"], ensure_ascii=False), row["truncation_exposed_count"]]
                      for row in summary["length_groups"]]),
              "\nĐộ dài = text.split() trên text cleaned, không preprocessing. N<30 được gắn small_sample; nhãn thiếu được ghi rõ, "
              "Macro-F1 vẫn cố định ba lớp. Ngay cả nhóm có đủ ba nhãn cũng có thể có class support nhỏ; xem class_counts. "
              "Nhóm không có mẫu trả N/A, không ghi F1 giả bằng 0.",
              f"Limit nội dung model = {summary['max_content_tokens']} token vì phải dành 1 vị trí CLS. "
              f"Có {sum(row['truncation_exposed_count'] for row in summary['length_groups'])} mẫu vượt limit này. "
              "Đây là exposure theo tokenizer/config, chưa chứng minh truncation là nguyên nhân lỗi. "
              "Khác biệt F1 theo độ dài còn chịu ảnh hưởng class/source mix và sample size; không có thử nghiệm nhân quả trong bước này.",
              "Nhóm dài nhất có " + str(summary["length_groups"][-1]["n"]) + " mẫu, class counts=" +
              json.dumps(summary["length_groups"][-1]["class_counts"], ensure_ascii=False) +
              "; cần thận trọng vì subgroup/class support nhỏ. Mức giảm Macro-F1 từ 1-10 token đến >128 token: " +
              "; ".join(f"{model}={summary['length_groups'][0][model+'_macro_f1']-summary['length_groups'][-1][model+'_macro_f1']:.6f}"
                         for model in MODEL_NAMES if summary["length_groups"][0][model+"_macro_f1"] is not None and summary["length_groups"][-1][model+"_macro_f1"] is not None) +
              ". Cả SVM cũng thay đổi theo độ dài; chưa thể quy toàn bộ khác biệt cho truncation của Transformer.",
              "\n![Length F1](../backend/outputs/error_analysis/length_group_macro_f1.png)",
              "", "## I. Ví dụ lỗi có tính quyết định", "",
              "CSV giữ nguyên text và sample_index 0-based trong Validation. Chọn tối đa 10 mẫu mỗi nhóm theo confidence "
              "liên quan giảm dần, hòa thì index tăng; hợp các nhóm rồi deduplicate index. Đây là tập ví dụ có chủ đích, "
              "không phải random sample đại diện. error_group là complementary group, ensemble_effect là cột độc lập. "
              "Không tự gán nguyên nhân lỗi bằng keyword."]
    for reason in ("svm_wrong_transformer_correct", "svm_correct_transformer_wrong", "both_wrong", "ensemble_rescues_svm", "ensemble_hurts_svm"):
        candidates = [row for row in summary["examples"] if reason in row["selection_reasons"].split(";")]
        if candidates:
            row = min(candidates, key=lambda item: item["sample_index"])
            lines += [f"\n### {reason}: index {row['sample_index']} ({row['source']})",
                      f"True={row['true_label']}; SVM={row['svm_label']} ({row['svm_confidence']:.6f}); "
                      f"Transformer={row['transformer_label']} ({row['transformer_confidence']:.6f}); "
                      f"Ensemble={row['ensemble_label']} ({row['ensemble_confidence']:.6f}).",
                      "", "```text", row["text"], "```"]
    lines += ["", "## J. Ba quan sát/giả thuyết nổi bật và khuyến nghị – RQ-E", "",
              f"1. **Ranh giới class**: nhóm cả hai sai có {complement['both_wrong_count']} mẫu; các confusion counts/F1 ở mục D "
              "cho thấy nơi cần rà soát. Khó phân biệt ngữ nghĩa hoặc annotation ambiguity là giả thuyết cần kiểm tra thủ công, "
              "không phải kết luận nhãn sai. Không sửa preprocessing đang khớp hash chỉ để khớp ví dụ Validation.",
              f"2. **Calibration/Transformer**: Macro-F1 Transformer {models['transformer']['macro_f1']:.6f} so SVM "
              f"{models['svm']['macro_f1']:.6f}; ECE {confidence['transformer']['ece']:.6f} so {confidence['svm']['ece']:.6f}; "
              f"tỷ lệ lỗi confidence>=.9 là {_percent(confidence['transformer']['wrong_high_confidence_rates']['0.9'])} so "
              f"{_percent(confidence['svm']['wrong_high_confidence_rates']['0.9'])}. Ưu tiên **Phần 2** chẩn đoán lớp yếu và calibration "
              "trước khi tăng độ phức tạp Ensemble; không có bằng chứng ở đây rằng tăng head/layer sẽ tốt hơn.",
              "3. **Độ dài/source**: F1 và class support theo nhóm ở mục E/H cho thấy độ ổn định khác nhau. "
              "Đây là liên hệ quan sát, không đủ kết luận domain shift hoặc truncation gây lỗi. Cần matched-case audit và "
              "thiết kế kiểm chứng riêng trong tương lai, không điều chỉnh model ở bước này.",
              f"**Data/preprocessing:** ưu tiên audit thủ công nhóm D ({complement['both_wrong_count']} mẫu), class/source có F1 yếu; "
              "kiểm tra hướng dẫn nhãn và tính đầy đủ thông tin trong cleaned text, chưa khẳng định label noise và không đổi dataset.",
              "**SVM/Phần 1:** giữ làm baseline đối chiếu, tập trung cặp nhầm lớn nhất và neutral theo source từ bảng D/E; "
              "chưa thêm rule hoặc retrain theo ví dụ Validation.",
              f"**Phần 3:** Transformer có thể đúng ở {complement['groups']['C']['count']} lỗi SVM, nhưng Ensemble chỉ sửa "
              f"{effects['svm_wrong_to_ensemble_correct']} và làm sai thêm {effects['svm_correct_to_ensemble_wrong']}. "
              "Có cơ sở nghiên cứu calibration/combination có điều kiện sau này, nhưng oracle cần ground truth nên không phải "
              "gating deployable. Mọi thử nghiệm mới cần protocol Validation đã định trước; không chọn alpha mới hoặc BEST_METHOD ở đây.",
              "", "## K. Reproducibility và giới hạn", "",
              "Chỉ Validation. Không mở test.csv, train/fine-tune, sửa model/weights/vocab/preprocessing, chọn BEST_METHOD "
              "hoặc aggregation tối ưu. Không chạy lại alpha sweep. Cache hợp lệ được tái sử dụng; chỉ fallback cleaned-batch "
              "khi cache lỗi nhưng frozen artifacts vẫn khớp; các metric lệch Bước 9 sẽ dừng trước khi xuất báo cáo.",
              "summary.json ghi source fingerprints, hash selection/cache/CSV, mapping, cache status và kiểm tra baseline. "
              "Bước 9 đã chọn alpha trên chính Validation; phân tích này là diagnostic của tập đã dùng lựa chọn, không phải "
              "đánh giá tổng quát hóa độc lập hoặc chứng minh ý nghĩa thống kê.",
              "Giới hạn alignment: cache Bước 9 không có text hash riêng từng dòng hoặc chữ ký integrity độc lập. "
              "Kiểm tra thứ tự dựa trên hash của toàn bộ CSV/script tạo cache và đối chiếu chính xác y_true/source từng dòng. "
              "Việc gắn probability với text còn dựa vào thứ tự mà script Bước 9 đã ghi; các kiểm tra này không phát hiện "
              "mọi hoán đổi probability giữa các dòng có cùng label/source nếu aggregate metrics vẫn giữ nguyên. "
              "Không quan sát thấy sai lệch cache trong lần chạy này; không inference lại chỉ để tạo bằng chứng mới.",
              "", "```powershell", "python -m unittest discover backend/tests", "python -m backend.run_error_analysis", "```",
              "", "Dừng tại Bước 9.1; chưa thực hiện Bước 10."]
    REPORT_PATH.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main():
    inputs = load_analysis_inputs()
    data = inputs["data"]
    print(f"Validation {len(data['texts'])} samples; alpha={inputs['alpha']}; cache={inputs['cache_status']}", flush=True)
    config = json.loads((BACKEND / "models" / "transformer" / "config.json").read_text(encoding="utf-8"))
    summary = analyze_validation_errors(data["labels"], inputs["svm_probs"], inputs["transformer_probs"],
                                        data["texts"], data["sources"], inputs["alpha"], config["max_len"] - 1)
    for row in summary["sources"]:
        frozen = next(value for value in inputs["selection"]["by_source"] if value["source"] == row["source"])
        for model, frozen_key in (("svm", "svm_macro_f1"), ("transformer", "transformer_macro_f1"), ("ensemble", "selected_alpha_macro_f1")):
            if not np.isclose(row[f"{model}_macro_f1"], frozen[frozen_key], atol=1e-12, rtol=0):
                raise ValueError("Source Macro-F1 không khớp Bước 9.")
    if collect_fingerprints() != inputs["selection"]["fingerprints"]:
        raise ValueError("Frozen inputs thay đổi trong analysis; dừng trước output.")
    summary["provenance"] = {
        "dataset": "validation", "dataset_sha256": data["sha256"],
        "selection_path": str(SELECTION_PATH), "selection_sha256": _sha(SELECTION_PATH),
        "cache_path": str(CACHE_PATH), "cache_sha256": _sha(CACHE_PATH) if CACHE_PATH.is_file() else None,
        "cache_status": inputs["cache_status"], "cache_reason": inputs["cache_reason"],
        "model_data_code_fingerprints_step9": inputs["selection"]["fingerprints"],
        "analysis_code_sha256": {"error_analysis.py": _sha(BACKEND / "src" / "error_analysis.py"), "run_error_analysis.py": _sha(Path(__file__))},
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_dataset_used": False,
        "alpha_retuned": False, "step9_metrics_verified": True, "row_alignment_verified": True,
        "alignment_checks": "Frozen dataset SHA and runner provenance + exact y_true/source row arrays + shape + class mapping.",
        "alignment_limitations": "No per-row text hashes or independent integrity signature in Step-9 cache. Probability/text correspondence relies on frozen producer ordering; equal-label/source probability permutations preserving metrics are not independently detectable.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _csv(OUTPUT_DIR / "per_class_metrics.csv", [{"model": model, **row} for model in MODEL_NAMES for row in summary["models"][model]["per_class"]])
    _csv(OUTPUT_DIR / "source_error_analysis.csv", summary["sources"])
    _csv(OUTPUT_DIR / "length_group_metrics.csv", summary["length_groups"])
    _csv(OUTPUT_DIR / "confusion_pairs.csv", [{"model": model, **row} for model in MODEL_NAMES for row in summary["models"][model]["confusion_pairs"]])
    _csv(OUTPUT_DIR / "confidence_summary.csv", [{"model": model, **{key: value for key, value in summary["confidence"][model].items() if key != "bins"}} for model in MODEL_NAMES])
    _csv(OUTPUT_DIR / "error_examples.csv", summary["examples"], ["sample_index", "source", "text", "true_label", "error_group", "ensemble_effect"])
    if inputs["cache_status"] == "recomputed":
        np.savez_compressed(OUTPUT_DIR / "recomputed_validation_probabilities.npz", svm_probs=inputs["svm_probs"],
                            transformer_probs=inputs["transformer_probs"], y_true=data["labels"], sources=np.array(data["sources"]))
    _plot_all(summary)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(summary)
    print("Groups:", json.dumps(summary["complementarity"]["groups"]))
    print("Rescue TF / SVM:", summary["complementarity"]["transformer_rescue_rate"], summary["complementarity"]["svm_rescue_rate"])
    print("Oracle accuracy:", summary["complementarity"]["oracle_accuracy"])
    print("Ensemble effects:", json.dumps(summary["ensemble_effects"], ensure_ascii=False))
    print("Confidence:", json.dumps({model: {key: value for key, value in summary["confidence"][model].items() if key != "bins"} for model in MODEL_NAMES}))
    print(f"Outputs: {OUTPUT_DIR}\nReport: {REPORT_PATH}")
    print("Validation diagnostics only; no Test, retraining, alpha tuning or model/aggregation selection.")


if __name__ == "__main__":
    main()
