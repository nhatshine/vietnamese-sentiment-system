"""Final Test evaluation with frozen Validation selection; no tuning or training."""

from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from .run_alpha_tuning import BACKEND, collect_fingerprints
from .src.alpha_tuning import compute_ensemble_probabilities
from .src.config import LABELS
from .src.error_analysis import MODEL_NAMES
from .src.final_evaluation import (
    evaluate_test, file_sha256, load_frozen_selection, load_test_dataset, ordered_row_ids, prediction_rows,
)

OUTPUT_DIR = BACKEND / "outputs" / "final_evaluation"
REPORT_PATH = BACKEND.parent / "docs" / "FINAL_MODEL_EVALUATION.md"
SELECTION_PATH = BACKEND / "outputs" / "alpha_tuning" / "alpha_selection.json"
TEST_PATH = BACKEND / "data" / "test.csv"
SVM_HANDOFF = Path("C:/Users/Admin/data+model_SVM/data_final/final/test.csv")
TRANSFORMER_ARCHIVE = Path("C:/Users/Admin/train_Transformer.zip")


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_test_dataset(target, svm_source, archive, expected_sha, *, expected_rows=8238):
    """Verify original Test bytes/config before copying a missing project CSV."""
    target, svm_source, archive = Path(target), Path(svm_source), Path(archive)
    svm_sha = file_sha256(svm_source)
    with zipfile.ZipFile(archive) as zipped:
        archive_bytes = zipped.read("data/test.csv")
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    if svm_sha != expected_sha or archive_sha != expected_sha:
        raise ValueError("Test handoff SHA-256 không khớp nhau hoặc training config; dừng.")
    # Validate all rows before creating a project copy. Never repair an existing mismatch.
    load_test_dataset(svm_source, expected_sha, expected_rows=expected_rows)
    if target.exists():
        data = load_test_dataset(target, expected_sha, expected_rows=expected_rows)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive_bytes)
        data = load_test_dataset(target, expected_sha, expected_rows=expected_rows)
    provenance = {"dataset": "test", "preprocessed": True, "sha256": expected_sha,
                  "svm_handoff_source": str(svm_source), "svm_handoff_test_sha256": svm_sha,
                  "transformer_archive_source": str(archive) + "!data/test.csv",
                  "transformer_archive_test_sha256": archive_sha,
                  "transformer_training_config_test_sha256": expected_sha,
                  "svm_training_dataset_hash_in_model_metadata": None,
                  "verification_note": "Both handoff Test CSVs and project copy are byte-identical; Transformer training config records the same SHA. SVM metadata has no training-dataset SHA."}
    return data, provenance


def _progress(name):
    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(f"{name}: {done}/{total}", flush=True)
    return progress


def infer_cleaned(texts):
    from .src.real_svm import predict_svm_cleaned_batch
    from .src.real_transformer import predict_transformer_cleaned_batch
    svm = predict_svm_cleaned_batch(texts, progress=_progress("SVM Test"))
    transformer = predict_transformer_cleaned_batch(texts, batch_size=64, device="cpu", progress=_progress("Transformer Test"))
    return tuple(np.array([[row[label] for label in LABELS] for row in rows], dtype=float) for rows in (svm, transformer))


def load_or_predict(data, frozen, output):
    """Persist the freeze before inference. Invalid existing artifacts stop, never silently rerun."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = ordered_row_ids(data)
    configuration = {**frozen, "test_sha256": data["sha256"], "sample_count": len(ids),
                     "ordered_row_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    freeze_path = output / "evaluation_freeze.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding="utf-8"))["configuration"] != configuration:
            raise ValueError("Frozen configuration/Test rows đã đổi; không chọn/tune lại bằng Test.")
    else:
        _json(freeze_path, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                            "selected_before_test_inference": True, "configuration": configuration})
    cache_path, record_path = output / "test_probabilities.npz", output / "inference_record.json"
    if cache_path.exists() or record_path.exists():
        if not cache_path.is_file() or not record_path.is_file():
            raise ValueError("Test cache/record chưa đầy đủ; dừng để tránh inference lặp không chủ ý.")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["cache_sha256"] != file_sha256(cache_path) or record["freeze_sha256"] != file_sha256(freeze_path):
            raise ValueError("Test cache integrity không khớp; không inference lại tự động.")
        with np.load(cache_path, allow_pickle=False) as cache:
            if (json.loads(cache["manifest_json"].item()) != configuration or cache["row_ids"].tolist() != ids
                    or not np.array_equal(cache["y_true"], data["labels"])
                    or cache["sources"].tolist() != data["sources"]):
                raise ValueError("Test cache manifest/row alignment không khớp.")
            svm, transformer = cache["svm_probs"].copy(), cache["transformer_probs"].copy()
        status = "reused"
    else:
        svm, transformer = infer_cleaned(data["texts"])
        compute_ensemble_probabilities(svm, transformer, frozen["selected_alpha"])
        if len(svm) != len(ids):
            raise ValueError("Inference sample alignment không khớp Test.")
        np.savez_compressed(cache_path, svm_probs=svm, transformer_probs=transformer,
                            y_true=data["labels"], sources=np.array(data["sources"]), row_ids=np.array(ids),
                            manifest_json=np.array(json.dumps(configuration, sort_keys=True)))
        _json(record_path, {"inference_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "cache_sha256": file_sha256(cache_path), "freeze_sha256": file_sha256(freeze_path),
                            "svm_cleaned_batch_calls": 1, "transformer_cleaned_batch_calls": 1,
                            "device": "cpu", "transformer_batch_size": 64, "preprocessing_repeated": False,
                            "retrained": False, "alpha_retuned": False})
        status = "computed_once"
    compute_ensemble_probabilities(svm, transformer, frozen["selected_alpha"])
    if svm.shape != (len(ids), 3):
        raise ValueError("Test probability shape không khớp.")
    return svm, transformer, status


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def compare_handoff_results(result, svm_meta_path, archive):
    """Post-hoc consistency only: never feeds selection, alpha or inference."""
    svm_meta_path, archive = Path(svm_meta_path), Path(archive)
    svm = json.loads(svm_meta_path.read_text(encoding="utf-8"))["test"]["overall"]
    with zipfile.ZipFile(archive) as zipped:
        member = zipped.read("transformer/metrics/metrics.json")
    transformer = json.loads(member)
    comparisons = {}
    for name, original in (("svm", svm), ("transformer", transformer["test"])):
        delta = {key: result["models"][name][key] - original[key] for key in ("accuracy", "macro_f1")}
        comparisons[name] = {"original_metrics": {key: original[key] for key in delta},
                             "differences": delta, "matches_within_tolerance": all(abs(v) <= 1e-12 for v in delta.values())}
    return {"performed_after_integrated_test_metrics": True, "used_for_selection": False,
            "absolute_tolerance": 1e-12, "models": comparisons,
            "svm_metadata_sha256": file_sha256(svm_meta_path),
            "transformer_metrics_member_sha256": hashlib.sha256(member).hexdigest(),
            "handoff_split_sizes": transformer.get("split_sizes")}


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows])


def _plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#2166ac", "#d97706", "#16836b")
    models = result["models"]

    def save(fig, name):
        fig.savefig(output / name, dpi=180); plt.close(fig)

    for metric, title in (("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")):
        fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
        bars = ax.bar(MODEL_NAMES, [models[m][metric] for m in MODEL_NAMES], color=colors)
        ax.bar_label(bars, fmt="%.6f", padding=4)
        ax.set(ylim=(0,1.08), ylabel=title, title=f"Final Test {title} | frozen alpha={result['selected_alpha']}")
        save(fig, f"test_{metric}_comparison.png")

    # A shared count scale and label order across all three matrices.
    maximum = max(max(max(row) for row in models[m]["confusion_matrix"]) for m in MODEL_NAMES)
    for model in MODEL_NAMES:
        fig, axes = plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
        for ax, key, title, vmax in zip(axes, ("confusion_matrix","confusion_matrix_row_normalized"), ("Counts","Row-normalized"),(maximum,1)):
            values = np.array(models[model][key])
            im = ax.imshow(values,cmap="Blues",vmin=0,vmax=vmax)
            for i in range(3):
                for j in range(3):
                    label = str(int(values[i,j])) if key=="confusion_matrix" else f"{values[i,j]:.3f}"
                    ax.text(j,i,label,ha="center",va="center",color="white" if values[i,j]>vmax/2 else "black")
            ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
                   xlabel="Predicted label",ylabel="Ground truth",title=title)
            fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle(f"Final Test | {model.upper()}")
        save(fig,f"test_confusion_matrix_{model}.png")

    for kind in ("class","source"):
        fig, ax = plt.subplots(figsize=(8.5,4.8),layout="constrained")
        if kind=="class":
            labels=LABELS
            values={m:[r["f1"] for r in models[m]["per_class"]] for m in MODEL_NAMES}
        else:
            labels=[f"{r['source']}\nN={r['n']:,}" for r in result["sources"]]
            values={m:[r["models"][m]["macro_f1"] if r["n"] else np.nan for r in result["sources"]] for m in MODEL_NAMES}
        for j,m in enumerate(MODEL_NAMES):
            bars=ax.bar(np.arange(3)+(j-1)*.24,values[m],.24,label=m,color=colors[j])
            ax.bar_label(bars,fmt="%.5f",padding=3,fontsize=8)
        ax.set(xticks=range(3),xticklabels=labels,ylim=(0,1.12),ylabel="F1" if kind=="class" else "Macro-F1",
               title="Final Test per-class F1" if kind=="class" else "Final Test Macro-F1 by source")
        ax.legend()
        save(fig,"test_per_class_f1.png" if kind=="class" else "test_source_f1_comparison.png")

    fig,ax=plt.subplots(figsize=(8,4.6),layout="constrained")
    for offset,key,label in ((-.18,"validation_macro_f1","Validation"),(.18,"test_macro_f1","Test")):
        bars=ax.bar(np.arange(3)+offset,[r[key] for r in result["validation_vs_test"]],.36,label=label)
        ax.bar_label(bars,fmt="%.5f",padding=3,fontsize=8)
    ax.set(xticks=range(3),xticklabels=MODEL_NAMES,ylim=(0,1.12),ylabel="Macro-F1",title="Frozen Validation vs final Test")
    ax.legend(); save(fig,"validation_vs_test_f1.png")


def _report(result, output, report_path):
    models=result["models"]; data=result["dataset"]; delta=result["ensemble_deltas"]
    positive=sum(v>0 for v in delta.values())
    conclusion=("Ensemble tốt hơn cả hai model trên Test." if positive==2 else
                "Ensemble chỉ tốt hơn một model trên Test." if positive==1 else
                "Ensemble không cải thiện các model riêng lẻ trên Test.")
    source_deltas=[(r['source'],r['models']['ensemble']['macro_f1']-r['models']['svm']['macro_f1']) for r in result['sources'] if r['n']]
    class_deltas=[models['ensemble']['per_class'][i]['f1']-models['svm']['per_class'][i]['f1'] for i in range(3)]
    consistency=result.get('handoff_consistency')
    train_n=(consistency or {}).get('handoff_split_sizes') or {}
    image_base=Path(output).resolve().as_posix()
    lines=["# Bước 10 — Final Model Evaluation on Test", "",
           "## 1. Mục tiêu thực nghiệm", "",
           "Đánh giá cuối ba phương án trên cùng Test nguyên gốc. Lựa chọn mô hình và alpha chỉ dựa trên Validation, trước Test inference; không tối ưu theo Test.", "",
           "## 2. Dataset", "",
           f"Train: {train_n.get('train', 'xem metadata bàn giao')} mẫu theo metadata, đã dùng huấn luyện trong các bộ bàn giao; không train lại hoặc đánh giá lại Train ở đây. Validation Bước 9: 8.238 mẫu. Test: **{data['sample_count']:,} mẫu**.",
           f"Test SHA-256: `{data['sha256']}`. Hai Test CSV bàn giao, bản project và hash training config được đối chiếu trước inference. SVM không cung cấp training-dataset SHA trong metadata; không suy diễn bằng chứng này.",
           "Dữ liệu text đã cleaned, dùng nguyên văn/thứ tự, không clean_text lại, không loại/shuffle mẫu. Nhãn 0=negative, 1=neutral, 2=positive.",
           _table(["Label","N"],[[k,v] for k,v in data['label_counts'].items()]),
           _table(["Source","N"],[[k,v] for k,v in data['source_counts'].items()]), "",
           "## 3. Cấu hình đã khóa", "",
           f"SVM đã huấn luyện và Transformer tự xây giữ nguyên checkpoint/vocab/tokenizer/preprocessing. Alpha = **{result['selected_alpha']}**; selected_method = **{result['selected_method']}**, xác định từ Validation Macro-F1 trước Test inference. Không đổi lựa chọn theo thứ hạng Test.",
           "Transformer chạy CPU, batch64, đường inference cũ dùng eval()/torch.inference_mode(); mỗi model gọi cleaned-batch một lượt. Hash model/config/vocab/code, mapping và môi trường được ghi trong evaluation_freeze.json trước inference.",
           "test_probabilities.npz lưu probability full precision, ordered row IDs (SHA256 của index/text/label/source), manifest và hash integrity ngoài cache. Chạy lại script chỉ tái xuất từ cache hợp lệ; cache/freeze sai thì dừng.", "",
           "## 4. Phương pháp đánh giá", "",
           "Dùng sklearn.metrics; macro Precision/Recall/F1 tính trực tiếp trên toàn tập, labels=[0,1,2], average='macro', zero_division=0. Weighted-F1 tính theo support. Không trung bình source F1 hoặc F1 hai model để thay metric tổng. Ensemble dùng hàm vector hóa hiện có, không làm tròn probability.", "",
           "## 5. Kết quả tổng thể — RQ1", "",
           _table(["Metric","SVM","Transformer","Ensemble"],[[key]+[f"{models[m][key]:.10f}" for m in MODEL_NAMES] for key in ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1')]),
           conclusion,
           _table(["So sánh Macro-F1","Chênh lệch tuyệt đối","Điểm phần trăm"],[[k,f"{v:+.10f}",f"{100*v:+.6f}"] for k,v in delta.items()]),
           "Các delta là quan sát trên Test, chưa kiểm định ý nghĩa thống kê. Không thay alpha/selected_method nếu kết quả không như kỳ vọng.",
           f"![Test Macro-F1]({image_base}/test_macro_f1_comparison.png)",
           f"![Test Accuracy]({image_base}/test_accuracy_comparison.png)", "",
           "## 6. Theo class và confusion — RQ2", "",
           _table(["Model","Label","Precision","Recall","F1","Support"],[[m,r['label'],f"{r['precision']:.6f}",f"{r['recall']:.6f}",f"{r['f1']:.6f}",r['support']] for m in MODEL_NAMES for r in models[m]['per_class']]),
           "Class F1 thấp nhất trên Test: " + "; ".join(f"{m}={min(models[m]['per_class'],key=lambda r:r['f1'])['label']}" for m in MODEL_NAMES) + ".",
           "So với SVM trên Test, Ensemble: " + "; ".join(f"{label} {'tăng' if value>0 else 'giảm' if value<0 else 'không đổi'} F1 ({value:+.8f})" for label,value in zip(LABELS,class_deltas)) + ".",
           "Bước 9.1: neutral thấp nhất, negative→neutral nhầm nhiều nhất; Ensemble tăng positive và giảm nhẹ neutral. Bảng dưới đối chiếu trực tiếp Test, không giả định các hiện tượng lặp lại.",
           _table(["Label","Test Ensemble F1 − SVM F1"],[[LABELS[i],f"{models['ensemble']['per_class'][i]['f1']-models['svm']['per_class'][i]['f1']:+.8f}"] for i in range(3)]),
           "Confusion matrices: hàng ground truth, cột predicted; thứ tự negative/neutral/positive. Cả ba ảnh dùng cùng thang counts và normalized [0,1].",
           _table(["Model","Cặp nhầm nhiều nhất","N"],[[m,(f"{models[m]['top_confusion']['true_label']} → {models[m]['top_confusion']['predicted_label']}" if models[m]['top_confusion'] else "Không có lỗi"),(models[m]['top_confusion']['count'] if models[m]['top_confusion'] else 0)] for m in MODEL_NAMES]),
           f"![Per-class F1]({image_base}/test_per_class_f1.png)"]
    for m in MODEL_NAMES:
        lines += [f"![Confusion {m}]({image_base}/test_confusion_matrix_{m}.png)"]
    lines += ["", "## 7. Theo source — RQ3", "",
              _table(["Source","N","negative/neutral/positive","SVM Macro-F1","Transformer Macro-F1","Ensemble Macro-F1"],[[r['source'],r['n'],'/'.join(str(r['label_counts'][k]) for k in LABELS)]+[f"{r['models'][m]['macro_f1']:.8f}" for m in MODEL_NAMES] for r in result['sources'] if r['n']]),
              "Nguồn có Macro-F1 cao nhất/thấp nhất theo từng phương án: " + "; ".join(f"{m}={max((r for r in result['sources'] if r['n']),key=lambda r:r['models'][m]['macro_f1'])['source']}/{min((r for r in result['sources'] if r['n']),key=lambda r:r['models'][m]['macro_f1'])['source']}" for m in MODEL_NAMES) + ".",
              f"Ensemble cải thiện Macro-F1 so SVM trên {sum(v>0 for _,v in source_deltas)}/{len(source_deltas)} nguồn. " + "; ".join(f"{k}: {v:+.8f}" for k,v in source_deltas) + ". Không chọn alpha theo source.",
              "test_source_metrics.csv chứa Accuracy, Macro-F1, F1 từng class và support. Macro-F1 cố định ba lớp; cần xem support khi diễn giải subgroup.",
              f"![Source F1]({image_base}/test_source_f1_comparison.png)", "",
              "## 8. Validation vs Test — RQ4", "",
              _table(["Model","Validation Macro-F1","Test Macro-F1","Test − Validation"],[[r['model'],f"{r['validation_macro_f1']:.10f}",f"{r['test_macro_f1']:.10f}",f"{r['test_minus_validation']:+.10f}"] for r in result['validation_vs_test']]),
              "Mức chênh lệch tuyệt đối lớn nhất = " + f"{max(abs(r['test_minus_validation']) for r in result['validation_vs_test']):.8f}" + ". Delta âm biểu thị giảm F1 ở Test; delta dương biểu thị tăng. Một chênh lệch nhỏ không đủ kết luận chắc chắn overfitting hoặc cơ chế nhân quả.",
              f"![Validation vs Test]({image_base}/validation_vs_test_f1.png)", "",
              "## 9. Thảo luận", "",
              f"RQ1: {conclusion} Alpha và selected_method vẫn giữ theo Validation. RQ2: class yếu nhất và hướng confusion lấy từ Test ở mục 6; hướng thay đổi F1 neutral của Ensemble so SVM là " +
              ("tăng" if class_deltas[1]>0 else "giảm" if class_deltas[1]<0 else "không đổi") +
              f", cần phân biệt với giảm nhẹ ở Validation. RQ3: Ensemble so SVM tăng trên {sum(v>0 for _,v in source_deltas)}/{len(source_deltas)} nguồn; không coi một tăng tổng thể là tăng đồng đều mọi source.",
              "Không chọn BEST_METHOD mới bằng Test. Bước 11 sau này chỉ đóng gói cấu hình lựa chọn từ Validation. Không dùng kết quả Test đã thấy để thêm rule hoặc điều chỉnh mô hình.", "",
              "## 10. Hạn chế và tái lập", "",
              "Transformer tự xây với kiến trúc/vocabulary cố định có thể còn giới hạn năng lực; thực nghiệm này không xác định lợi ích của tăng layer/head. Weighted Ensemble có thể chỉ thay đổi nhẹ; chưa kiểm định thống kê. Khác biệt source chưa chứng minh label noise/domain shift. Confidence không bảo đảm đúng.",
              "Document Aggregation chưa có benchmark document-level độc lập phù hợp; không gộp các dòng Test hoặc chọn Majority Voting/Mean Probability/Confidence Weighted theo nhãn câu.",
              "Các bộ bàn giao đã có báo cáo Test riêng từ trước. Chúng không tham gia chọn alpha/method; đánh giá này là lần đánh giá cấu hình tích hợp đã khóa, không tuyên bố Test chưa từng được chủ model xem. Mọi đối chiếu báo cáo bàn giao chỉ thực hiện sau khi hoàn tất inference/metrics.",
              ("Đối chiếu sau thực nghiệm với Accuracy/Macro-F1 Test trong hai bộ bàn giao (tolerance 1e-12): " +
               "; ".join(f"{m}={'khớp' if r['matches_within_tolerance'] else 'không khớp'}" for m,r in consistency['models'].items()) +
               ". Kết quả chi tiết/hash nằm trong handoff_consistency của test_metrics.json; không dùng để đổi cấu hình."
               if consistency else "Đối chiếu số liệu bàn giao không áp dụng cho dữ liệu synthetic này."),
              "Không retrain/fine-tune, không sửa tokenizer/vocab/preprocessing, không tune alpha, không chọn aggregation; chưa xây FastAPI/React. Dừng tại Bước 10.",
              "```powershell\npython -m unittest discover backend/tests\npython -m backend.run_final_evaluation\n```"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join(lines)+"\n",encoding="utf-8")


def export_results(result, data, matrices, output, report_path):
    output, report_path = Path(output), Path(report_path)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "test_metrics.json", result)
    _csv(output / "test_predictions.csv", prediction_rows(data, matrices))
    _csv(output / "test_per_class_metrics.csv", [{"model":m,**r} for m in MODEL_NAMES for r in result['models'][m]['per_class']])
    source_rows=[]
    for r in result['sources']:
        for m,v in r['models'].items():
            source_rows.append({"source":r['source'],"model":m,"n":r['n'],
                                **{f"support_{k}":n for k,n in r['label_counts'].items()},
                                "accuracy":v['accuracy'],"macro_f1":v['macro_f1'],
                                **{f"f1_{p['label']}":p['f1'] for p in v['per_class']}})
    _csv(output / "test_source_metrics.csv", source_rows)
    _plots(result,output)
    _report(result,output,report_path)


def main():
    # Selection is resolved before Test access. This is an evaluation freeze, not deployment config.
    frozen=load_frozen_selection(SELECTION_PATH)
    if collect_fingerprints()!=frozen['step9_fingerprints']:
        raise ValueError("Model/config/vocab/preprocess/Validation đã đổi từ Bước 9; dừng.")
    versions={p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','torch','joblib','matplotlib')}
    frozen.update(package_versions=versions,python_version=sys.version,inference_device='cpu',transformer_batch_size=64)
    print(f"Frozen from Validation: method={frozen['selected_method']}; alpha={frozen['selected_alpha']}",flush=True)
    config=json.loads((BACKEND/'models'/'transformer'/'config.json').read_text(encoding='utf-8'))
    data,provenance=prepare_test_dataset(TEST_PATH,SVM_HANDOFF,TRANSFORMER_ARCHIVE,config['dataset_sha256']['test'])
    _json(BACKEND/'data'/'test_provenance.json',provenance)
    print(f"Test verified: N={data['sample_count']}; SHA256={data['sha256']}; sources={data['source_counts']}",flush=True)
    svm,transformer,status=load_or_predict(data,frozen,OUTPUT_DIR)
    if collect_fingerprints()!=frozen['step9_fingerprints'] or file_sha256(TEST_PATH)!=data['sha256'] or file_sha256(SELECTION_PATH)!=frozen['selection_sha256']:
        raise ValueError("Frozen inputs thay đổi trong evaluation; không xuất metrics.")
    result,matrices=evaluate_test(data['labels'],svm,transformer,data['sources'],frozen)
    result['dataset']={k:v for k,v in data.items() if k not in ('texts','labels','sources')}
    result['dataset']['provenance']=provenance
    result['frozen_configuration']=frozen
    result['handoff_consistency']=compare_handoff_results(
        result, Path('C:/Users/Admin/data+model_SVM/model_SVM/models/svm_meta.json'), TRANSFORMER_ARCHIVE)
    result['provenance']={"cache_status":status,"selected_before_test_inference":True,
                          "selection_sha256":frozen['selection_sha256'],"model_fingerprints":frozen['step9_fingerprints'],
                          "evaluation_freeze_sha256":file_sha256(OUTPUT_DIR/'evaluation_freeze.json'),
                          "inference_record":json.loads((OUTPUT_DIR/'inference_record.json').read_text(encoding='utf-8')),
                          "evaluation_code_sha256":{p.name:file_sha256(p) for p in (Path(__file__),BACKEND/'src'/'final_evaluation.py')},
                          "created_at_utc":datetime.now(timezone.utc).isoformat(),"retrained":False,"alpha_retuned":False,
                          "test_based_method_selection":False,"aggregation_evaluated":False}
    export_results(result,data,matrices,OUTPUT_DIR,REPORT_PATH)
    print('\nMetric                 SVM          Transformer  Ensemble')
    for key in ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1'):
        print(f"{key:22s} "+' '.join(f"{result['models'][m][key]:.10f}" for m in MODEL_NAMES))
    print('Ensemble deltas:',result['ensemble_deltas'])
    print('Validation vs Test:',result['validation_vs_test'])
    print(f"Cache={status}; outputs={OUTPUT_DIR}; report={REPORT_PATH}")
    print('Frozen Validation selection unchanged; no retraining/tuning/aggregation selection/API/UI. Stop at Step 10.')


if __name__=='__main__':
    main()
