"""Chỉ tuning trên Validation: python -m backend.run_alpha_tuning."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from .src.alpha_tuning import (
    DEFAULT_ALPHAS, compute_ensemble_probabilities, evaluate_probabilities, tune_alpha,
)
from .src.config import LABELS, LABEL_TO_ID


BACKEND = Path(__file__).resolve().parent
VALIDATION_PATH = BACKEND / "data" / "val.csv"
OUTPUT_DIR = BACKEND / "outputs" / "alpha_tuning"
SOURCES = ("hieu_sa", "neu_esc", "vsfc")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation(path: Path, expected_sha256: str, *, expected_rows: int = 8238,
                    expected_sources=SOURCES) -> dict:
    """Kiểm tra hash và từng dòng; không clean, bỏ dòng hoặc đọc file Test."""
    path = Path(path)
    if path.name != "val.csv":
        raise ValueError("Thực nghiệm chỉ chấp nhận Validation tên val.csv, không nhận Test.")
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu Validation: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise ValueError(f"Validation SHA-256 không khớp metadata huấn luyện: {actual_hash}")
    texts, labels, sources = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "label", "source"}.issubset(reader.fieldnames or []):
            raise ValueError("Validation phải có các cột text, label, source.")
        for line, row in enumerate(reader, start=2):
            text, label, source = row.get("text"), row.get("label"), row.get("source")
            if not isinstance(text, str) or not text.strip() or label not in ("0", "1", "2"):
                raise ValueError(f"Validation có text/label rỗng hoặc sai ở dòng {line}; không bỏ qua dòng.")
            if source not in expected_sources:
                raise ValueError(f"Source không hợp lệ tại dòng {line}: {source!r}")
            texts.append(text)
            labels.append(int(label))
            sources.append(source)
    if len(texts) != expected_rows or set(sources) != set(expected_sources):
        raise ValueError(f"Validation không khớp số mẫu/source: {len(texts)}, {dict(Counter(sources))}")
    return {"texts": texts, "labels": np.array(labels, dtype=np.int64), "sources": sources,
            "sha256": actual_hash, "source_counts": dict(Counter(sources))}


def collect_fingerprints() -> dict:
    files = [
        VALIDATION_PATH, BACKEND / "data" / "validation_provenance.json",
        BACKEND / "models" / "svm" / "svm_pipeline.joblib",
        BACKEND / "src" / "real_svm.py", BACKEND / "src" / "real_transformer.py",
        BACKEND / "src" / "preprocess.py", BACKEND / "src" / "ensemble.py",
        BACKEND / "src" / "alpha_tuning.py", Path(__file__).resolve(),
        BACKEND / "requirements.txt",
    ]
    files.extend(BACKEND / "models" / "transformer" / name for name in (
        "best_transformer.pth", "model.py", "tokenizer.py", "predict_transformer.py",
        "preprocess.py", "vocab.json", "config.json", "label_map.json",
    ))
    return {path.relative_to(BACKEND).as_posix(): sha256_file(path) for path in files}


def _progress(name):
    def report(done, total):
        if done % 512 == 0 or done == total:
            print(f"{name}: {done}/{total}", flush=True)
    return report


def _write_plot(scores, selected_alpha: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = next(row for row in scores if row["alpha"] == selected_alpha)
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.plot([row["alpha"] for row in scores], [row["macro_f1"] for row in scores], "o-", label="Validation Macro-F1")
    axis.scatter([selected_alpha], [selected["macro_f1"]], s=180, marker="*", color="#b42318", zorder=4,
                 label=f"Selected alpha={selected_alpha:.1f}; F1={selected['macro_f1']:.6f}")
    axis.set(xlabel="Alpha (SVM weight)", ylabel="Validation Macro-F1",
             title="Validation alpha sweep | 8,238 samples", xticks=list(DEFAULT_ALPHAS))
    axis.grid(alpha=.25)
    axis.legend(loc="best")
    figure.savefig(OUTPUT_DIR / "alpha_vs_macro_f1.png", dpi=180)
    plt.close(figure)


def main() -> None:
    # Cố định Validation, không có tùy chọn CLI chuyển sang Test.
    config = json.loads((BACKEND / "models" / "transformer" / "config.json").read_text(encoding="utf-8"))
    expected_sha = config["dataset_sha256"]["val"]
    data = load_validation(VALIDATION_PATH, expected_sha)
    provenance = json.loads((BACKEND / "data" / "validation_provenance.json").read_text(encoding="utf-8"))
    hash_fields = ("sha256", "svm_handoff_validation_sha256", "transformer_archive_validation_sha256",
                   "transformer_training_config_validation_sha256")
    if provenance.get("preprocessed") is not True or any(provenance.get(key) != expected_sha for key in hash_fields):
        raise ValueError("Validation provenance không khớp; dừng trước inference.")
    fingerprints = collect_fingerprints()
    print(f"Validation verified: {len(data['texts'])} rows, sources={data['source_counts']}", flush=True)
    print(f"SHA-256: {expected_sha}; cleaned inference only, no Test", flush=True)

    from .src import real_svm, real_transformer
    svm_rows = real_svm.predict_svm_cleaned_batch(data["texts"], progress=_progress("SVM"))
    transformer_rows = real_transformer.predict_transformer_cleaned_batch(
        data["texts"], batch_size=64, device="cpu", progress=_progress("Transformer"))
    svm = np.array([[row[label] for label in LABELS] for row in svm_rows], dtype=float)
    transformer = np.array([[row[label] for label in LABELS] for row in transformer_rows], dtype=float)
    truth = data["labels"]
    svm_baseline = evaluate_probabilities(truth, svm)
    transformer_baseline = evaluate_probabilities(truth, transformer)
    tuning = tune_alpha(truth, svm, transformer)
    zero = compute_ensemble_probabilities(svm, transformer, 0)
    one = compute_ensemble_probabilities(svm, transformer, 1)
    if not np.array_equal(zero.argmax(1), transformer.argmax(1)) or not np.array_equal(one.argmax(1), svm.argmax(1)):
        raise ValueError("Alpha endpoint labels không khớp baseline; dừng.")
    endpoint_scores = {row["alpha"]: row for row in tuning["scores"]}
    if endpoint_scores[0.0]["macro_f1"] != transformer_baseline["macro_f1"] or endpoint_scores[1.0]["macro_f1"] != svm_baseline["macro_f1"]:
        raise ValueError("Alpha endpoint Macro-F1 không khớp baseline; dừng.")
    selected = compute_ensemble_probabilities(svm, transformer, tuning["selected_alpha"])
    source_values = np.array(data["sources"])
    by_source = []
    for source in SOURCES:
        mask = source_values == source
        by_source.append({"source": source, "n": int(mask.sum()),
                          "svm_macro_f1": evaluate_probabilities(truth[mask], svm[mask])["macro_f1"],
                          "transformer_macro_f1": evaluate_probabilities(truth[mask], transformer[mask])["macro_f1"],
                          "selected_alpha_macro_f1": evaluate_probabilities(truth[mask], selected[mask])["macro_f1"]})
    if collect_fingerprints() != fingerprints:
        raise ValueError("Input/model/code thay đổi trong lúc thực nghiệm; không xuất kết quả.")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "torch", "joblib", "matplotlib")}
    selection = {
        "selection_dataset": "validation", "primary_metric": "macro_f1",
        "selected_alpha": tuning["selected_alpha"],
        "selected_validation_macro_f1": tuning["selected_validation_macro_f1"],
        "svm_validation_macro_f1": svm_baseline["macro_f1"],
        "transformer_validation_macro_f1": transformer_baseline["macro_f1"],
        "ensemble_validation_macro_f1": tuning["selected_validation_macro_f1"],
        "difference_vs_svm": tuning["selected_validation_macro_f1"] - svm_baseline["macro_f1"],
        "difference_vs_transformer": tuning["selected_validation_macro_f1"] - transformer_baseline["macro_f1"],
        "alpha_grid": list(DEFAULT_ALPHAS), "tied_alphas": tuning["tied_alphas"],
        "tie_break_rule": tuning["tie_break_rule"], "sample_count": len(truth),
        "source_counts": data["source_counts"], "label_counts": {str(k): int(v) for k, v in Counter(truth.tolist()).items()},
        "by_source": by_source, "dataset_sha256": data["sha256"], "dataset_provenance": provenance,
        "fingerprints": fingerprints, "label_mapping": LABEL_TO_ID, "probability_columns": LABELS,
        "svm_classes": [int(value) for value in real_svm.load_model().classes_],
        "transformer_logits_labels": list(real_transformer.load_model().labels),
        "preprocessing": "Already-cleaned Validation; no clean_text calls in either inference path.",
        "inference_device": "cpu", "transformer_batch_size": 64,
        "python_version": sys.version, "package_versions": versions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probability_cache_policy": "Saved for audit only; each run recomputes fresh predictions. No cache is read.",
        "test_dataset_used": False, "pipeline_default_alpha_changed": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "alpha_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["alpha", "accuracy", "macro_precision", "macro_recall", "macro_f1"])
        writer.writeheader()
        writer.writerows(tuning["scores"])
    np.savez_compressed(OUTPUT_DIR / "val_probabilities.npz", svm_probs=svm, transformer_probs=transformer,
                        y_true=truth, sources=source_values, fingerprint_json=np.array(json.dumps(fingerprints)))
    _write_plot(tuning["scores"], tuning["selected_alpha"])
    (OUTPUT_DIR / "alpha_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nalpha  accuracy   precision  recall     macro_f1")
    for row in tuning["scores"]:
        print(f"{row['alpha']:.1f}    {row['accuracy']:.8f} {row['macro_precision']:.8f} {row['macro_recall']:.8f} {row['macro_f1']:.8f}")
    print(f"\nSelected alpha={tuning['selected_alpha']}; Validation Macro-F1={tuning['selected_validation_macro_f1']:.10f}")
    print(f"SVM baseline={svm_baseline['macro_f1']:.10f}; Transformer baseline={transformer_baseline['macro_f1']:.10f}")
    print(f"Difference vs SVM={selection['difference_vs_svm']:+.10f}; vs Transformer={selection['difference_vs_transformer']:+.10f}")
    print("By source:", json.dumps(by_source, ensure_ascii=False))
    if tuning["selected_alpha"] in (0.0, 1.0):
        print("Selected endpoint is equivalent to a single model; do not claim an ensemble improvement over that model.")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Validation only; no BEST_METHOD/aggregation selection and no pipeline default changes.")


if __name__ == "__main__":
    main()
