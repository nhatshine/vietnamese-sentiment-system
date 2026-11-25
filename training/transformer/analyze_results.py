"""Produce report-ready Transformer and source-only evaluation tables (read-only).

Requires completed training predictions. The source-only baseline uses TRAIN labels
exclusively to learn a majority label for each data source. Test is evaluated
only after model selection; this script never changes model weights or alpha.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = (0, 1, 2)
NAMES = ("negative", "neutral", "positive")
BASELINE_12 = {"val_macro_f1": 0.792806, "test_macro_f1": 0.794304,
               "note": "Mốc lấy từ báo cáo 12 epoch, chưa kiểm chứng lại từ notebook hiện tại"}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def score(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))}


def safe_float(x):
    return float(x)


def ece_10_bins(y_true, y_pred, confidences):
    # Equal-width bins: (0,0.1],...,(0.9,1], leftmost includes 0.
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    conf = np.asarray(confidences, dtype=float)
    bins = np.minimum(np.floor(conf * 10).astype(int), 9)
    out = 0.0
    details = []
    for b in range(10):
        idx = (bins == b)
        count = int(idx.sum())
        if count:
            acc = float(np.mean(y_pred[idx] == y_true[idx]))
            avg_conf = float(conf[idx].mean())
            out += (count / len(y_true)) * abs(acc - avg_conf)
        else:
            acc, avg_conf = 0.0, 0.0
        details.append({"bin": b + 1, "count": count,
                        "bin_lower": b / 10, "bin_upper": (b + 1) / 10,
                        "accuracy": acc, "avg_confidence": avg_conf})
    return float(out), details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    metric_dir = a.out_dir / "metrics"
    summary = json.loads((metric_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((a.out_dir / "config.json").read_text(encoding="utf-8"))

    # TRAIN-only source-majority baseline (tie-break by smallest label ID).
    train = read_csv(a.data_dir / "train.csv")
    source_counts = defaultdict(Counter)
    for r in train:
        if not r.get("source"):
            raise ValueError("train.csv chứa source rỗng; không tính được baseline Source-only chính xác")
        source_counts[r["source"]][int(r["label"])] += 1
    majority = {src: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                for src, c in source_counts.items()}
    source_majority_rows = [{"source": s, "n_train": sum(c.values()),
                             "majority_label": majority[s], "majority_name": NAMES[majority[s]],
                             "negative_train": c[0], "neutral_train": c[1], "positive_train": c[2]}
                            for s, c in sorted(source_counts.items())]
    write_csv(metric_dir / "source_only_train_mapping.csv", source_majority_rows,
              ["source", "n_train", "majority_label", "majority_name", "negative_train", "neutral_train", "positive_train"])

    per_source, per_class, baselines, ece_rows, errors = [], [], [], [], []
    for split in ("val", "test"):
        orig_rows = read_csv(a.data_dir / f"{split}.csv")
        preds = read_csv(metric_dir / f"{split}_predictions.csv")
        if len(orig_rows) != len(preds):
            raise ValueError(f"{split} số mẫu dự đoán != dữ liệu gốc")
        yt, yp, source_list, confidence = [], [], [], []
        for i, (real, pred) in enumerate(zip(orig_rows, preds)):
            required_id = f"{split}:{i:06d}"
            if (pred.get("sample_id") != required_id
                    or int(pred["row_index"]) != i
                    or real["text"] != pred["text"]
                    or int(real["label"]) != int(pred["true_label"])
                    or real["source"] != pred["source"]):
                raise ValueError(f"{split} sample_id/text/label/source lệch tại hàng {i}; không ghép dự đoán")
            prob = [safe_float(pred[f"p_{s}"]) for s in NAMES]
            if not np.all(np.isfinite(prob)) or abs(sum(prob) - 1.0) > 1e-4:
                raise ValueError(f"{split} xác suất không hợp lệ tại hàng {i}: {prob}")
            true, prediction = int(pred["true_label"]), int(pred["pred_label"])
            if prediction != int(np.argmax(prob)):
                raise ValueError(f"{split} dự đoán không khớp argmax ở hàng {i}")
            yt.append(true); yp.append(prediction)
            source_list.append(pred["source"])
            confidence.append(max(prob))
            if split == "test" and true != prediction:
                errors.append({"sample_id": pred["sample_id"], "source": pred["source"],
                               "text": pred["text"], "true_label": true,
                               "pred_label": prediction, "confidence": max(prob)})
        check = score(yt, yp)
        if abs(check["macro_f1"] - summary["validation" if split == "val" else "test"]["macro_f1"]) > 1e-9:
            raise ValueError(f"{split} Macro-F1 dự đoán không khớp metrics.json")

        groups = {src: [j for j, x in enumerate(source_list) if x == src]
                  for src in sorted(set(source_list))}
        for src, inds in groups.items():
            y_true, y_pred = [yt[j] for j in inds], [yp[j] for j in inds]
            per_source.append({"split": split, "source": src, "n_samples": len(inds),
                               **score(y_true, y_pred)})
            if src not in majority:
                raise ValueError(f"Nguồn {src} có trong {split}, không có trong Train")
            baselines.append({"split": split, "source": src, "n_samples": len(inds),
                              "majority_label_from_train": majority[src],
                              **score(y_true, [majority[src]] * len(inds))})

        per_source.append({"split": split, "source": "ALL", "n_samples": len(yt), **score(yt, yp)})
        source_pred = [majority[src] for src in source_list]
        baselines.append({"split": split, "source": "ALL", "n_samples": len(yt),
                          "majority_label_from_train": "by_source", **score(yt, source_pred)})

        precision, recall, f1, support = precision_recall_fscore_support(
            yt, yp, labels=LABELS, zero_division=0)
        for lab, name in enumerate(NAMES):
            per_class.append({"split": split, "label": lab, "label_name": name,
                              "support": int(support[lab]), "precision": float(precision[lab]),
                              "recall": float(recall[lab]), "f1": float(f1[lab])})

        ece, ece_bins = ece_10_bins(yt, yp, confidence)
        ece_rows.append({"split": split, "n_samples": len(yt), "ece_10_bins": ece,
                         "mean_confidence": float(np.mean(confidence)),
                         "n_wrong_confidence_ge_0_9": sum(t != p and c >= 0.9 for t, p, c in zip(yt, yp, confidence))})
        write_csv(metric_dir / f"{split}_ece_bins.csv", ece_bins,
                  ["bin", "count", "bin_lower", "bin_upper", "accuracy", "avg_confidence"])

    write_csv(metric_dir / "transformer_by_source.csv", per_source,
              ["split", "source", "n_samples", "accuracy", "macro_f1"])
    write_csv(metric_dir / "transformer_by_class.csv", per_class,
              ["split", "label", "label_name", "support", "precision", "recall", "f1"])
    write_csv(metric_dir / "source_only_baseline.csv", baselines,
              ["split", "source", "n_samples", "majority_label_from_train", "accuracy", "macro_f1"])
    write_csv(metric_dir / "transformer_ece.csv", ece_rows,
              ["split", "n_samples", "ece_10_bins", "mean_confidence", "n_wrong_confidence_ge_0_9"])
    errors.sort(key=lambda r: r["confidence"], reverse=True)
    write_csv(metric_dir / "test_high_confidence_errors_top50.csv", errors[:50],
              ["sample_id", "source", "text", "true_label", "pred_label", "confidence"])

    current_val = summary["validation"]["macro_f1"]
    current_test = summary["test"]["macro_f1"]
    comparison = [
        {"experiment": "original_12ep_report", "best_epoch": 12,
         "validation_macro_f1": BASELINE_12["val_macro_f1"],
         "test_macro_f1": BASELINE_12["test_macro_f1"],
         "status": "reference_from_old_report"},
        {"experiment": "retrain_up_to_30ep", "best_epoch": summary["best_epoch"],
         "validation_macro_f1": current_val, "test_macro_f1": current_test,
         "status": "measured_from_new_predictions"}
    ]
    write_csv(metric_dir / "transformer_12_vs_30.csv", comparison,
              ["experiment", "best_epoch", "validation_macro_f1", "test_macro_f1", "status"])
    # No tuning based on Test comparison; historical test score is only descriptive.
    (metric_dir / "handover_manifest.json").write_text(json.dumps({
        "best_epoch": summary["best_epoch"], "stop_reason": summary["stop_reason"],
        "data_sha256": config["dataset_sha256"],
        "labels": dict(enumerate(NAMES)), "row_index": "0-based index in original split CSV",
        "sample_id": "split:zero_padded_row_index",
        "prediction_columns": ["sample_id", "row_index", "text", "true_label", "pred_label",
                               "p_negative", "p_neutral", "p_positive", "source"],
        "source_only_baseline": "majority class by source fitted on train.csv only",
        "note": "Only Validation selects checkpoints or ensemble weights; Test remains evaluation only"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== PHÂN TÍCH BỔ SUNG CHO BÁO CÁO ===")
    print("Best epoch:", summary["best_epoch"], "/ Last epoch:", summary["last_epoch_completed"],
          "/ Stop reason:", summary["stop_reason"])
    print("12 epoch (mốc báo cáo): Val F1", BASELINE_12["val_macro_f1"],
          "Test F1", BASELINE_12["test_macro_f1"])
    print("Train mới: Val F1", round(current_val, 6),
          "Test F1", round(current_test, 6))
    for row in per_source:
        print("Transformer", row["split"], row["source"],
              "N", row["n_samples"], "Macro-F1", round(row["macro_f1"], 6))
    for row in ece_rows:
        print("ECE", row["split"], "=", round(row["ece_10_bins"], 6))
    print("Các CSV báo cáo đã lưu vào:", metric_dir)


if __name__ == "__main__":
    main()
