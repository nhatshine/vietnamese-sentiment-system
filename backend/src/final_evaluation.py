"""Pure metrics and input boundaries for frozen final evaluation; no model loading."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .alpha_tuning import _alpha, _labels, _probabilities, compute_ensemble_probabilities
from .config import ID_TO_LABEL, LABELS, LABEL_TO_ID
from .error_analysis import MODEL_NAMES, SOURCE_NAMES, model_metrics


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_selection(path) -> dict:
    """Derive selection solely from saved Validation scores, before Test access."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (raw.get("selection_dataset") != "validation" or raw.get("test_dataset_used") is not False
            or raw.get("primary_metric") != "macro_f1"):
        raise ValueError("Selection phải dựa trên Validation Macro-F1, chưa dùng Test.")
    if raw.get("probability_columns") != LABELS or raw.get("label_mapping") != LABEL_TO_ID:
        raise ValueError("Label mapping/order không khớp cấu hình đã kiểm thử.")
    alpha = _alpha(raw["selected_alpha"])
    scores = {name: raw[f"{name}_validation_macro_f1"] for name in MODEL_NAMES}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1
           for v in scores.values()):
        raise ValueError("Validation metrics không hợp lệ.")
    if scores["ensemble"] != raw["selected_validation_macro_f1"]:
        raise ValueError("Selected alpha score không khớp Ensemble Validation score.")
    winners = [name for name, value in scores.items() if value == max(scores.values())]
    if len(winners) != 1:
        raise ValueError("Validation có hòa; cần policy đã khóa trước Test, không tự chọn theo Test.")
    return {"selected_alpha": alpha, "selected_method": winners[0], "selection_dataset": "validation",
            "validation_scores": scores, "selection_sha256": file_sha256(path),
            "label_mapping": LABEL_TO_ID.copy(), "probability_columns": LABELS.copy(),
            "step9_fingerprints": raw["fingerprints"]}


def load_test_dataset(path, expected_sha256, *, expected_rows=8238) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu Test CSV: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError("Test SHA-256 không khớp provenance/training config.")
    texts, labels, sources = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "label", "source"}.issubset(reader.fieldnames or []):
            raise ValueError("Test phải có text, label, source.")
        for line, row in enumerate(reader, 2):
            text, label, source = row.get("text"), row.get("label"), row.get("source")
            if not isinstance(text, str) or not text.strip() or label not in ("0", "1", "2") or source not in SOURCE_NAMES:
                raise ValueError(f"Test có null/giá trị sai tại dòng {line}; không loại mẫu.")
            texts.append(text); labels.append(int(label)); sources.append(source)
    if len(texts) != expected_rows or set(sources) != set(SOURCE_NAMES):
        raise ValueError(f"Test sai số mẫu/source: N={len(texts)}; không shuffle/bỏ mẫu.")
    return {"texts": texts, "labels": np.array(labels, dtype=np.int64), "sources": sources,
            "sha256": actual, "sample_count": len(texts), "source_counts": dict(Counter(sources)),
            "label_counts": {label: labels.count(LABEL_TO_ID[label]) for label in LABELS}}


def ordered_row_ids(data) -> list[str]:
    count = len(data["texts"])
    labels = _labels(data["labels"], count)
    if len(data["sources"]) != count:
        raise ValueError("Source/text/label alignment mismatch.")
    return [hashlib.sha256(json.dumps([i, text, int(labels[i]), data["sources"][i]],
                                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            for i, text in enumerate(data["texts"])]


def _metrics(truth, probabilities):
    result = model_metrics(truth, probabilities)
    predictions = np.array([LABEL_TO_ID[label] for label in LABELS])[probabilities.argmax(axis=1)]
    result["weighted_f1"] = float(f1_score(truth, predictions, labels=[0, 1, 2], average="weighted", zero_division=0))
    return result


def evaluate_test(y_true, svm_probs, transformer_probs, sources, frozen) -> tuple[dict, dict]:
    svm = _probabilities(svm_probs, "svm_probs")
    truth = _labels(y_true, len(svm))
    ensemble = compute_ensemble_probabilities(svm, transformer_probs, frozen["selected_alpha"])
    if isinstance(sources, str) or len(sources) != len(truth) or any(s not in SOURCE_NAMES for s in sources):
        raise ValueError("Sources không khớp sample alignment hoặc mapping.")
    matrices = {"svm": svm, "transformer": _probabilities(transformer_probs, "transformer_probs"), "ensemble": ensemble}
    models = {name: _metrics(truth, values) for name, values in matrices.items()}
    rows = []
    for source in SOURCE_NAMES:
        mask = np.array(sources) == source
        rows.append({"source": source, "n": int(mask.sum()),
                     "label_counts": {label: int((truth[mask] == LABEL_TO_ID[label]).sum()) for label in LABELS},
                     "models": {name: _metrics(truth[mask], values[mask]) for name, values in matrices.items()} if mask.any() else {}})
    comparison = [{"model": name, "validation_macro_f1": frozen["validation_scores"][name],
                   "test_macro_f1": models[name]["macro_f1"],
                   "test_minus_validation": models[name]["macro_f1"] - frozen["validation_scores"][name]} for name in MODEL_NAMES]
    return {"sample_count": len(truth), "selected_alpha": frozen["selected_alpha"],
            "selected_method": frozen["selected_method"], "selection_dataset": "validation",
            "label_order": LABELS.copy(), "models": models, "sources": rows,
            "ensemble_deltas": {f"vs_{name}": models["ensemble"]["macro_f1"] - models[name]["macro_f1"] for name in ("svm", "transformer")},
            "validation_vs_test": comparison}, matrices


def prediction_rows(data, matrices) -> list[dict]:
    truth = _labels(data["labels"], len(data["texts"]))
    if len(data["sources"]) != len(truth):
        raise ValueError("Prediction CSV source alignment mismatch.")
    values = {name: _probabilities(matrices[name], name) for name in MODEL_NAMES}
    if any(len(p) != len(truth) for p in values.values()):
        raise ValueError("Prediction CSV probability alignment mismatch.")
    rows = []
    for i, text in enumerate(data["texts"]):
        row = {"sample_index": i, "source": data["sources"][i], "text": text, "true_label": ID_TO_LABEL[int(truth[i])]}
        for name, probs in values.items():
            row[f"{name}_pred"] = LABELS[int(probs[i].argmax())]
            row[f"{name}_confidence"] = float(probs[i].max())
            row.update({f"{name}_p_{label}": float(probs[i, j]) for j, label in enumerate(LABELS)})
        rows.append(row)
    return rows
