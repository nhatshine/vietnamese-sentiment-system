"""Diagnostic analysis từ probability đã có; không import hoặc load model."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .alpha_tuning import _labels, _probabilities, compute_ensemble_probabilities, evaluate_probabilities
from .config import ID_TO_LABEL, LABELS, LABEL_TO_ID


MODEL_NAMES = ("svm", "transformer", "ensemble")
SOURCE_NAMES = ("hieu_sa", "neu_esc", "vsfc")
LENGTH_GROUPS = (("1-10", 1, 10), ("11-20", 11, 20), ("21-50", 21, 50),
                 ("51-128", 51, 128), (">128", 129, None))


def _class_ids():
    return np.array([LABEL_TO_ID[label] for label in LABELS])


def _prediction(probabilities):
    return _class_ids()[probabilities.argmax(axis=1)]


def _pair(y_true, svm_probs, transformer_probs):
    svm = _probabilities(svm_probs, "svm_probs")
    transformer = _probabilities(transformer_probs, "transformer_probs")
    if svm.shape != transformer.shape:
        raise ValueError("SVM và Transformer phải có cùng số mẫu/cột theo LABELS.")
    return _labels(y_true, len(svm)), svm, transformer


def _triple(y_true, svm_probs, transformer_probs, alpha):
    truth, svm, transformer = _pair(y_true, svm_probs, transformer_probs)
    return truth, {"svm": svm, "transformer": transformer,
                   "ensemble": compute_ensemble_probabilities(svm, transformer, alpha)}


def _strings(values, count, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} phải là sequence các chuỗi.")
    if len(values) != count:
        raise ValueError(f"{name} không cùng thứ tự/số mẫu với probability.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} không được chứa giá trị rỗng hoặc non-string.")
    return list(values)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def complementary_errors(y_true, svm_probs, transformer_probs) -> dict:
    """Bốn nhóm rời nhau, rescue rates và oracle chọn giữa hai hard labels."""
    truth, svm, transformer = _pair(y_true, svm_probs, transformer_probs)
    svm_pred, transformer_pred = _prediction(svm), _prediction(transformer)
    cs, ct = svm_pred == truth, transformer_pred == truth
    masks = {"A": cs & ct, "B": cs & ~ct, "C": ~cs & ct, "D": ~cs & ~ct}
    names = {"A": "both_correct", "B": "svm_correct_transformer_wrong",
             "C": "svm_wrong_transformer_correct", "D": "both_wrong"}
    groups = {key: {"name": names[key], "count": int(mask.sum()), "rate": float(mask.mean())}
              for key, mask in masks.items()}
    svm_wrong, transformer_wrong = int((~cs).sum()), int((~ct).sum())
    oracle = np.where(cs | ct, truth, svm_pred)
    return {
        "groups": groups, "svm_correct_count": int(cs.sum()),
        "transformer_correct_count": int(ct.sum()), "svm_wrong_count": svm_wrong,
        "transformer_wrong_count": transformer_wrong,
        "transformer_correct_when_svm_wrong": groups["C"]["count"],
        "svm_correct_when_transformer_wrong": groups["B"]["count"],
        "both_wrong_count": groups["D"]["count"],
        "transformer_rescue_rate": _ratio(groups["C"]["count"], svm_wrong),
        "svm_rescue_rate": _ratio(groups["B"]["count"], transformer_wrong),
        "oracle_accuracy": float((cs | ct).mean()),
        "oracle_macro_f1": float(f1_score(truth, oracle, labels=_class_ids(), average="macro", zero_division=0)),
        "oracle_rule": "Use ground truth if either model is correct; otherwise use SVM prediction. Diagnostic only, not deployable.",
    }


def model_metrics(y_true, probabilities) -> dict:
    values = _probabilities(probabilities, "probabilities")
    truth = _labels(y_true, len(values))
    predicted = _prediction(values)
    matrix = confusion_matrix(truth, predicted, labels=_class_ids())
    support = matrix.sum(axis=1)
    normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float), where=support[:, None] != 0)
    precision, recall, f1, counts = precision_recall_fscore_support(
        truth, predicted, labels=_class_ids(), zero_division=0)
    per_class = [{"label": label, "precision": float(precision[index]), "recall": float(recall[index]),
                  "f1": float(f1[index]), "support": int(counts[index])} for index, label in enumerate(LABELS)]
    wrong = int((truth != predicted).sum())
    pairs = [{"true_label": LABELS[i], "predicted_label": LABELS[j], "count": int(matrix[i, j]),
              "fraction_of_model_errors": _ratio(int(matrix[i, j]), wrong)}
             for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j]
    top = max(pairs, key=lambda row: row["count"]) if wrong else None
    return {**evaluate_probabilities(truth, values), "correct_count": len(truth) - wrong, "wrong_count": wrong,
            "confusion_matrix": matrix.tolist(), "confusion_matrix_row_normalized": normalized.tolist(),
            "per_class": per_class, "confusion_pairs": pairs, "top_confusion": top}


def confidence_analysis(y_true, probabilities, bins: int = 10) -> dict:
    """ECE equal-width bins; thresholds có mẫu số là số prediction sai."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins phải là số nguyên dương.")
    values = _probabilities(probabilities, "probabilities")
    truth = _labels(y_true, len(values))
    confidence = values.max(axis=1)
    correct = _prediction(values) == truth
    wrong = ~correct
    wrong_count = int(wrong.sum())
    thresholds = (.8, .9, .95)
    counts = {str(threshold): int((wrong & (confidence >= threshold)).sum()) for threshold in thresholds}
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = bin_ids == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(truth) * abs(accuracy - mean)
        rows.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                     "upper_inclusive": index == bins - 1, "count": count,
                     "accuracy": accuracy, "mean_confidence": mean})
    return {"correct_count": int(correct.sum()), "wrong_count": wrong_count,
            "mean_confidence": float(confidence.mean()), "accuracy": float(correct.mean()),
            "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
            "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
            "mean_confidence_wrong": float(confidence[wrong].mean()) if wrong_count else None,
            "median_confidence_wrong": float(np.median(confidence[wrong])) if wrong_count else None,
            "wrong_high_confidence_counts": counts,
            "wrong_high_confidence_rates": {key: _ratio(count, wrong_count) for key, count in counts.items()},
            "threshold_rate_denominator": "wrong_count", "ece": float(ece), "bins": rows}


def ensemble_effects(y_true, svm_probs, transformer_probs, alpha) -> dict:
    truth, matrices = _triple(y_true, svm_probs, transformer_probs, alpha)
    correct = {model: _prediction(values) == truth for model, values in matrices.items()}
    cs, ct, ce = (correct[model] for model in MODEL_NAMES)
    metrics = {model: model_metrics(truth, values) for model, values in matrices.items()}
    result = {
        "svm_wrong_to_ensemble_correct": int((~cs & ce).sum()),
        "svm_correct_to_ensemble_wrong": int((cs & ~ce).sum()),
        "transformer_wrong_to_ensemble_correct": int((~ct & ce).sum()),
        "transformer_correct_to_ensemble_wrong": int((ct & ~ce).sum()),
        "both_wrong_to_ensemble_correct": int((~cs & ~ct & ce).sum()),
        "both_correct_to_ensemble_wrong": int((cs & ct & ~ce).sum()),
        "ensemble_correct_count": int(ce.sum()), "net_correct_vs_svm": int(ce.sum() - cs.sum()),
        "accuracy_difference_vs_svm": metrics["ensemble"]["accuracy"] - metrics["svm"]["accuracy"],
        "macro_f1_difference_vs_svm": metrics["ensemble"]["macro_f1"] - metrics["svm"]["macro_f1"],
        "class_f1_changes": [
            {"label": label, "svm_f1": metrics["svm"]["per_class"][i]["f1"],
             "transformer_f1": metrics["transformer"]["per_class"][i]["f1"],
             "ensemble_f1": metrics["ensemble"]["per_class"][i]["f1"],
             "difference_vs_svm": metrics["ensemble"]["per_class"][i]["f1"] - metrics["svm"]["per_class"][i]["f1"]}
            for i, label in enumerate(LABELS)
        ],
    }
    assert result["net_correct_vs_svm"] == result["svm_wrong_to_ensemble_correct"] - result["svm_correct_to_ensemble_wrong"]
    return result


def _subset_metrics(truth, matrices, mask):
    count = int(mask.sum())
    result = {"n": count}
    for model, values in matrices.items():
        metrics = model_metrics(truth[mask], values[mask]) if count else None
        for key in ("accuracy", "macro_f1"):
            result[f"{model}_{key}"] = metrics[key] if count else None
        result[f"{model}_error_rate"] = metrics["wrong_count"] / count if count else None
        for i, label in enumerate(LABELS):
            result[f"{model}_f1_{label}"] = metrics["per_class"][i]["f1"] if count else None
    return result


def source_analysis(y_true, svm_probs, transformer_probs, sources, alpha) -> list[dict]:
    truth, matrices = _triple(y_true, svm_probs, transformer_probs, alpha)
    source_values = np.array(_strings(sources, len(truth), "sources"))
    if not set(source_values).issubset(SOURCE_NAMES):
        raise ValueError("sources phải thuộc hieu_sa, neu_esc, vsfc.")
    rows = []
    for source in SOURCE_NAMES:
        mask = source_values == source
        row = {"source": source, **_subset_metrics(truth, matrices, mask)}
        row.update({f"support_{label}": int((truth[mask] == LABEL_TO_ID[label]).sum()) for label in LABELS})
        if mask.any():
            complement = complementary_errors(truth[mask], matrices["svm"][mask], matrices["transformer"][mask])
            effects = ensemble_effects(truth[mask], matrices["svm"][mask], matrices["transformer"][mask], alpha)
            row.update({key: complement[key] for key in ("svm_correct_count", "transformer_correct_count", "svm_wrong_count",
                        "transformer_wrong_count", "both_wrong_count", "transformer_rescue_rate", "svm_rescue_rate")})
            row["svm_wrong_transformer_correct"] = complement["groups"]["C"]["count"]
            row["svm_correct_transformer_wrong"] = complement["groups"]["B"]["count"]
            row["ensemble_rescues_svm"] = effects["svm_wrong_to_ensemble_correct"]
            row["ensemble_hurts_svm"] = effects["svm_correct_to_ensemble_wrong"]
        rows.append(row)
    return rows


def length_analysis(y_true, svm_probs, transformer_probs, texts, alpha, max_content_tokens: int = 127) -> list[dict]:
    truth, matrices = _triple(y_true, svm_probs, transformer_probs, alpha)
    lengths = np.array([len(text.split()) for text in _strings(texts, len(truth), "texts")])
    rows = []
    for group, lower, upper in LENGTH_GROUPS:
        mask = lengths >= lower
        if upper is not None:
            mask = mask & (lengths <= upper)
        present = [label for label in LABELS if (truth[mask] == LABEL_TO_ID[label]).any()]
        count = int(mask.sum())
        rows.append({"length_group": group, **_subset_metrics(truth, matrices, mask),
                     "class_counts": {label: int((truth[mask] == LABEL_TO_ID[label]).sum()) for label in LABELS},
                     "present_labels": present, "missing_labels": [label for label in LABELS if label not in present],
                     "small_sample": count < 30,
                     "truncation_exposed_count": int((lengths[mask] > max_content_tokens).sum())})
    return rows


def extract_error_examples(y_true, svm_probs, transformer_probs, texts, sources, alpha, limit_per_group: int = 10) -> list[dict]:
    """Chọn theo confidence giảm dần rồi index tăng; text giữ nguyên, không gán nguyên nhân."""
    if isinstance(limit_per_group, bool) or not isinstance(limit_per_group, int) or limit_per_group < 1:
        raise ValueError("limit_per_group phải là số nguyên dương.")
    truth, matrices = _triple(y_true, svm_probs, transformer_probs, alpha)
    texts, sources = _strings(texts, len(truth), "texts"), _strings(sources, len(truth), "sources")
    predicted = {model: _prediction(values) for model, values in matrices.items()}
    confidence = {model: values.max(axis=1) for model, values in matrices.items()}
    cs, ct, ce = (predicted[model] == truth for model in MODEL_NAMES)
    groups = np.where(cs & ct, "both_correct", np.where(cs, "svm_correct_transformer_wrong",
                     np.where(ct, "svm_wrong_transformer_correct", "both_wrong")))
    candidates = {
        "svm_wrong_transformer_correct": (~cs & ct, confidence["svm"]),
        "svm_correct_transformer_wrong": (cs & ~ct, confidence["transformer"]),
        "both_wrong": (~cs & ~ct, np.maximum(confidence["svm"], confidence["transformer"])),
        "ensemble_rescues_svm": (~cs & ce, confidence["ensemble"]),
        "ensemble_hurts_svm": (cs & ~ce, confidence["ensemble"]),
        "both_wrong_ensemble_correct": (~cs & ~ct & ce, confidence["ensemble"]),
    }
    for model in MODEL_NAMES:
        candidates[f"high_confidence_wrong_{model}"] = ((predicted[model] != truth) & (confidence[model] >= .9), confidence[model])
    reasons = {}
    for reason, (mask, priority) in candidates.items():
        indices = sorted(np.flatnonzero(mask), key=lambda index: (-float(priority[index]), int(index)))[:limit_per_group]
        for index in indices:
            reasons.setdefault(int(index), []).append(reason)
    rows = []
    for index in sorted(reasons):
        effect = "ensemble_rescues_svm" if not cs[index] and ce[index] else (
            "ensemble_hurts_svm" if cs[index] and not ce[index] else "unchanged_vs_svm")
        row = {"sample_index": index, "source": sources[index], "text": texts[index],
               "true_label": ID_TO_LABEL[int(truth[index])], "error_group": str(groups[index]),
               "ensemble_effect": effect, "selection_reasons": ";".join(reasons[index])}
        for model in MODEL_NAMES:
            row[f"{model}_label"] = ID_TO_LABEL[int(predicted[model][index])]
            row[f"{model}_confidence"] = float(confidence[model][index])
            for column, label in enumerate(LABELS):
                row[f"{model}_p_{label}"] = float(matrices[model][index, column])
        rows.append(row)
    return rows


def analyze_validation_errors(y_true, svm_probs, transformer_probs, texts, sources, alpha, max_content_tokens=127) -> dict:
    """Trả diagnostic summary; không inference, tuning hoặc model selection."""
    truth, matrices = _triple(y_true, svm_probs, transformer_probs, alpha)
    return {
        "sample_count": len(truth), "alpha": float(alpha), "label_order": LABELS.copy(),
        "models": {model: model_metrics(truth, values) for model, values in matrices.items()},
        "complementarity": complementary_errors(truth, matrices["svm"], matrices["transformer"]),
        "ensemble_effects": ensemble_effects(truth, matrices["svm"], matrices["transformer"], alpha),
        "sources": source_analysis(truth, matrices["svm"], matrices["transformer"], sources, alpha),
        "confidence": {model: confidence_analysis(truth, values) for model, values in matrices.items()},
        "length_groups": length_analysis(truth, matrices["svm"], matrices["transformer"], texts, alpha, max_content_tokens),
        "max_content_tokens": max_content_tokens,
        "examples": extract_error_examples(truth, matrices["svm"], matrices["transformer"], texts, sources, alpha),
        "oracle_is_deployable": False,
    }
