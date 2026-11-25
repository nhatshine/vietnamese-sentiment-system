"""Validation-only research helpers. Not connected to the production selector."""

import numpy as np

from .alpha_tuning import _labels, _probabilities, compute_ensemble_probabilities
from .config import LABELS
from .error_analysis import SOURCE_NAMES, confidence_analysis, model_metrics


FIXED_ALPHA = 0.8
MODEL_TIE_RULE = "Exact equality of maximum probabilities: prefer SVM; no rounding or isclose."
CLASS_TIE_RULE = "Argmax ties: first class in negative, neutral, positive order."


def select_max_confidence(svm_probs, transformer_probs, *, label_order=LABELS) -> dict:
    """Select a whole probability row. Class and sample orders must match."""
    if list(label_order) != LABELS:
        raise ValueError("Probability columns must be negative, neutral, positive in that order.")
    svm = _probabilities(svm_probs, "svm_probs")
    transformer = _probabilities(transformer_probs, "transformer_probs")
    if svm.shape != transformer.shape:
        raise ValueError("Both models must use the same samples and label columns.")
    svm_confidence, transformer_confidence = svm.max(axis=1), transformer.max(axis=1)
    choose_svm = svm_confidence >= transformer_confidence
    return {
        "probabilities": np.where(choose_svm[:, None], svm, transformer),
        "selected_models": np.where(choose_svm, "svm", "transformer"),
        "svm_confidence": svm_confidence,
        "transformer_confidence": transformer_confidence,
        "ties": svm_confidence == transformer_confidence,
    }


def compare_methods(y_true, svm_probs, transformer_probs, sources) -> dict:
    """Two rules applied to the same rows, without fitting or alpha tuning."""
    chosen = select_max_confidence(svm_probs, transformer_probs)
    svm = _probabilities(svm_probs, "svm_probs")
    transformer = _probabilities(transformer_probs, "transformer_probs")
    truth = _labels(y_true, len(svm))
    source_values = np.asarray(sources)
    if source_values.ndim != 1 or len(source_values) != len(truth):
        raise ValueError("Source order/length must match the probability rows.")
    if not np.isin(source_values, SOURCE_NAMES).all():
        raise ValueError("Unknown source in Validation.")
    matrices = {"max_confidence": chosen["probabilities"],
                "weighted_ensemble": compute_ensemble_probabilities(svm, transformer, FIXED_ALPHA)}
    predictions = {name: probs.argmax(axis=1) for name, probs in matrices.items()}
    mc_correct = predictions["max_confidence"] == truth
    we_correct = predictions["weighted_ensemble"] == truth
    disagree = predictions["max_confidence"] != predictions["weighted_ensemble"]
    groups = {"both_correct": int((mc_correct & we_correct).sum()),
              "max_confidence_only": int((mc_correct & ~we_correct).sum()),
              "weighted_ensemble_only": int((~mc_correct & we_correct).sum()),
              "both_wrong": int((~mc_correct & ~we_correct).sum())}
    by_source = []
    for source in SOURCE_NAMES:
        mask = source_values == source
        if mask.any():
            by_source.append({"source": source, "sample_count": int(mask.sum()),
                              "models": {name: model_metrics(truth[mask], probs[mask])
                                         for name, probs in matrices.items()}})
    selected_accuracy = {}
    for name, probs in (("svm", svm), ("transformer", transformer)):
        mask = chosen["selected_models"] == name
        selected_accuracy[name] = {"count": int(mask.sum()),
                                  "accuracy": float((probs[mask].argmax(1) == truth[mask]).mean()) if mask.any() else None}
    models = {name: model_metrics(truth, probs) for name, probs in matrices.items()}
    delta = models["weighted_ensemble"]["macro_f1"] - models["max_confidence"]["macro_f1"]
    return {
        "sample_count": len(truth), "alpha": FIXED_ALPHA, "label_order": list(LABELS),
        "model_tie_rule": MODEL_TIE_RULE, "class_tie_rule": CLASS_TIE_RULE,
        "models": models, "by_source": by_source,
        "baselines": {"svm": model_metrics(truth, svm), "transformer": model_metrics(truth, transformer)},
        "selection_counts": {"svm": int((chosen["selected_models"] == "svm").sum()),
                             "transformer": int((chosen["selected_models"] == "transformer").sum()),
                             "exact_ties": int(chosen["ties"].sum())},
        "selected_model_accuracy": selected_accuracy,
        "correctness_groups": groups,
        "disagreements": {"count": int(disagree.sum()), "indices": np.flatnonzero(disagree).tolist(),
                          "both_wrong_count": int((disagree & ~mc_correct & ~we_correct).sum())},
        "macro_f1_delta_weighted_minus_max": float(delta),
        "winner_on_validation": "weighted_ensemble" if delta > 0 else "max_confidence" if delta < 0 else "tie",
        "confidence_diagnostics": {name: confidence_analysis(truth, probs) for name, probs in
                                   {"svm": svm, "transformer": transformer, **matrices}.items()},
    }
