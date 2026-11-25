"""Resolve deployment selection from frozen Validation artifacts, without inference."""

import csv
import hashlib
import json
import math
from pathlib import Path

from .alpha_tuning import TIE_BREAK_RULE, _alpha
from .config import LABELS, LABEL_TO_ID
from .result_selector import SUPPORTED_METHODS

BACKEND = Path(__file__).resolve().parents[1]
SELECTION_PATH = BACKEND / "outputs" / "alpha_tuning" / "alpha_selection.json"
SCORES_PATH = SELECTION_PATH.with_name("alpha_scores.csv")
FREEZE_PATH = BACKEND / "outputs" / "final_evaluation" / "evaluation_freeze.json"
DEPLOYMENT_PATH = BACKEND / "config" / "deployment_selection.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Macro-F1 phải hữu hạn trong [0,1].")
    return float(value)


def _verify_fingerprints(fingerprints, backend_root):
    if not isinstance(fingerprints, dict) or not fingerprints or not any(k.startswith("models/") for k in fingerprints):
        raise ValueError("Thiếu model fingerprints trong artifact.")
    root = Path(backend_root).resolve()
    for name, expected in fingerprints.items():
        # Deployment never opens Validation/Test CSV. Audit their recorded hashes as metadata only.
        if name.startswith("data/"):
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Fingerprint path nằm ngoài backend.")
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu frozen artifact/code: {path}")
        if _sha(path) != expected:
            raise ValueError(f"Model/code fingerprint không khớp Validation: {name}")


def build_deployment_config(selection_path=SELECTION_PATH, scores_path=SCORES_PATH,
                            freeze_path=FREEZE_PATH, *, backend_root=BACKEND, test_path=None) -> dict:
    selection_path, scores_path = map(Path, (selection_path, scores_path))
    freeze_path = Path(freeze_path) if freeze_path is not None else None
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if raw.get("selection_dataset") != "validation" or raw.get("primary_metric") != "macro_f1" or raw.get("test_dataset_used") is not False:
        raise ValueError("Selection phải lấy từ Validation Macro-F1, không từ Test.")
    if raw.get("label_mapping") != LABEL_TO_ID or raw.get("probability_columns") != LABELS:
        raise ValueError("Label mapping/probability columns không khớp.")
    alpha = _alpha(raw["selected_alpha"])
    if raw.get("tie_break_rule") != TIE_BREAK_RULE:
        raise ValueError("Không hỗ trợ tie-break khác quy tắc đã lưu từ Bước 9.")
    with scores_path.open(encoding="utf-8", newline="") as stream:
        rows = [{"alpha": _alpha(float(r["alpha"])), "macro_f1": _score(float(r["macro_f1"]))} for r in csv.DictReader(stream)]
    grid = [r["alpha"] for r in rows]
    if not rows or len(grid) != len(set(grid)) or grid != raw["alpha_grid"] or 0.0 not in grid or 1.0 not in grid:
        raise ValueError("Alpha grid CSV không khớp artifact hoặc thiếu endpoints.")
    maximum = max(r["macro_f1"] for r in rows)
    tied = [r["alpha"] for r in rows if r["macro_f1"] == maximum]
    chosen = min(tied, key=lambda a: (a not in (0., 1.), a))
    if alpha != chosen or sorted(tied) != raw["tied_alphas"]:
        raise ValueError("Selected alpha/tied_alphas không khớp CSV và tie-break.")
    scores = {m: _score(raw[f"{m}_validation_macro_f1"]) for m in SUPPORTED_METHODS}
    endpoints = {r["alpha"]: r["macro_f1"] for r in rows}
    if (scores["ensemble"] != _score(raw["selected_validation_macro_f1"]) or scores["ensemble"] != maximum
            or scores["svm"] != endpoints[1.] or scores["transformer"] != endpoints[0.]):
        raise ValueError("Validation scores JSON/CSV không nhất quán.")
    # Endpoint preference in the saved experiment also resolves method equivalence.
    method = "transformer" if alpha == 0 else "svm" if alpha == 1 else "ensemble"
    if scores[method] != max(scores.values()):
        raise ValueError("Selected method không đạt Validation Macro-F1 cao nhất.")
    _verify_fingerprints(raw["fingerprints"], backend_root)
    hashes = {"alpha_selection.json": _sha(selection_path), "alpha_scores.csv": _sha(scores_path)}
    verified = False
    if freeze_path is not None and freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        cfg = frozen["configuration"]
        if (frozen.get("selected_before_test_inference") is not True or cfg["selected_alpha"] != alpha
                or cfg.get("selection_dataset") != "validation" or cfg.get("label_mapping") != LABEL_TO_ID
                or cfg.get("probability_columns") != LABELS
                or cfg["selected_method"] != method or cfg["selection_sha256"] != hashes["alpha_selection.json"]
                or cfg["validation_scores"] != scores or cfg["step9_fingerprints"] != raw["fingerprints"]):
            raise ValueError("Deployment selection không khớp frozen selection Bước 10.")
        hashes["evaluation_freeze.json"] = _sha(freeze_path)
        verified = True
    # Test metrics are optional display metadata only, read AFTER the selection above.
    test_path = (Path(backend_root) / "outputs" / "final_evaluation" / "test_metrics.json"
                 if test_path is None else Path(test_path))
    test_scores = None
    if test_path.is_file():
        report = json.loads(test_path.read_text(encoding="utf-8"))
        if report["selected_method"] != method or report["selected_alpha"] != alpha or report["selection_dataset"] != "validation":
            raise ValueError("Test report metadata không khớp frozen Validation selection.")
        test_scores = {m: _score(report["models"][m]["macro_f1"]) for m in SUPPORTED_METHODS}
        hashes["test_metrics.json"] = _sha(test_path)
    config = {"selection_source": "validation", "selection_metric": "macro_f1", "selected_method": method,
              "selected_alpha": alpha, "default_view": "user", "comparison_view_available": True,
              "document_aggregation_default": "mean_probability", "aggregation_validated_optimal": False,
              "validation_macro_f1": scores, "test_macro_f1": test_scores, "metric_scope": "dataset_not_input_confidence",
              "tie_break_rule": raw["tie_break_rule"], "frozen_selection_verified": verified,
              "model_fingerprints": raw["fingerprints"], "artifact_hashes": hashes}
    validate_deployment_config(config)
    return config


def validate_deployment_config(config):
    if not isinstance(config, dict):
        raise TypeError("Deployment config phải là dictionary.")
    _alpha(config["selected_alpha"])
    if config.get("selected_method") not in SUPPORTED_METHODS:
        raise ValueError("Deployment selected_method không hợp lệ.")
    expected = {"selection_source": "validation", "selection_metric": "macro_f1", "default_view": "user",
                "comparison_view_available": True, "document_aggregation_default": "mean_probability",
                "aggregation_validated_optimal": False}
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in expected.items()):
        raise ValueError("Deployment policy không hợp lệ; Mean Probability chưa được kiểm chứng tối ưu.")


def write_deployment_config(path=DEPLOYMENT_PATH, config=None):
    if config is None:
        existing = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
        if existing.get("artifact_profile") == "transformer_epoch21":
            from .epoch21_deployment import build_epoch21_deployment_config
            config = build_epoch21_deployment_config()
        else:
            config = build_deployment_config()
    validate_deployment_config(config)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_deployment_config(path=DEPLOYMENT_PATH, **artifact_paths):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_deployment_config(config)
    profile = config.get("artifact_profile")
    if profile == "transformer_epoch21":
        if artifact_paths:
            raise ValueError("Không trộn artifact profile epoch21 với đường dẫn thí nghiệm khác.")
        from .epoch21_deployment import build_epoch21_deployment_config
        expected = build_epoch21_deployment_config()
    elif profile is not None:
        raise ValueError("Deployment artifact profile không được hỗ trợ.")
    else:
        expected = build_deployment_config(**artifact_paths)
    if config != expected:
        raise ValueError("Deployment JSON đã cũ hoặc không khớp artifacts; không fallback. Tạo lại bằng python -m backend.src.deployment_config.")
    return config


if __name__ == "__main__":
    write_deployment_config()
    print(f"Deployment selection verified and written: {DEPLOYMENT_PATH}")
