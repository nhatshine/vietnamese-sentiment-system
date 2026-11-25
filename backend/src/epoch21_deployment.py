"""Versioned local deployment; original Step 9/10 artifacts remain historical."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .config import LABELS
from .contracts import validate_prediction_output
from .deployment_config import BACKEND, build_deployment_config

ARTIFACT_DIR = BACKEND / "outputs" / "epoch21_deployment"
MODEL_DIR = BACKEND / "models" / "transformer_epoch21"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_epoch21_deployment_config():
    cfg = build_deployment_config(
        ARTIFACT_DIR / "alpha_selection.json", ARTIFACT_DIR / "alpha_scores.csv", None,
        backend_root=BACKEND, test_path=ARTIFACT_DIR / "test_metrics.json",
    )
    lock = json.loads((ARTIFACT_DIR / "deployment_lock.json").read_text(encoding="utf-8"))
    if (lock.get("scope") != "deployment_snapshot"
            or lock.get("selection_source") != "validation"
            or lock.get("test_used_for_selection") is not False
            or lock.get("checkpoint_epoch") != 21
            or lock.get("selected_alpha") != cfg["selected_alpha"]
            or lock.get("artifact_hashes") != cfg["artifact_hashes"]
            or lock.get("model_fingerprints") != cfg["model_fingerprints"]):
        raise ValueError("Epoch21 deployment lock không khớp model, trọng số hoặc số liệu.")
    mapping = json.loads((MODEL_DIR / "label_map.json").read_text(encoding="utf-8"))
    if mapping != {str(i): label for i, label in enumerate(LABELS)}:
        raise ValueError("Thứ tự nhãn Transformer epoch21 không khớp.")
    cfg.update(artifact_profile="transformer_epoch21", transformer_checkpoint_epoch=21,
               deployment_lock_scope="deployment_snapshot", frozen_selection_verified=True)
    return cfg


@lru_cache(maxsize=1)
def load_epoch21_transformer():
    # The delivered network is byte-identical to the already-integrated architecture.
    # Reuse its strict weights_only CPU loader with the new folder, never the old weights.
    legacy = BACKEND / "models" / "transformer"
    if (MODEL_DIR / "model.py").read_bytes() != (legacy / "model.py").read_bytes():
        raise ValueError("Kiến trúc epoch21 khác adapter đã tích hợp.")
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    expected = cfg.get("preprocess_sha256")
    for path in [MODEL_DIR / "preprocess.py", legacy / "preprocess.py", BACKEND / "src" / "preprocess.py"]:
        if not expected or _sha(path) != expected:
            raise ValueError("Preprocessing không khớp Transformer epoch21.")
    from backend.models.transformer.predict_transformer import _get_predictor
    return _get_predictor(str(MODEL_DIR.resolve()), "cpu")


def predict_transformer_epoch21(text):
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi (str).")
    if not text.strip():
        raise ValueError("text không được rỗng.")
    result = load_epoch21_transformer().predict(text)
    ordered = {label: float(result[label]) for label in LABELS}
    validate_prediction_output(ordered)
    return ordered
