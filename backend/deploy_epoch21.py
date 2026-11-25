"""Stage the delivered checkpoint and verified comparison as a versioned local demo."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from backend.src.alpha_tuning import TIE_BREAK_RULE
from backend.src.config import LABELS, LABEL_TO_ID
from backend.src.deployment_config import BACKEND, build_deployment_config, write_deployment_config
from backend.src.epoch21_deployment import ARTIFACT_DIR, MODEL_DIR, build_epoch21_deployment_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(archive, results_dir, *, activate=False):
    archive, results_dir = Path(archive), Path(results_dir)
    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    if (results["checkpoint_epoch"] != 21 or results["selected_alpha"] != .7
            or sha(archive) != results["provenance"]["archive_sha256"]):
        raise ValueError("Sai bộ kết quả hoặc ZIP Transformer đã kiểm tra.")
    for split in ("val", "test"):
        if sha(BACKEND / "data" / f"{split}.csv") != results["provenance"]["dataset_sha256"][split]:
            raise ValueError("Dữ liệu không khớp lượt thực nghiệm mới.")
    if sha(BACKEND / "models/svm/svm_pipeline.joblib") != results["provenance"]["svm_model_sha256"]:
        raise ValueError("SVM không khớp kết quả đã đánh giá.")
    if not MODEL_DIR.resolve().is_relative_to((BACKEND / "models").resolve()):
        raise ValueError("Model directory nằm ngoài backend/models.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    names = ["best_transformer.pth", "config.json", "vocab.json", "label_map.json", "preprocess.py",
             "model.py", "tokenizer.py", "predict_transformer.py"]
    with ZipFile(archive) as source:
        for name in names:
            (MODEL_DIR / name).write_bytes(source.read("transformer_retrain_30ep/" + name))
    (MODEL_DIR / "__init__.py").write_text('"""Delivered Transformer checkpoint epoch 21 artifacts."""\n', encoding="utf-8")
    if sha(MODEL_DIR / "best_transformer.pth") != results["provenance"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint không khớp bộ dự đoán epoch21.")
    if (MODEL_DIR / "model.py").read_bytes() != (BACKEND / "models/transformer/model.py").read_bytes():
        raise ValueError("Không thể tái sử dụng adapter: kiến trúc đã thay đổi.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (results_dir / "validation_alpha_scores.csv").open(encoding="utf-8-sig", newline="") as f:
        scores = list(csv.DictReader(f))
    with (ARTIFACT_DIR / "alpha_scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scores[0])); w.writeheader(); w.writerows(scores)
    relative = ["models/svm/svm_pipeline.joblib", "models/transformer/model.py",
                "models/transformer/tokenizer.py", "models/transformer/preprocess.py",
                "models/transformer/predict_transformer.py", "src/preprocess.py", "src/real_svm.py",
                "src/ensemble.py", "src/alpha_tuning.py", "src/epoch21_deployment.py",
                "src/deployment_config.py", "src/prediction_service.py", "src/api_dependencies.py"]
    relative += ["models/transformer_epoch21/" + name for name in names]
    fingerprints = {name: sha(BACKEND / name) for name in relative}
    fingerprints.update({"data/" + split + ".csv": digest for split, digest in results["provenance"]["dataset_sha256"].items()})
    val = results["splits"]["val"]["models"]
    selection = {"selection_dataset": "validation", "primary_metric": "macro_f1", "test_dataset_used": False,
                 "selected_alpha": results["selected_alpha"], "selected_validation_macro_f1": val["ensemble"]["macro_f1"],
                 "alpha_grid": [float(row["alpha"]) for row in scores], "tied_alphas": [.7],
                 "tie_break_rule": TIE_BREAK_RULE, "label_mapping": LABEL_TO_ID, "probability_columns": LABELS,
                 "fingerprints": fingerprints,
                 **{name + "_validation_macro_f1": val[name]["macro_f1"] for name in ("svm", "transformer", "ensemble")}}
    (ARTIFACT_DIR / "alpha_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    test = {"selected_method": "ensemble", "selected_alpha": .7, "selection_dataset": "validation",
            "sample_count": 8238, "models": results["splits"]["test"]["models"]}
    (ARTIFACT_DIR / "test_metrics.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = build_deployment_config(ARTIFACT_DIR / "alpha_selection.json", ARTIFACT_DIR / "alpha_scores.csv", None,
                                  backend_root=BACKEND, test_path=ARTIFACT_DIR / "test_metrics.json")
    # This locks a deployment snapshot. It does not claim that already-delivered Test
    # predictions were generated after a new preregistration or after this file existed.
    lock = {"scope": "deployment_snapshot", "selection_source": "validation", "test_used_for_selection": False,
            "checkpoint_epoch": 21, "selected_alpha": cfg["selected_alpha"],
            "artifact_hashes": cfg["artifact_hashes"], "model_fingerprints": cfg["model_fingerprints"],
            "report_results_sha256": sha(results_dir / "results.json"),
            "checkpoint_sha256": sha(MODEL_DIR / "best_transformer.pth")}
    (ARTIFACT_DIR / "deployment_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    current = build_epoch21_deployment_config()
    (ARTIFACT_DIR / "deployment_selection.json").write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    if activate:
        write_deployment_config(config=current)
    return current


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path.home() / "Downloads/transformer_retrain_30ep-20261122T164834Z-1-001.zip")
    p.add_argument("--results", type=Path, default=BACKEND.parents[1] / "Ket_qua_ket_hop_Transformer_epoch21")
    p.add_argument("--activate", action="store_true")
    args = p.parse_args()
    cfg = stage(args.archive, args.results, activate=args.activate)
    print(json.dumps({"activated": args.activate, "profile": cfg["artifact_profile"],
                      "checkpoint_epoch": cfg["transformer_checkpoint_epoch"], "alpha": cfg["selected_alpha"],
                      "validation_macro_f1": cfg["validation_macro_f1"], "test_macro_f1": cfg["test_macro_f1"]}, indent=2))
