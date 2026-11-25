"""Startup service factory and safe cached metadata; no dataset access."""

from fastapi import HTTPException, Request

from .api_schemas import ModelInfoResponse
from .deployment_config import validate_deployment_config
from .prediction_service import PredictionService


def create_prediction_service():
    service = PredictionService()
    # Confirm readiness without running a warmup prediction or changing preprocessing.
    from .real_svm import load_model as load_svm
    from .real_transformer import load_model as load_transformer
    load_svm()
    if service.config.get("artifact_profile") == "transformer_epoch21":
        from .epoch21_deployment import load_epoch21_transformer
        load_epoch21_transformer()
    else:
        load_transformer(device="cpu")
    return service


def build_model_info(config):
    validate_deployment_config(config)
    # Explicit whitelist: never expose model paths, fingerprints or arbitrary config extras.
    info = {
        "selected_method": config["selected_method"], "selected_alpha": config["selected_alpha"],
        "selection_source": config["selection_source"], "selection_metric": config["selection_metric"],
        "models": {"svm": {"name": "TF-IDF + SVM"}, "transformer": {"name": "Custom Transformer"},
                   "ensemble": {"name": "Weighted Soft Voting"}},
        "validation_metrics": {name: {"macro_f1": score} for name, score in config["validation_macro_f1"].items()},
        "test_metrics": ({name: {"macro_f1": score} for name, score in config["test_macro_f1"].items()}
                         if config["test_macro_f1"] is not None else None),
        "metric_scope": "dataset_not_input_confidence", "device": "cpu",
        "document_aggregation": {"default_method": config["document_aggregation_default"],
                                 "validated_optimal": config["aggregation_validated_optimal"]},
    }
    if config.get("transformer_checkpoint_epoch") == 21:
        info["models"]["transformer"]["name"] = "Transformer tự xây dựng · epoch 21"
    return ModelInfoResponse.model_validate(info).model_dump()


def get_prediction_service(request: Request):
    service = getattr(request.app.state, "prediction_service", None)
    if not getattr(request.app.state, "ready", False) or service is None:
        raise HTTPException(status_code=503, detail="Model/config chưa sẵn sàng.")
    return service
