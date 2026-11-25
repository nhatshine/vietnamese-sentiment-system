"""HTTP schemas preserving the canonical Step11 response and full precision."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from .result_selector import select_result

Method = Literal["svm", "transformer", "ensemble"]
Label = Literal["negative", "neutral", "positive"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: StrictStr = Field(min_length=1, description="Raw Vietnamese text, unchanged. Runtime character limit rejects oversized input without truncation.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, text):
        if not text.strip():
            raise ValueError("Văn bản không được rỗng hoặc chỉ whitespace.")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Văn bản phải là Unicode hợp lệ.") from error
        return text


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)


class Probabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)


class PredictionResult(CanonicalModel):
    method: Method
    label: Label
    confidence: float = Field(ge=0, le=1)
    probabilities: Probabilities


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    svm: PredictionResult
    transformer: PredictionResult
    ensemble: PredictionResult


class SelectionInfo(CanonicalModel):
    selected_method: Method
    selected_alpha: float = Field(ge=0, le=1)
    selected_by: Literal["validation_macro_f1"]


class AggregationInfo(CanonicalModel):
    method: Literal["mean_probability"]
    applied_to_final: bool
    validated_optimal: Literal[False]
    policy: str


class SentenceResult(CanonicalModel):
    sentence_index: int = Field(ge=0)
    sentence: str
    final_result: PredictionResult
    comparison: Comparison


class AnalyzeResponse(CanonicalModel):
    input_text: str
    sentence_count: int = Field(ge=1)
    sentences: list[SentenceResult]
    document_results: dict[str, Any]
    final_result: PredictionResult
    selection_info: SelectionInfo
    aggregation_info: AggregationInfo
    comparison: Comparison
    display_info: dict[str, Any]
    research_info: dict[str, Any]

    @model_validator(mode="after")
    def canonical_consistency(self):
        if self.sentence_count != len(self.sentences) or [s.sentence_index for s in self.sentences] != list(range(self.sentence_count)):
            raise ValueError("Canonical sentence count/order không nhất quán.")
        method = self.selection_info.selected_method
        if self.final_result.model_dump() != select_result(self.comparison.model_dump(), method):
            raise ValueError("Canonical final_result không khớp selected method.")
        for sentence in self.sentences:
            if sentence.final_result.model_dump() != select_result(sentence.comparison.model_dump(), method):
                raise ValueError("Sentence final_result không khớp selected method.")
        return self


class ModelMetric(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    macro_f1: float = Field(ge=0, le=1)


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    selected_method: Method
    selected_alpha: float = Field(ge=0, le=1)
    selection_source: Literal["validation"]
    selection_metric: Literal["macro_f1"]
    models: dict[Method, dict[str, str]]
    validation_metrics: dict[Method, ModelMetric]
    test_metrics: dict[Method, ModelMetric] | None
    metric_scope: Literal["dataset_not_input_confidence"]
    document_aggregation: dict[str, Any]
    device: Literal["cpu"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["sentiment-analysis-api"]
    version: str
    model_ready: bool
