"""Thin local HTTP layer over the frozen real PredictionService."""

from contextlib import asynccontextmanager
import json
import logging
import os
from threading import Lock
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api_dependencies import build_model_info, create_prediction_service, get_prediction_service
from .api_schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, ModelInfoResponse

VERSION = "1.0.0"
logger = logging.getLogger(__name__)
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _cleaned_empty(error):
    # Existing Step7 contract: recognize only this specific input error, not arbitrary ValueErrors.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ValueError) and str(error) == "Văn bản rỗng sau preprocessing; không thực hiện inference.":
            return True
        error = error.__cause__
    return False


def create_app(*, service_factory=None, max_text_chars=None, cors_origins=None):
    if max_text_chars is None:
        max_text_chars = int(os.environ.get("SENTIMENT_MAX_TEXT_CHARS", "20000"))
    if isinstance(max_text_chars, bool) or not isinstance(max_text_chars, int) or max_text_chars < 1:
        raise ValueError("SENTIMENT_MAX_TEXT_CHARS phải là số nguyên dương.")
    if cors_origins is None:
        extras = [s.strip() for s in os.environ.get("SENTIMENT_CORS_ORIGINS", "").split(",") if s.strip()]
        cors_origins = list(dict.fromkeys((*DEFAULT_ORIGINS, *extras)))
    if not isinstance(cors_origins, (list, tuple)):
        raise TypeError("CORS origins phải là list/tuple.")
    for origin in cors_origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("CORS origins phải là origin HTTP(S) cụ thể, không wildcard/path.")
    factory = create_prediction_service if service_factory is None else service_factory

    @asynccontextmanager
    async def lifespan(application):
        application.state.ready = False
        try:
            service = await run_in_threadpool(factory)
            config = service.config
            info = build_model_info(config)
            application.state.prediction_service = service
            application.state.model_info = info
            application.state.frozen_selection = (config["selected_method"], config["selected_alpha"])
            application.state.ready = True
        except Exception as error:
            logger.error("Service startup failed (%s)", type(error).__name__)
            raise RuntimeError("Không thể khởi tạo model/config thật; kiểm tra artifacts và môi trường. Không fallback mock.") from error
        try:
            yield
        finally:
            application.state.ready = False
            application.state.prediction_service = None
            application.state.model_info = None

    application = FastAPI(title="Vietnamese Sentiment Analysis API",
                          description="Vietnamese sentiment analysis using SVM, Transformer and Weighted Ensemble. Frozen Validation selection; canonical data for user/research views.",
                          version=VERSION, lifespan=lifespan)
    application.state.ready = False
    application.state.prediction_service = None
    application.state.model_info = None
    application.state.inference_lock = Lock()
    application.state.max_text_chars = max_text_chars
    application.add_middleware(CORSMiddleware, allow_origins=list(cors_origins), allow_credentials=False,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Exclude input/ctx to avoid echoing request bodies or unserializable exception objects.
        details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.exception_handler(StarletteHTTPException)
    async def invalid_transport(request, error):
        if error.status_code == 400 and isinstance(error.__cause__, UnicodeDecodeError):
            return JSONResponse(status_code=422, content={"detail": "JSON body phải dùng Unicode/UTF-8 hợp lệ."})
        return await http_exception_handler(request, error)

    @application.get("/api/health", response_model=HealthResponse, summary="API liveness and model readiness")
    def health(request: Request):
        return {"status": "ok", "service": "sentiment-analysis-api", "version": VERSION,
                "model_ready": bool(request.app.state.ready and request.app.state.prediction_service is not None)}

    @application.get("/api/model-info", response_model=ModelInfoResponse,
                     summary="Frozen selection and dataset metrics, not input confidence", responses={503: {"description": "Model/config not ready"}})
    def model_info(request: Request, service=Depends(get_prediction_service)):
        return request.app.state.model_info

    @application.post("/api/analyze", response_model=AnalyzeResponse,
                      summary="Analyze raw Vietnamese text once and return full canonical data",
                      description="Only text is accepted. React retains canonical response to switch user/research presentation without new inference.",
                      responses={422: {"description": "Invalid input or a sentence becomes empty after preprocessing"},
                                 503: {"description": "Model/config not ready"}, 500: {"description": "Internal inference/output failure; no fabricated result"}})
    def analyze(payload: AnalyzeRequest, request: Request, service=Depends(get_prediction_service)):
        if len(payload.text) > request.app.state.max_text_chars:
            raise HTTPException(status_code=422, detail=f"Văn bản vượt giới hạn {request.app.state.max_text_chars} ký tự; không tự cắt bớt.")
        try:
            with request.app.state.inference_lock:
                result = service.analyze(payload.text)
        except Exception as error:
            if _cleaned_empty(error):
                raise HTTPException(status_code=422, detail="Một câu không còn nội dung sau preprocessing; không bỏ qua câu.") from None
            # No raw text, exception string or traceback in default inference logs.
            logger.error("Prediction failed (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Không thể hoàn tất phân tích; không có kết quả dự đoán thay thế.") from None
        try:
            if (result["selection_info"]["selected_method"], result["selection_info"]["selected_alpha"]) != request.app.state.frozen_selection:
                raise ValueError("Selection differs from startup snapshot")
            json.dumps(result, allow_nan=False, ensure_ascii=False).encode("utf-8")
            return AnalyzeResponse.model_validate(result)
        except Exception as error:
            logger.error("Invalid canonical output (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Output phân tích không hợp lệ; không trả dự đoán giả.") from None

    return application


app = create_app()
