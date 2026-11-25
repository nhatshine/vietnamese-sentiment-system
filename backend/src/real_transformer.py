"""Inference Transformer gốc trên CPU/CUDA, cache weights và giữ tokenizer."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .config import LABELS
from .contracts import validate_prediction_output


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "transformer"
PREPROCESS_PATH = Path(__file__).with_name("preprocess.py")
_ARTIFACT_FILES = (
    "best_transformer.pth", "model.py", "tokenizer.py", "predict_transformer.py",
    "vocab.json", "config.json", "label_map.json", "preprocess.py",
)


def _read_labels(path: Path, num_classes: int) -> tuple[str, ...]:
    """Đọc mapping cột logits từ label_map, không mặc định thứ tự lớp."""
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if num_classes != len(LABELS) or not isinstance(mapping, dict):
        raise ValueError("label_map.json phải ánh xạ đúng ba class của hệ thống.")
    if set(mapping) != {str(index) for index in range(num_classes)}:
        raise ValueError("label_map.json có class index bị thiếu hoặc không hợp lệ.")
    labels = tuple(mapping[str(index)] for index in range(num_classes))
    if not all(isinstance(label, str) for label in labels) or set(labels) != set(LABELS):
        raise ValueError("label_map.json phải có đủ ba nhãn lowercase, không trùng.")
    return labels


@lru_cache(maxsize=2)
def _load_cached_model(device: str):
    missing = [name for name in _ARTIFACT_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu artifact Transformer trong {MODEL_DIR}: {', '.join(missing)}")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    labels = _read_labels(MODEL_DIR / "label_map.json", config["num_classes"])
    expected_hash = config.get("preprocess_sha256")
    shared_hash = hashlib.sha256(PREPROCESS_PATH.read_bytes()).hexdigest()
    artifact_hash = hashlib.sha256((MODEL_DIR / "preprocess.py").read_bytes()).hexdigest()
    if not expected_hash or shared_hash != expected_hash or artifact_hash != expected_hash:
        raise ValueError("preprocess.py không khớp SHA-256 của preprocessing dùng khi train Transformer.")

    try:
        import torch
        from backend.models.transformer.predict_transformer import TransformerPredictor
    except ImportError as error:
        raise ImportError(
            "Real Transformer cần PyTorch tương thích. Dùng virtual environment của project "
            "và cài: python -m pip install -r backend/requirements.txt"
        ) from error

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA không có sẵn; hãy dùng device='cpu'.")
    # Constructor gốc: kiến trúc từ config, weights_only=True, map_location='cpu',
    # load_state_dict(strict=True) và model.eval(). Không thay đổi kiến trúc.
    original = TransformerPredictor(MODEL_DIR, device=str(selected_device))
    return SimpleNamespace(
        model=original.model, vocab=original.vocab, max_len=original.max_len,
        device=original.device, config=config, labels=labels,
    )


def load_model(device: str = "cpu"):
    """Load/cached resources theo device; CPU mặc định, không yêu cầu GPU."""
    if not isinstance(device, str):
        raise TypeError("device phải là chuỗi như 'cpu' hoặc 'cuda'.")
    return _load_cached_model(device)


def predict_transformer(text: str, *, device: str = "cpu") -> dict[str, float]:
    """Nhận raw text; tokenizer gốc tự clean đúng một lần, không clean trước."""
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi (str).")
    if not text.strip():
        raise ValueError("text không được rỗng hoặc chỉ chứa khoảng trắng.")

    resources = load_model(device)
    import torch
    from backend.models.transformer.tokenizer import encode_text

    ids, mask = encode_text(text, resources.vocab, resources.max_len)
    with torch.inference_mode():
        input_ids = torch.tensor([ids], dtype=torch.long, device=resources.device)
        attention_mask = torch.tensor([mask], dtype=torch.long, device=resources.device)
        probabilities = resources.model(input_ids, attention_mask).softmax(dim=-1).cpu().tolist()[0]

    if len(probabilities) != len(resources.labels):
        raise ValueError("Số cột logits không khớp label_map.json.")
    mapped = {label: float(value) for label, value in zip(resources.labels, probabilities)}
    result = {label: mapped[label] for label in LABELS}
    validate_prediction_output(result)
    return result


def encode_cleaned_text(text: str, vocab: dict[str, int], max_len: int):
    """Cùng CLS/UNK/PAD/truncation của encode_text gốc, bỏ duy nhất clean_text."""
    from backend.models.transformer.tokenizer import CLS_ID, UNK_ID, PAD_ID
    if not isinstance(text, str):
        raise TypeError("Cleaned text phải là str.")
    if not text.strip():
        raise ValueError("Cleaned text không được rỗng.")
    if max_len < 2:
        raise ValueError("max_len phải >= 2.")
    ids = [CLS_ID] + [vocab.get(token, UNK_ID) for token in text.split()[:max_len - 1]]
    mask = [1] * len(ids)
    padding = max_len - len(ids)
    return ids + [PAD_ID] * padding, mask + [0] * padding


def predict_transformer_cleaned_batch(
    texts: list[str], *, batch_size: int = 64, device: str = "cpu", progress=None,
) -> list[dict[str, float]]:
    """Batch CSV cleaned: dùng model/vocab đã cache, không gọi preprocessing."""
    if not isinstance(texts, list):
        raise TypeError("texts phải là list[str] đã preprocessing.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương.")
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Mỗi cleaned text phải là str.")
        if not text.strip():
            raise ValueError("Cleaned text không được rỗng.")
    if not texts:
        return []
    resources = load_model(device)
    import torch
    results = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded = [encode_cleaned_text(text, resources.vocab, resources.max_len) for text in chunk]
            ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=resources.device)
            masks = torch.tensor([item[1] for item in encoded], dtype=torch.long, device=resources.device)
            rows = resources.model(ids, masks).softmax(dim=-1).cpu().tolist()
            if len(rows) != len(chunk):
                raise ValueError("Transformer trả sai số mẫu trong batch.")
            for row in rows:
                if len(row) != len(resources.labels):
                    raise ValueError("Số cột logits không khớp label_map.json.")
                mapped = {label: float(value) for label, value in zip(resources.labels, row)}
                result = {label: mapped[label] for label in LABELS}
                validate_prediction_output(result)
                results.append(result)
            if progress is not None:
                progress(len(results), len(texts))
    return results
