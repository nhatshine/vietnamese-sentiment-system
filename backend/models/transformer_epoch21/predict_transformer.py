"""Portable Transformer inference API for the NLP Part-3 backend."""
from __future__ import annotations
import json
import os
import hashlib
from functools import lru_cache
from pathlib import Path
import torch
from model import SentimentTransformer
from tokenizer import encode_text, load_vocab

LABEL_NAMES = ("negative", "neutral", "positive")
MODEL_KEYS = ("vocab_size", "max_len", "d_model", "num_heads", "num_layers", "d_ff", "dropout", "num_classes")


class TransformerPredictor:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        folder = Path(model_dir).resolve()
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.max_len = int(cfg["max_len"])
        if "preprocess_sha256" in cfg:
            source = (Path(__file__).resolve().parent / "preprocess.py")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != cfg["preprocess_sha256"]:
                raise ValueError("preprocess.py không khớp file dùng lúc train; không thể inference an toàn.")
        self.vocab = load_vocab(folder / "vocab.json")
        if cfg["vocab_size"] != len(self.vocab):
            raise ValueError("vocab_size không khớp vocab.json")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SentimentTransformer(**{key: cfg[key] for key in MODEL_KEYS}).to(self.device)
        weights = torch.load(folder / "best_transformer.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, texts: list[str], batch_size: int = 64) -> list[dict[str, float]]:
        if not isinstance(texts, list):
            raise TypeError("texts phải là list[str]")
        results = []
        for offset in range(0, len(texts), batch_size):
            chunk = texts[offset:offset + batch_size]
            encoding = [encode_text(text, self.vocab, self.max_len) for text in chunk]
            input_ids = torch.tensor([x[0] for x in encoding], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([x[1] for x in encoding], dtype=torch.long, device=self.device)
            p = self.model(input_ids, attention_mask).softmax(dim=-1).cpu().tolist()
            results.extend([{name: float(prob) for name, prob in zip(LABEL_NAMES, row)} for row in p])
        return results

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]


@lru_cache(maxsize=4)
def _get_predictor(model_dir: str, device: str | None):
    return TransformerPredictor(model_dir, device)


def predict_transformer(text: str, model_dir: str | Path | None = None,
                        device: str | None = None) -> dict[str, float]:
    folder = Path(model_dir or os.environ.get("TRANSFORMER_MODEL_DIR") or Path(__file__).resolve().parent)
    return _get_predictor(str(folder.resolve()), device).predict(text)


def predict_transformer_batch(texts: list[str], model_dir: str | Path | None = None,
                              device: str | None = None, batch_size: int = 64) -> list[dict[str, float]]:
    folder = Path(model_dir or os.environ.get("TRANSFORMER_MODEL_DIR") or Path(__file__).resolve().parent)
    return _get_predictor(str(folder.resolve()), device).predict_batch(texts, batch_size=batch_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--text", default="Dịch vụ rất tốt, mình rất hài lòng")
    args = parser.parse_args()
    print(json.dumps(predict_transformer(args.text, args.model_dir), ensure_ascii=False, indent=2))
