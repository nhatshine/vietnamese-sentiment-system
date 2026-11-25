"""Train/evaluate a Vietnamese sentiment Transformer from random initialization.

No data leakage: vocabulary is built only from train.csv; val chooses the best
checkpoint by Macro-F1; test is evaluated only after training ends.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SentimentTransformer
from preprocess import clean_text
PREPROCESS_DESCRIPTION = "Shared team preprocess.py: html.unescape, NFC, lowercase, URLs/emails/mentions removed, emoticon->emopos/emoneg, xnum, repeated chars, slang, punctuation and whitespace; keep Vietnamese negations"
from tokenizer import build_vocab, encode_text, save_vocab, UNK_ID, tokenize

LABEL_NAMES = ["negative", "neutral", "positive"]
COLUMNS = ["text", "label", "source"]
CORE_CONFIG_KEYS = ("max_len", "d_model", "num_heads", "num_layers", "d_ff", "dropout", "num_classes", "vocab_size")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            if not set(COLUMNS).issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{path} phải có đủ 3 cột: {COLUMNS}; thực tế: {reader.fieldnames}")
        rows = []
        for i, row in enumerate(reader, start=2):
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or not text.strip() or str(label) not in ("0", "1", "2"):
                raise ValueError(f"Dữ liệu không hợp lệ: {path}, dòng {i}")
            rows.append({"text": text, "label": int(label), "source": row.get("source", "") or ""})
    if not rows:
        raise ValueError(f"File rỗng: {path}")
    return rows


def sha256_file(path: Path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def verify_splits(datasets):
    normalized = {}
    for name, rows in datasets.items():
        texts = [clean_text(row["text"]) for row in rows]
        if len(set(texts)) != len(texts):
            print(f"CẢNH BÁO: {name} chứa {len(texts) - len(set(texts))} văn bản trùng sau preprocessing")
        normalized[name] = set(texts)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = len(normalized[a] & normalized[b])
        if overlap:
            raise ValueError(f"Phát hiện {overlap} câu trùng giữa {a} và {b} sau preprocessing: kiểm tra dữ liệu!")
    print("DATA CHECK: không có trùng text giữa các split sau preprocessing.")


class SentimentDataset(Dataset):
    def __init__(self, rows, vocab, max_len):
        encoded = [encode_text(row["text"], vocab, max_len) for row in rows]
        self.ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        self.masks = torch.tensor([e[1] for e in encoded], dtype=torch.long)
        self.labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids[index], self.masks[index], self.labels[index]


def run_eval(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, probabilities = [], [], []
    with torch.inference_mode():
        for ids, mask, label in loader:
            ids, mask, label = ids.to(device), mask.to(device), label.to(device)
            logits = model(ids, mask)
            loss = loss_fn(logits, label)
            prob = torch.softmax(logits, dim=-1)
            pred = prob.argmax(dim=1)
            total_loss += float(loss.item()) * len(label)
            y_true.extend(label.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            probabilities.extend(prob.cpu().tolist())
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "y_true": y_true, "y_pred": y_pred, "probabilities": probabilities,
    }


def save_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(rows, evaluated, path: Path):
    """Export stable split-local IDs to safely align SVM and Transformer outputs."""
    if len(rows) != len(evaluated["y_pred"]) or len(rows) != len(evaluated["probabilities"]):
        raise ValueError(f"Prediction count != source rows for {path.name}")
    split = path.stem.split("_")[0]  # val or test
    if split not in ("val", "test"):
        raise ValueError(f"Unknown prediction split: {split}")
    out_rows = []
    for i, (row, prediction, prob) in enumerate(zip(rows, evaluated["y_pred"], evaluated["probabilities"])):
        out_rows.append({"sample_id": f"{split}:{i:06d}", "row_index": i,
                         "text": row["text"], "true_label": row["label"],
                         "pred_label": prediction, "p_negative": prob[0],
                         "p_neutral": prob[1], "p_positive": prob[2], "source": row["source"]})
    save_csv(path, ["sample_id", "row_index", "text", "true_label", "pred_label",
                    "p_negative", "p_neutral", "p_positive", "source"], out_rows)


def save_split_report(evaluated, out: Path, split: str):
    report = classification_report(evaluated["y_true"], evaluated["y_pred"],
                                   labels=[0, 1, 2], target_names=LABEL_NAMES,
                                   zero_division=0)
    (out / f"{split}_classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(evaluated["y_true"], evaluated["y_pred"], labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABEL_NAMES,
           yticklabels=LABEL_NAMES, xlabel="Predicted", ylabel="True",
           title=f"{split.upper()} confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / f"{split}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return {"loss": evaluated["loss"], "accuracy": evaluated["accuracy"],
            "macro_precision": evaluated["macro_precision"], "macro_recall": evaluated["macro_recall"],
            "macro_f1": evaluated["macro_f1"], "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                evaluated["y_true"], evaluated["y_pred"], labels=[0, 1, 2],
                target_names=LABEL_NAMES, zero_division=0, output_dict=True)}


def save_history(history, out):
    if not history:
        return
    save_csv(out / "train_history.csv", list(history[0].keys()), history)
    x = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["train_loss"] for r in history], marker="o", label="Train loss")
    ax.plot(x, [r["val_loss"] for r in history], marker="o", label="Val loss")
    ax.set(xlabel="Epoch", ylabel="CrossEntropyLoss", title="Train/Validation Loss")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["val_accuracy"] for r in history], marker="o", label="Val Accuracy")
    ax.plot(x, [r["val_macro_f1"] for r in history], marker="o", label="Val Macro-F1")
    ax.set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(out / "validation_metrics.png", dpi=160); plt.close(fig)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-vocab-size", type=int, default=25000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume from last_training_state.pth if available")
    return p.parse_args()


def main():
    a = get_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CẢNH BÁO: GPU chưa bật. Colab > Runtime > Change runtime type > T4 GPU")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
    last_state_path = a.out_dir / "last_training_state.pth"
    if a.resume:
        if not last_state_path.is_file() or not (a.out_dir / "config.json").is_file() or not (a.out_dir / "vocab.json").is_file():
            raise FileNotFoundError("RESUME=True nhưng thiếu last_training_state.pth/config.json/vocab.json trong output. "
                                    "Chỉ tiếp tục phiên cũ với đúng bộ dữ liệu và cùng output folder.")
    else:
        existing = [n for n in ("last_training_state.pth", "best_transformer.pth", "config.json", "vocab.json")
                    if (a.out_dir / n).exists()]
        if existing:
            raise FileExistsError(f"Output {a.out_dir} đã chứa kết quả {existing}. "
                                  "Không train từ đầu để tránh ghi đè; chọn thư mục mới hoặc RESUME=True.")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = a.out_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {name: a.data_dir / f"{name}.csv" for name in ("train", "val", "test")}
    for path in data_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy: {path}")
    rows = {name: read_csv(path) for name, path in data_paths.items()}
    verify_splits(rows)
    dataset_sha256 = {name: sha256_file(path) for name, path in data_paths.items()}
    preprocess_sha256 = sha256_file(Path(__file__).resolve().parent / "preprocess.py")
    for name, value in rows.items():
        print(name.upper(), "samples =", len(value), "labels =", dict(Counter(r["label"] for r in value)))

    config_path = a.out_dir / "config.json"
    if a.resume and config_path.exists() and (a.out_dir / "vocab.json").exists():
        from tokenizer import load_vocab
        vocab = load_vocab(a.out_dir / "vocab.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = dict(max_len=a.max_len, d_model=a.d_model, num_heads=a.num_heads,
                        num_layers=a.num_layers, d_ff=a.d_ff, dropout=a.dropout,
                        num_classes=3, vocab_size=len(vocab))
        if any(config.get(k) != v for k, v in expected.items()) or config.get("batch_size") != a.batch_size or config.get("lr") != a.lr or config.get("seed") != a.seed or config.get("early_stopping_patience") != a.patience:
            raise ValueError("Resume có config khác: phải dùng đúng hyperparameter cũ hoặc thư mục output mới")
        if config.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Dữ liệu train/val/test thay đổi so với lần chạy trước")
        if config.get("preprocess_sha256") != preprocess_sha256:
            raise ValueError("preprocess.py thay đổi so với checkpoint trước. Cần dùng lại file cũ hoặc output folder mới.")
    else:
        vocab = build_vocab((r["text"] for r in rows["train"]), a.min_freq, a.max_vocab_size)
        save_vocab(vocab, a.out_dir / "vocab.json")
        config = dict(vocab_size=len(vocab), max_len=a.max_len, d_model=a.d_model,
                      num_heads=a.num_heads, num_layers=a.num_layers, d_ff=a.d_ff,
                      dropout=a.dropout, num_classes=3, min_freq=a.min_freq,
                      max_vocab_size=a.max_vocab_size, lr=a.lr, batch_size=a.batch_size,
                      seed=a.seed, max_epochs=a.epochs, early_stopping_patience=a.patience,
                      seed_strategy="python/numpy/torch + resume RNG state",
                      optimizer="AdamW", loss="CrossEntropyLoss",
                      tokenizer="Vietnamese whitespace/syllable tokenizer",
                      preprocessing=PREPROCESS_DESCRIPTION,
                      dataset_sha256=dataset_sha256,
                      preprocess_sha256=preprocess_sha256)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (a.out_dir / "label_map.json").write_text(json.dumps({str(i): s for i,s in enumerate(LABEL_NAMES)}, indent=2), encoding="utf-8")
    print("Vocabulary size:", len(vocab))

    train_set = SentimentDataset(rows["train"], vocab, a.max_len)
    val_set = SentimentDataset(rows["val"], vocab, a.max_len)
    test_set = SentimentDataset(rows["test"], vocab, a.max_len)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=a.batch_size * 2, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=a.batch_size * 2, num_workers=0)
    model = SentimentTransformer(**{k: config[k] for k in CORE_CONFIG_KEYS}).to(device)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    best_score, best_epoch, no_improve = -1.0, 0, 0
    start_epoch = 1
    if a.resume:
        state = torch.load(last_state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        history = state["history"]
        best_score, best_epoch, no_improve = state["best_score"], state["best_epoch"], state["no_improve"]
        start_epoch = state["epoch"] + 1
        # Restore RNG to make shuffling/dropout reproducible after a Colab disconnect.
        if "rng_python" in state:
            random.setstate(state["rng_python"])
            np_state = state["rng_numpy"]
            np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32),
                                 np_state[2], np_state[3], np_state[4]))
            torch.set_rng_state(state["rng_torch"])
            if torch.cuda.is_available() and state.get("rng_cuda"):
                torch.cuda.set_rng_state_all(state["rng_cuda"])
        print("RESUMED after epoch", start_epoch - 1, "best epoch", best_epoch)

    stop_reason = "max_epochs"
    if no_improve >= a.patience:
        print("Đã đạt Early Stopping ở lần chạy trước; không train thêm epoch.")
        stop_reason = "early_stopping_before_resume"
    for epoch in range(start_epoch, a.epochs + 1) if no_improve < a.patience else []:
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(y)
        val = run_eval(model, val_loader, device, loss_fn)
        scheduler.step(val["macro_f1"])
        row = {"epoch": epoch, "train_loss": total_loss / len(train_set),
               "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        improved = val["macro_f1"] > best_score + 1e-9
        if improved:
            best_score, best_epoch, no_improve = val["macro_f1"], epoch, 0
            torch.save(model.state_dict(), a.out_dir / "best_transformer.pth")
        else:
            no_improve += 1
        save_history(history, metric_dir)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best_score": best_score, "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "rng_python": random.getstate(),
                    "rng_numpy": (np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                  np.random.get_state()[2], np.random.get_state()[3], np.random.get_state()[4]),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    }, last_state_path)
        print(f"Epoch {epoch:02d}/{a.epochs} | train_loss={row['train_loss']:.4f} | "
              f"val_loss={val['loss']:.4f} | val_acc={val['accuracy']:.4f} | "
              f"val_macro_f1={val['macro_f1']:.4f} | best={best_score:.4f} epoch={best_epoch} "
              f"| {time.perf_counter()-t0:.1f}s", flush=True)
        if no_improve >= a.patience:
            stop_reason = "early_stopping"
            print(f"EARLY STOP at epoch {epoch} (patience={a.patience})")
            break
    if not (a.out_dir / "best_transformer.pth").exists():
        raise RuntimeError("Chưa có best_transformer.pth: kiểm tra train loop!")
    model.load_state_dict(torch.load(a.out_dir / "best_transformer.pth", map_location="cpu", weights_only=True))
    model.eval()
    # Test is evaluated here, after the best epoch was chosen ONLY by Validation.
    results = {}
    for name, loader in (("val", val_loader), ("test", test_loader)):
        result = run_eval(model, loader, device, loss_fn)
        save_predictions(rows[name], result, metric_dir / f"{name}_predictions.csv")
        results[name] = save_split_report(result, metric_dir, name)
        print(f"BEST MODEL {name}: accuracy={result['accuracy']:.4f} "
              f"macro_F1={result['macro_f1']:.4f} "
              f"precision={result['macro_precision']:.4f} recall={result['macro_recall']:.4f}")
    summary = {"best_epoch": best_epoch, "best_val_macro_f1": best_score,
                "last_epoch_completed": history[-1]["epoch"] if history else 0,
                "early_stopping_patience": a.patience,
                "stop_reason": stop_reason,
                "epoch_limit": a.epochs,
               "split_sizes": {k: len(v) for k,v in rows.items()},
               "split_label_counts": {k: dict(Counter(str(r["label"]) for r in v)) for k,v in rows.items()},
               "validation": results["val"], "test": results["test"],
               "data_sha256": dataset_sha256,
               "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                            "numpy": np.__version__}}
    (metric_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Copy the exact code used for training into the portable handover folder.
    src_dir = Path(__file__).resolve().parent
    for filename in ("model.py", "preprocess.py", "tokenizer.py", "predict_transformer.py",
                     "train_transformer.py", "evaluate_transformer.py", "requirements.txt", "README.md"):
        src = src_dir / filename
        if src.is_file():
            shutil.copy2(src, a.out_dir / filename)
    print("\nDONE. Files saved at:", a.out_dir)
    print("Inference: python predict_transformer.py --model-dir", a.out_dir, '--text "Dịch vụ rất tốt"')


if __name__ == "__main__":
    main()
