"""Re-run val/test reports from the saved best checkpoint, without retraining."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from model import SentimentTransformer
from tokenizer import load_vocab
from train_transformer import (read_csv, SentimentDataset, run_eval,
                               save_predictions, save_split_report, CORE_CONFIG_KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    vocab = load_vocab(args.model_dir / "vocab.json")
    model = SentimentTransformer(**{k: cfg[k] for k in CORE_CONFIG_KEYS}).to(device)
    model.load_state_dict(torch.load(args.model_dir / "best_transformer.pth", map_location="cpu", weights_only=True))
    model.eval()
    folder = args.model_dir / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        rows = read_csv(args.data_dir / f"{split}.csv")
        loader = DataLoader(SentimentDataset(rows, vocab, cfg["max_len"]), batch_size=args.batch_size)
        res = run_eval(model, loader, device, torch.nn.CrossEntropyLoss())
        save_predictions(rows, res, folder / f"{split}_predictions.csv")
        stats = save_split_report(res, folder, split)
        (folder / f"{split}_metrics_recomputed.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(split, "accuracy", round(res["accuracy"], 4), "macro F1", round(res["macro_f1"], 4))


if __name__ == "__main__":
    main()
