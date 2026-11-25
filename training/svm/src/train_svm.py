"""
train_svm.py - Huấn luyện TF-IDF (Word + Char) + Linear SVM cho sentiment tiếng Việt 3 nhãn.

Quy trình:
  1. Đọc data/final/{train,val,test}.csv (cột: text, label, source).
  2. Tuning (vectorizer config x C x class_weight) - CHỈ dựa vào Validation (Macro-F1).
  3. Huấn luyện lại cấu hình tốt nhất trên TRAIN, bọc CalibratedClassifierCV để có xác suất
     (cần cho Ensemble). Không dùng val/test để huấn luyện => val sạch cho Người 3.
  4. Đánh giá trên Val (tổng + từng nguồn), phân tích lỗi, lưu model + báo cáo.
  5. Test chỉ được dùng khi chạy với cờ --final (làm MỘT LẦN khi đã chốt cấu hình).

Chạy:
    python train_svm.py            # tuning + đánh giá val, KHÔNG đụng test
    python train_svm.py --final    # như trên + đánh giá test (chạy 1 lần ở cuối)

Cần: pip install scikit-learn pandas numpy joblib
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================= CẤU HÌNH =======================
SEED = 42
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Các cấu hình vectorizer thử nghiệm (mỗi cấu hình phải fit lại TF-IDF nên chỉ để ít cấu hình)
VECTORIZER_CONFIGS = [
    {"name": "A", "word_ngram": (1, 2), "char_analyzer": "char_wb", "char_ngram": (2, 5)},
    {"name": "B", "word_ngram": (1, 3), "char_analyzer": "char", "char_ngram": (3, 5)},
]
C_GRID = [0.1, 0.3, 1.0, 3.0]
CLASS_WEIGHT_GRID = [None, "balanced"]
WORD_MAX_FEATURES = 300_000
CHAR_MAX_FEATURES = 400_000
# ========================================================


def find_data_dir(cli_value: str | None) -> Path:
    """Tìm thư mục data/final, không phụ thuộc thư mục đang đứng khi chạy lệnh."""
    here = Path(__file__).resolve().parent
    candidates = [Path(cli_value)] if cli_value else []
    candidates += [here / "data" / "final", here.parent / "data" / "final",
                   Path.cwd() / "data" / "final"]
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    raise FileNotFoundError(
        "Không tìm thấy data/final/train.csv. Đã thử:\n  " + "\n  ".join(map(str, candidates))
        + "\nDùng --data-dir để chỉ rõ."
    )


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{name}.csv")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def make_features(cfg: dict) -> FeatureUnion:
    """Word TF-IDF + Char TF-IDF ghép lại. Chỉ dùng lớp có sẵn của sklearn để pickle gọn."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg["word_ngram"],
        token_pattern=r"(?u)\b\w+\b",   # giữ cả token 1 ký tự (ạ, à, ơ...), mặc định sẽ bỏ
        min_df=2,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer=cfg["char_analyzer"],
        ngram_range=cfg["char_ngram"],
        min_df=3,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def make_svc(C: float, class_weight) -> LinearSVC:
    return LinearSVC(C=C, class_weight=class_weight, random_state=SEED, max_iter=5000)


def metrics_overall(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
    }


def metrics_by_source(y_true, y_pred, sources) -> pd.DataFrame:
    rows = []
    for s in sorted(set(sources)):
        m = sources == s
        r = metrics_overall(y_true[m], y_pred[m])
        r.update({"source": s, "n": int(m.sum())})
        rows.append(r)
    return pd.DataFrame(rows)[["source", "n", "accuracy", "macro_f1"]].round(4)


def source_only_baseline(train: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Mốc so sánh: chỉ biết nguồn rồi đoán nhãn phổ biến nhất của nguồn đó (học từ train)."""
    maj = train.groupby("source")["label"].agg(lambda s: s.value_counts().idxmax())
    pred = df["source"].map(maj).values
    return metrics_overall(df["label"].values, pred)


def top_word_features(union: FeatureUnion, svc: LinearSVC, k: int = 15) -> dict:
    names = union.get_feature_names_out()
    idx = np.array([i for i, n in enumerate(names) if n.startswith("word__")])
    out = {}
    for ci, cls in enumerate(svc.classes_):
        w = svc.coef_[ci][idx]
        top = np.argsort(w)[::-1][:k]
        out[int(cls)] = [names[idx[i]][len("word__"):] for i in top]
    return out


def predict_with_proba(pipe: Pipeline, texts: pd.Series):
    proba = pipe.predict_proba(texts.tolist())
    classes = np.asarray(pipe.classes_)
    pred = classes[proba.argmax(axis=1)]
    return pred, proba


def eval_block(title: str, pipe: Pipeline, df: pd.DataFrame, train: pd.DataFrame):
    """Đánh giá đầy đủ trên một tập, trả về (dòng markdown, dict metric, DataFrame lỗi)."""
    y = df["label"].values
    src = df["source"].values
    pred, proba = predict_with_proba(pipe, df["text"])
    ov = metrics_overall(y, pred)
    ov["log_loss"] = float(log_loss(y, proba, labels=LABELS))
    by_src = metrics_by_source(y, pred, src)
    base = source_only_baseline(train, df)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=LABELS),
                      index=[f"gốc:{LABEL_NAMES[i]}" for i in LABELS],
                      columns=[f"đoán:{LABEL_NAMES[i]}" for i in LABELS])
    rep = classification_report(y, pred, labels=LABELS,
                                target_names=[LABEL_NAMES[i] for i in LABELS],
                                digits=4, zero_division=0)

    res = df.copy()
    res["pred"] = pred
    res["conf"] = proba.max(axis=1)
    for i, c in enumerate(pipe.classes_):
        res[f"p_{LABEL_NAMES[int(c)]}"] = proba[:, i]
    res["n_words"] = res["text"].str.split().str.len()
    errors = res[res["label"] != res["pred"]].sort_values("conf", ascending=False)

    bucket = pd.cut(res["n_words"], [0, 3, 10, 30, 10_000], labels=["1-3 từ", "4-10", "11-30", ">30"])
    err_by_len = (res.assign(err=(res["label"] != res["pred"]), bucket=bucket)
                  .groupby("bucket", observed=True)["err"].agg(["size", "mean"]).round(4))
    err_by_len.columns = ["số câu", "tỉ lệ sai"]

    md = [f"## {title}\n",
          f"- Accuracy: **{ov['accuracy']:.4f}** | Macro-F1: **{ov['macro_f1']:.4f}** | "
          f"Log-loss: {ov['log_loss']:.4f}",
          f"- Mốc 'chỉ biết nguồn': Accuracy {base['accuracy']:.4f}, Macro-F1 {base['macro_f1']:.4f}\n",
          "### Báo cáo theo lớp\n```", rep, "```\n",
          "### Confusion matrix\n```", cm.to_string(), "```\n",
          "### Điểm theo nguồn\n```", by_src.to_string(index=False), "```\n",
          "### Tỉ lệ sai theo độ dài câu\n```", err_by_len.to_string(), "```\n"]
    return md, {"overall": ov, "baseline_source_only": base,
                "by_source": by_src.to_dict("records")}, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="đánh giá thêm trên TEST (chỉ chạy một lần khi đã chốt cấu hình)")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = find_data_dir(args.data_dir)
    root = data_dir.parents[1]
    models_dir, reports_dir = root / "models", root / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    print(f"Dữ liệu: {data_dir}\nModel -> {models_dir}\nBáo cáo -> {reports_dir}")

    train, val = load_split(data_dir, "train"), load_split(data_dir, "val")
    y_tr, y_va = train["label"].values, val["label"].values
    print(f"train={len(train)} val={len(val)}")

    # ---------------- 1. Tuning trên VAL ----------------
    results, best = [], {"score": -1.0}
    for cfg in VECTORIZER_CONFIGS:
        t0 = time.time()
        union = make_features(cfg)
        X_tr = union.fit_transform(train["text"])
        X_va = union.transform(val["text"])
        print(f"\n[Config {cfg['name']}] {X_tr.shape[1]} đặc trưng "
              f"(vectorize {time.time() - t0:.0f}s)")
        for cw in CLASS_WEIGHT_GRID:
            for C in C_GRID:
                svc = make_svc(C, cw).fit(X_tr, y_tr)
                pred = svc.predict(X_va)
                m = metrics_overall(y_va, pred)
                results.append({"config": cfg["name"], "C": C, "class_weight": str(cw),
                                "val_macro_f1": round(m["macro_f1"], 4),
                                "val_acc": round(m["accuracy"], 4)})
                print(f"   C={C:<4} cw={str(cw):9s} Macro-F1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f}")
                if m["macro_f1"] > best["score"]:
                    best = {"score": m["macro_f1"], "cfg": cfg, "C": C, "cw": cw,
                            "union": union, "svc": svc}
    res_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    print(f"\n>>> Tốt nhất: config {best['cfg']['name']}, C={best['C']}, "
          f"class_weight={best['cw']}, Val Macro-F1={best['score']:.4f}")

    top_feats = top_word_features(best["union"], best["svc"])

    # ---------------- 2. Huấn luyện bản cuối (có xác suất) ----------------
    print("\nHuấn luyện bản cuối + calibrate xác suất (CV 5 fold trên TRAIN)...")
    pipe = Pipeline([
        ("features", make_features(best["cfg"])),
        ("clf", CalibratedClassifierCV(
            make_svc(best["C"], best["cw"]), method="sigmoid",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))),
    ])
    pipe.fit(train["text"].tolist(), y_tr)

    # ---------------- 3. Đánh giá VAL ----------------
    md = ["# Báo cáo TF-IDF + Linear SVM (Người 1)\n",
          f"- Thời điểm: {datetime.now():%Y-%m-%d %H:%M} | scikit-learn {sklearn.__version__}",
          f"- Train {len(train)} / Val {len(val)} dòng",
          f"- Cấu hình: word n-gram {best['cfg']['word_ngram']}, "
          f"char ({best['cfg']['char_analyzer']}) {best['cfg']['char_ngram']}, "
          f"C={best['C']}, class_weight={best['cw']}, calibrate=sigmoid (cv=5)",
          "- Tuning chỉ dựa trên Validation (Macro-F1). Model huấn luyện CHỈ trên train.\n",
          "## Kết quả tuning (Validation, SVM thô chưa calibrate)\n```",
          res_df.to_string(index=False), "```\n"]

    val_md, val_metrics, val_errors = eval_block("Đánh giá trên VALIDATION (model đã calibrate)",
                                                 pipe, val, train)
    md += val_md

    md.append("## Từ nổi bật nhất mỗi lớp (trọng số SVM, đặc trưng word)\n```")
    for c in LABELS:
        md.append(f"{LABEL_NAMES[c]:9s}: " + ", ".join(top_feats[c]))
    md.append("```\n")

    cols = ["source", "label", "pred", "conf", "text"]
    md.append("## 15 câu sai với độ tự tin cao nhất (Validation)\n```")
    md.append(val_errors[cols].head(15).assign(text=lambda d: d["text"].str[:90]).to_string(index=False))
    md.append("```\n")
    val_errors.to_csv(reports_dir / "svm_val_errors.csv", index=False, encoding="utf-8-sig")

    # ---------------- 4. TEST (chỉ khi --final) ----------------
    meta = {
        "classes": [int(c) for c in pipe.classes_],
        "label_names": LABEL_NAMES,
        "best_params": {"config": best["cfg"]["name"], "word_ngram": list(best["cfg"]["word_ngram"]),
                        "char_analyzer": best["cfg"]["char_analyzer"],
                        "char_ngram": list(best["cfg"]["char_ngram"]),
                        "C": best["C"], "class_weight": best["cw"]},
        "val": val_metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.final:
        test = load_split(data_dir, "test")
        test_md, test_metrics, test_errors = eval_block(
            "Đánh giá trên TEST (dùng MỘT LẦN, không chỉnh model theo kết quả này)", pipe, test, train)
        md += test_md
        test_errors.to_csv(reports_dir / "svm_test_errors.csv", index=False, encoding="utf-8-sig")
        meta["test"] = test_metrics
        print(f"\nTEST: Macro-F1={test_metrics['overall']['macro_f1']:.4f} "
              f"Acc={test_metrics['overall']['accuracy']:.4f}")
    else:
        md.append("_Test chưa được đánh giá. Chạy `python train_svm.py --final` một lần khi đã chốt cấu hình._\n")

    # ---------------- 5. Lưu ----------------
    joblib.dump(pipe, models_dir / "svm_pipeline.joblib", compress=3)
    (models_dir / "svm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (reports_dir / "svm_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "\n".join(val_md[:4]))
    print(f"\n>>> Đã lưu: {models_dir / 'svm_pipeline.joblib'}")
    print(f">>> Báo cáo: {reports_dir / 'svm_report.md'}")


if __name__ == "__main__":
    main()
