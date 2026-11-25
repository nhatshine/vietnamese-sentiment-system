"""
Gộp 3 dataset sentiment tiếng Việt về định dạng thống nhất.

Quy ước nhãn chung: 0 = Negative, 1 = Neutral, 2 = Positive.
Bước này CHỈ gộp + chuẩn hóa nhãn, KHÔNG làm sạch văn bản.

Chạy:  python src/merge_datasets.py
"""
from pathlib import Path

import pandas as pd

# ===================== CẤU HÌNH (sửa cho khớp máy bạn) =====================
RAW = Path("data/raw")
OUT_DIR = Path("data")

# --- Hieu (vietnamese-SA-dataset): chỉ có train + test ---
HIEU_FILES = [RAW / "hieu_train.csv", RAW / "hieu_test.csv"]
HIEU_TEXT_COL, HIEU_LABEL_COL = "text", "label"          # <-- sửa theo df.columns

# --- UIT-VSFC: có train + validation + test ---
VSFC_FILES = [
    RAW / "vsfc_train.csv",
    RAW / "vsfc_validation.csv",
    RAW / "vsfc_test.csv",
]
VSFC_TEXT_COL, VSFC_LABEL_COL = "sentence", "sentiment"  # <-- kiểm tra lại theo df.columns

# --- NEU-ESC: có train + val + test ---
NEU_FILES = [
    RAW / "neu_esc_train.csv",
    RAW / "neu_esc_val.csv",
    RAW / "neu_esc_test.csv",
]
NEU_TEXT_COL, NEU_LABEL_COL = "text", "sentiment"

# Nhãn toxic (3) của NEU-ESC: "negative" = gộp vào Negative, "drop" = bỏ toxic
TOXIC_POLICY = "negative"

# --- Bảng quy đổi về 0 = Negative, 1 = Neutral, 2 = Positive ---
# VSFC gốc: 0 = neg, 1 = neu, 2 = pos  (giữ nguyên)
VSFC_MAP = {0: 0, 1: 1, 2: 2}

# NEU-ESC gốc: 0 = neu, 1 = pos, 2 = neg, 3 = toxic
NEU_MAP = {
    2: 0,                                              # neg   -> Negative
    0: 1,                                              # neu   -> Neutral
    1: 2,                                              # pos   -> Positive
    3: 0 if TOXIC_POLICY == "negative" else None,      # toxic -> Negative hoặc bỏ
}

# Hieu gốc: nhãn dạng chữ neg / neu / pos (không phân biệt hoa thường)
# Nếu nhãn của Hieu là số, thay bằng dict số tương ứng, ví dụ {0: 0, 1: 1, 2: 2}
HIEU_MAP = {
    "neg": 0, "negative": 0,
    "neu": 1, "neutral": 1,
    "pos": 2, "positive": 2,
}

# Giữ cột nhãn gốc (label_orig) trong file trung gian để truy vết / phân tích lỗi
KEEP_LABEL_ORIG = True
# ============================================================================


def load_concat(files: list[Path]) -> pd.DataFrame:
    """Đọc và gộp tất cả split gốc (train/val/test) của một tập."""
    frames = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {f}")
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def prep(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_map: dict,
    source: str,
) -> pd.DataFrame:
    """Giữ các cột cần thiết, quy đổi nhãn về 0/1/2 và gán source."""
    for col in (text_col, label_col):
        if col not in df.columns:
            raise KeyError(
                f"[{source}] Không có cột '{col}'. Các cột hiện có: {df.columns.tolist()}"
            )

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    n_before = len(out)

    # Bỏ dòng thiếu nhãn trước (NEU-ESC lưu nhãn dạng float nên có thể có NaN)
    n_nan = int(out["label"].isna().sum())
    out = out.dropna(subset=["label"]).copy()
    out["label_orig"] = out["label"]

    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].astype(int)         # 2.0 -> 2 cho khớp khóa dict
    else:
        out["label"] = out["label"].astype(str).str.strip().str.lower()

    out["label"] = out["label"].map(label_map)
    n_unmapped = int(out["label"].isna().sum())

    print(
        f"[{source}] {n_before} dòng gốc | {n_nan} thiếu nhãn | "
        f"{n_unmapped} không quy đổi được/bị bỏ (toxic drop hoặc nhãn lạ)"
    )
    if n_unmapped > 0:
        print(f"    Nhãn gốc không quy đổi được: "
              f"{out.loc[out['label'].isna(), 'label_orig'].value_counts().to_dict()}")

    out["source"] = source
    return out


def main() -> None:
    parts = [
        prep(load_concat(HIEU_FILES), HIEU_TEXT_COL, HIEU_LABEL_COL, HIEU_MAP, "hieu_sa"),
        prep(load_concat(VSFC_FILES), VSFC_TEXT_COL, VSFC_LABEL_COL, VSFC_MAP, "vsfc"),
        prep(load_concat(NEU_FILES), NEU_TEXT_COL, NEU_LABEL_COL, NEU_MAP, "neu_esc"),
    ]

    merged = pd.concat(parts, ignore_index=True)

    # Bỏ dòng không có nhãn hợp lệ hoặc text rỗng (KHÔNG loại trùng ở bước này)
    merged = merged.dropna(subset=["text", "label"]).copy()
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"] != ""]
    merged["label"] = merged["label"].astype(int)

    assert set(merged["label"].unique()) <= {0, 1, 2}, "Có nhãn ngoài 0/1/2!"

    # ---- Báo cáo kiểm tra ----
    print("\n=== Số dòng theo nguồn ===")
    print(merged["source"].value_counts())

    print("\n=== Phân bố nhãn theo nguồn (0=Neg, 1=Neu, 2=Pos) ===")
    print(pd.crosstab(merged["source"], merged["label"]))

    print("\n=== Phân bố nhãn gốc -> nhãn chung (truy vết) ===")
    print(pd.crosstab([merged["source"], merged["label_orig"]], merged["label"]))

    merged["n_words"] = merged["text"].str.split().str.len()
    print("\n=== Độ dài câu (số từ) theo nguồn ===")
    print(merged.groupby("source")["n_words"].agg(["min", "mean", "max"]).round(1))

    print("\n=== 5 dòng mẫu mỗi nguồn (đọc để kiểm tra nhãn có bị đảo không) ===")
    sample = merged.groupby("source").sample(5, random_state=42)
    for _, r in sample.iterrows():
        print(f"[{r['source']}] label={r['label']} (gốc={r['label_orig']}): {r['text'][:120]}")

    # ---- Lưu file ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["text", "label", "source"] + (["label_orig"] if KEEP_LABEL_ORIG else [])
    merged[cols].to_csv(OUT_DIR / "merged_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu {OUT_DIR / 'merged_raw.csv'}: {len(merged)} dòng, cột {cols}")


if __name__ == "__main__":
    main()