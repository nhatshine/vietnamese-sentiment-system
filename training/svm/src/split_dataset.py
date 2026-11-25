"""
split_dataset.py - Bước 5: chia Train / Val / Test CHUNG cho cả nhóm.

Đầu vào : data/cleaned.csv  (đã làm sạch + loại trùng bởi clean_dataset.py)
Đầu ra  : data/final/train.csv, val.csv, test.csv   (cột: text, label, source)
          data/final/data_report.md                 (thống kê để bàn giao)

Chia stratify theo (source x label) để mỗi tập giữ cùng phân bố nguồn và nhãn.
Chạy:  python src/split_dataset.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ======================= CẤU HÌNH =======================
INPUT = Path("data/cleaned.csv")
OUT_DIR = Path("data/final")
SEED = 42
VAL_SIZE = 0.10
TEST_SIZE = 0.10
# ========================================================

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def dist_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Số dòng và % theo `by` (label hoặc source)."""
    n = df[by].value_counts().sort_index()
    return pd.DataFrame({"n": n, "%": (n / len(df) * 100).round(1)})


def main() -> None:
    df = pd.read_csv(INPUT)[["text", "label", "source"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    assert set(df["label"].unique()) <= {0, 1, 2}
    assert df["text"].is_unique, "Còn câu trùng! Hãy chạy lại clean_dataset.py"

    strata = df["source"] + "_" + df["label"].astype(str)
    counts = strata.value_counts()
    print("Số dòng mỗi tầng (nguồn_nhãn):")
    print(counts.to_string())
    if counts.min() < 10:
        raise ValueError("Có tầng quá nhỏ (<10 dòng), không chia stratify được.")

    # Chia test trước, rồi chia val từ phần còn lại
    trainval, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=strata, random_state=SEED
    )
    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    strata_tv = trainval["source"] + "_" + trainval["label"].astype(str)
    train, val = train_test_split(
        trainval, test_size=val_rel, stratify=strata_tv, random_state=SEED
    )

    splits = {"train": train, "val": val, "test": test}

    # ---- Kiểm tra rò rỉ: không có câu nào nằm ở 2 tập ----
    names = list(splits)
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = set(splits[names[i]]["text"]) & set(splits[names[j]]["text"])
            assert not overlap, f"Rò rỉ giữa {names[i]} và {names[j]}: {len(overlap)} câu"
    assert sum(len(s) for s in splits.values()) == len(df)
    print("\nKiểm tra rò rỉ: OK (không có câu nào xuất hiện ở 2 tập)")

    # ---- Lưu ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        s.reset_index(drop=True).to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # ---- Báo cáo ----
    md = ["# Báo cáo dữ liệu (Người 1)\n"]
    md.append("## Quy ước\n")
    md.append("- Cột: `text`, `label`, `source`. Văn bản đã qua `preprocess.clean_text`.")
    md.append("- Nhãn: 0 = Negative, 1 = Neutral, 2 = Positive.")
    md.append(f"- Chia: {int((1 - VAL_SIZE - TEST_SIZE) * 100)}/{int(VAL_SIZE * 100)}/"
              f"{int(TEST_SIZE * 100)}, stratify theo (source x label), random_state={SEED}.")
    md.append("- Đã loại trùng (theo văn bản đã làm sạch) và bỏ câu trùng nhưng khác nhãn.")
    md.append("- Không có câu nào xuất hiện ở hơn một tập (đã kiểm tra).\n")

    md.append("## Kích thước\n")
    md.append("```")
    md.append(pd.Series({k: len(v) for k, v in splits.items()}, name="số dòng").to_string())
    md.append("```\n")

    md.append("## Phân bố nhãn (%) theo tập\n```")
    md.append(pd.DataFrame({k: v["label"].value_counts(normalize=True).sort_index().mul(100).round(1)
                            for k, v in splits.items()}).rename(index=LABEL_NAMES).to_string())
    md.append("```\n")

    md.append("## Phân bố nguồn (%) theo tập\n```")
    md.append(pd.DataFrame({k: v["source"].value_counts(normalize=True).mul(100).round(1)
                            for k, v in splits.items()}).to_string())
    md.append("```\n")

    md.append("## Số dòng theo nguồn x nhãn (từng tập)\n")
    for k, v in splits.items():
        md.append(f"**{k}**\n```")
        md.append(pd.crosstab(v["source"], v["label"]).rename(columns=LABEL_NAMES).to_string())
        md.append("```\n")

    md.append("## Độ dài (số từ) theo tập\n```")
    md.append(pd.DataFrame({k: v["text"].str.split().str.len().describe(percentiles=[.5, .95, .99])
                            for k, v in splits.items()}).round(1).to_string())
    md.append("```\n")

    md.append("## Lưu ý khi huấn luyện và báo cáo\n")
    md.append("- Nhãn Neutral chủ yếu đến từ `neu_esc`; nguồn có thể là đường tắt để đoán nhãn. "
              "Hãy báo cáo điểm (Macro-F1) **riêng từng nguồn** ngoài điểm tổng.")
    md.append("- Token đặc biệt do `clean_text` tạo ra: `emopos`, `emoneg` (emoji), `xnum` (số).")
    md.append("- Test chỉ dùng một lần ở cuối. Mọi tuning (C, ngưỡng, ensemble weight) dùng `val`.")

    (OUT_DIR / "data_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    print(f"\n>>> Đã lưu train/val/test + data_report.md vào {OUT_DIR}")


if __name__ == "__main__":
    main()
