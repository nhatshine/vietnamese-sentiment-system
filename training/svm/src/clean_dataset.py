"""
clean_dataset.py - Bước 3-4: làm sạch + lọc + loại trùng.

Đầu vào : data/merged_raw.csv   (text, label, source, label_orig)
Đầu ra  : data/cleaned.csv      (text, label, source, label_orig, text_raw)
          data/clean_report.txt (báo cáo để kiểm tra)

Chạy:  python src/clean_dataset.py
(Đặt preprocess.py cùng thư mục src/.)
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import _SHORTCODE, TOK_NUM, clean_text, shortcode_to_token  # noqa: E402

# ======================= CẤU HÌNH =======================
INPUT = Path("data/merged_raw.csv")
OUTPUT = Path("data/cleaned.csv")
REPORT = Path("data/clean_report.txt")

MAX_WORDS = 256        # bỏ câu quá dài sau khi làm sạch (Transformer cũng khó xử lý)
MIN_WORDS = 1          # câu 1 từ như "ngon", "thích" là hợp lệ -> giữ
# Câu trùng NHƯNG khác nhãn: "drop" = bỏ hết các bản (khuyến nghị)
CONFLICT_POLICY = "drop"
KEEP_PUNCT = False     # xem docstring của clean_text
# ========================================================

LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    LINES.append(str(s))


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df.rename(columns={"text": "text_raw"})
    n0 = len(df)
    out(f"Đọc {n0} dòng từ {INPUT}")

    # ---- 1. Làm sạch ----
    df["text"] = df["text_raw"].map(lambda x: clean_text(x, keep_punct=KEEP_PUNCT))

    # ---- 2. Lọc ----
    drop_log = {}
    words = df["text"].str.split()
    n_words = words.str.len()

    only_num = words.map(lambda w: len(w) > 0 and set(w) <= {TOK_NUM})
    mask_empty = (n_words < MIN_WORDS) | only_num
    drop_log["rỗng / chỉ còn số sau khi làm sạch"] = int(mask_empty.sum())
    df = df[~mask_empty].copy()

    n_words = df["text"].str.split().str.len()
    mask_long = n_words > MAX_WORDS
    drop_log[f"quá dài (> {MAX_WORDS} từ)"] = int(mask_long.sum())
    out("\nSố câu quá dài bị bỏ theo nguồn / nhãn gốc:")
    out(df[mask_long].groupby(["source", "label"]).size().to_string() or "(không có)")
    df = df[~mask_long].copy()

    # ---- 3. Loại trùng (trên text ĐÃ làm sạch) ----
    grp = df.groupby("text")["label"].nunique()
    conflict_texts = set(grp[grp > 1].index)
    mask_conf = df["text"].isin(conflict_texts)
    out(f"\nSố câu (đã làm sạch) trùng nhưng KHÁC nhãn: {len(conflict_texts)} "
        f"({int(mask_conf.sum())} dòng)")
    out("Ví dụ:")
    for t in list(conflict_texts)[:8]:
        sub = df[df["text"] == t]
        out(f"   {t[:80]!r}: nhãn {sorted(sub['label'].unique().tolist())}, "
            f"nguồn {sorted(sub['source'].unique().tolist())}")
    if CONFLICT_POLICY == "drop":
        drop_log["trùng nhưng khác nhãn (bỏ hết)"] = int(mask_conf.sum())
        df = df[~mask_conf].copy()

    # trùng cùng nhãn: giữ 1 bản
    cross = df.groupby("text")["source"].nunique()
    out(f"Câu giống nhau xuất hiện ở >= 2 nguồn (cùng nhãn): {int((cross > 1).sum())}")
    dup = df.duplicated(subset=["text"], keep="first")
    drop_log["trùng cùng nhãn (giữ 1 bản)"] = int(dup.sum())
    df = df[~dup].copy()

    # ---- 4. Báo cáo ----
    out("\n" + "=" * 60)
    out("TÓM TẮT")
    out("=" * 60)
    out(f"Trước: {n0} dòng | Sau: {len(df)} dòng ({len(df) / n0 * 100:.1f}%)")
    out("Số dòng bị loại theo lý do:")
    for k, v in drop_log.items():
        out(f"   {k}: {v}")

    out("\nSố dòng theo nguồn:")
    out(df["source"].value_counts().to_string())
    out("\nPhân bố nhãn theo nguồn (0=Neg, 1=Neu, 2=Pos):")
    out(pd.crosstab(df["source"], df["label"]).to_string())
    out("\nPhân bố nhãn theo nguồn (% mỗi nguồn):")
    out(pd.crosstab(df["source"], df["label"], normalize="index").mul(100).round(1).to_string())
    out("\nTỉ lệ mỗi nguồn trong TỪNG nhãn (%): phát hiện nhãn bị 'một nguồn' chi phối")
    out(pd.crosstab(df["label"], df["source"], normalize="index").mul(100).round(1).to_string())

    df["n_words"] = df["text"].str.split().str.len()
    out("\nSố từ sau làm sạch theo nguồn:")
    out(df.groupby("source")["n_words"].describe(percentiles=[.5, .95, .99]).round(1).to_string())

    # shortcode emoji trong dữ liệu gốc và token chúng được đổi thành
    codes = Counter(c for t in df["text_raw"].astype(str) for c in _SHORTCODE.findall(t.lower()))
    out("\nTop 25 shortcode emoji trong dữ liệu gốc -> token (KIỂM TRA xem phân loại đúng chưa):")
    for code, c in codes.most_common(25):
        tok = shortcode_to_token(code).strip() or "(bỏ)"
        out(f"   {code:45s} {c:6d} -> {tok}")

    out("\n10 ví dụ trước/sau mỗi nguồn:")
    for s in sorted(df["source"].unique()):
        out(f"\n[{s}]")
        sub = df[df["source"] == s]
        for _, r in sub.sample(min(10, len(sub)), random_state=42).iterrows():
            out(f"   TRƯỚC: {str(r['text_raw'])[:110]!r}")
            out(f"   SAU  : {r['text'][:110]!r}  (label={r['label']})")

    # ---- 5. Lưu ----
    df = df.drop(columns=["n_words"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label", "source", "label_orig", "text_raw"]].to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    out(f"\n>>> Đã lưu {OUTPUT} ({len(df)} dòng) và {REPORT}")


if __name__ == "__main__":
    main()