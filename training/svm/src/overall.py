"""
EDA nhanh cho data/merged_raw.csv (chạy TRƯỚC khi viết preprocess.py).

Mục đích: biết dữ liệu thật sự có những vấn đề gì (URL, emoji, slang, tách từ bằng "_",
trùng lặp, nhãn mâu thuẫn...) để viết quy tắc làm sạch cho sát.

Chạy:  python src/eda.py
       python src/eda.py --input data/merged_raw.csv --report data/eda_report.txt
Kết quả in ra màn hình và lưu vào file báo cáo (hãy dán file này cho Claude đọc).
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

LINES: list[str] = []


def out(s: str = "") -> None:
    """In ra màn hình đồng thời lưu vào báo cáo."""
    print(s)
    LINES.append(str(s))


def title(s: str) -> None:
    out("\n" + "=" * 70)
    out(s)
    out("=" * 70)


# ---------------- Các mẫu regex cần đếm ----------------
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U0000FE0F]"
)
PATTERNS = {
    "URL": re.compile(r"https?://\S+|www\.\S+", re.I),
    "Email": re.compile(r"\S+@\S+\.\S+"),
    "Mention (@user)": re.compile(r"@\w+"),
    "Hashtag (#tag)": re.compile(r"#\w+"),
    "Emoji": EMOJI,
    "Kéo dài ký tự (aaa)": re.compile(r"(\w)\1{2,}", re.U),
    "Nối từ bằng _ (sinh_viên)": re.compile(r"[^\W\d_]+_[^\W\d_]+", re.U),
    "Dấu câu tách rời (' .', ' ,')": re.compile(r"\s[.,!?;:]"),
    "Có chữ số": re.compile(r"\d"),
    "Viết HOA toàn từ (>=3 ký tự)": re.compile(r"\b[A-ZÀ-Ỹ]{3,}\b"),
    "Lặp dấu câu (!!!, ???, ...)": re.compile(r"([!?.])\1{1,}"),
    "Thực thể HTML (&amp; ...)": re.compile(r"&\w+;"),
    "Ký tự điều khiển/xuống dòng": re.compile(r"[\r\n\t]"),
    "Nhiều khoảng trắng liên tiếp": re.compile(r"\s{2,}"),
}

# Từ phủ định: tuyệt đối không được xóa khi làm sạch
NEGATIONS = ["không", "chẳng", "chưa", "chả", "đâu", "ko", "k", "hok", "hổng", "khg", "kh"]

# Một số viết tắt/slang hay gặp để đếm thử (chỉ là gợi ý, xem thêm bảng token ngắn bên dưới)
SLANG_CANDIDATES = [
    "ko", "k", "hok", "hông", "khg", "kh", "dc", "đc", "đk", "dk", "j", "gì", "ntn", "vs",
    "r", "rùi", "roi", "ok", "oke", "okê", "mn", "ae", "bn", "bt", "bth", "cx", "cũg",
    "vl", "vkl", "đm", "dm", "cc", "clm", "wa", "wá", "quá", "thik", "thích", "iu", "yêu",
    "mik", "mk", "mình", "tui", "t", "m", "bít", "biết", "ng", "người", "sv", "gv",
]


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "0%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/merged_raw.csv")
    ap.add_argument("--report", default="data/eda_report.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["text"] = df["text"].astype(str)
    sources = sorted(df["source"].unique())

    # ---------- 1. Tổng quan ----------
    title("1. TỔNG QUAN")
    out(f"Số dòng: {len(df)} | Cột: {df.columns.tolist()}")
    out(f"Null: {df.isna().sum().to_dict()}")
    out("\nSố dòng theo nguồn:")
    out(df["source"].value_counts().to_string())

    # ---------- 2. Phân bố nhãn ----------
    title("2. PHÂN BỐ NHÃN (0=Neg, 1=Neu, 2=Pos)")
    out("Tổng:")
    out(df["label"].value_counts(normalize=True).sort_index().mul(100).round(1).to_string())
    out("\nTheo nguồn (số dòng):")
    out(pd.crosstab(df["source"], df["label"]).to_string())
    out("\nTheo nguồn (% mỗi nguồn):")
    out(pd.crosstab(df["source"], df["label"], normalize="index").mul(100).round(1).to_string())
    out("\n-> Nếu một lớp chiếm < ~15% hoặc nguồn nào thiếu hẳn một lớp: ghi lại để "
        "dùng class_weight / báo cáo.")

    # ---------- 3. Độ dài câu ----------
    title("3. ĐỘ DÀI CÂU")
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()
    out("Số từ theo nguồn:")
    out(df.groupby("source")["n_words"].describe(percentiles=[.05, .5, .95, .99]).round(1).to_string())
    out("\nSố ký tự theo nguồn:")
    out(df.groupby("source")["n_chars"].describe(percentiles=[.5, .95, .99]).round(0).to_string())
    for k in (1, 2, 3):
        n = int((df["n_words"] <= k).sum())
        out(f"Câu <= {k} từ: {n} ({pct(n, len(df))})")
    out("\n10 câu NGẮN nhất (<=2 từ) ngẫu nhiên:")
    short = df[df["n_words"] <= 2]
    for _, r in short.sample(min(10, len(short)), random_state=42).iterrows():
        out(f"  [{r['source']}] label={r['label']}: {r['text']!r}")
    out("\n3 câu DÀI nhất:")
    for _, r in df.nlargest(3, "n_words").iterrows():
        out(f"  [{r['source']}] {r['n_words']} từ, label={r['label']}: {r['text'][:150]!r}...")

    # ---------- 4. Các mẫu cần xử lý ----------
    title("4. CÁC MẪU CẦN XỬ LÝ (số câu chứa mẫu / % mỗi nguồn)")
    rows = []
    for name, pat in PATTERNS.items():
        has = df["text"].str.contains(pat, regex=True)
        row = {"mẫu": name, "TỔNG": f"{int(has.sum())} ({pct(int(has.sum()), len(df))})"}
        for s in sources:
            m = df["source"] == s
            row[s] = pct(int((has & m).sum()), int(m.sum()))
        rows.append(row)
    out(pd.DataFrame(rows).to_string(index=False))

    # Unicode chưa chuẩn NFC
    not_nfc = df["text"].map(lambda t: unicodedata.normalize("NFC", t) != t)
    out(f"\nText chưa ở dạng Unicode NFC: {int(not_nfc.sum())} ({pct(int(not_nfc.sum()), len(df))})")
    for s in sources:
        m = df["source"] == s
        out(f"   {s}: {int((not_nfc & m).sum())}")

    # Ví dụ cho từng mẫu
    out("\nVí dụ cho từng mẫu (tối đa 3 câu):")
    for name, pat in PATTERNS.items():
        sub = df[df["text"].str.contains(pat, regex=True)]
        if len(sub) == 0:
            continue
        out(f"\n  -- {name} --")
        for _, r in sub.sample(min(3, len(sub)), random_state=42).iterrows():
            out(f"     [{r['source']}] {r['text'][:140]!r}")

    # ---------- 5. Trùng lặp ----------
    title("5. TRÙNG LẶP")
    dup_exact = df.duplicated(subset=["text"], keep=False)
    out(f"Câu trùng y hệt (tính cả bản gốc): {int(dup_exact.sum())} dòng "
        f"({pct(int(dup_exact.sum()), len(df))})")
    out(f"Số dòng thừa nếu giữ 1 bản mỗi câu: {int(df.duplicated(subset=['text']).sum())}")

    # Trùng sau khi chuẩn hóa nhẹ (lowercase + bỏ ký tự không phải chữ/số)
    norm = df["text"].str.lower().str.replace(r"[\W_]+", " ", regex=True).str.strip()
    out(f"Số dòng thừa sau chuẩn hóa nhẹ (lowercase, bỏ dấu câu): "
        f"{int(norm.duplicated().sum())}")

    # Trùng xuyên nguồn
    g = df.assign(norm=norm).groupby("norm")["source"].nunique()
    cross = g[g > 1]
    out(f"Câu xuất hiện ở >= 2 nguồn khác nhau: {len(cross)}")
    if len(cross):
        out("   Ví dụ:")
        for t in cross.index[:5]:
            out(f"     {t[:100]!r} -> {sorted(df.loc[norm == t, 'source'].unique())}")

    # Nhãn mâu thuẫn
    lab = df.assign(norm=norm).groupby("norm")["label"].nunique()
    conflict = lab[lab > 1]
    out(f"\nCâu trùng nhưng KHÁC nhãn: {len(conflict)}")
    for t in conflict.index[:8]:
        sub = df[norm == t]
        out(f"     {t[:100]!r} -> nhãn {sorted(sub['label'].unique())}, "
            f"nguồn {sorted(sub['source'].unique())}")

    # ---------- 6. Từ phủ định ----------
    title("6. TỪ PHỦ ĐỊNH (KHÔNG ĐƯỢC XÓA KHI LÀM SẠCH)")
    tokens_by_src = {
        s: Counter(w for t in df.loc[df["source"] == s, "text"].str.lower() for w in t.split())
        for s in sources
    }
    rows = []
    for w in NEGATIONS:
        row = {"từ": w}
        for s in sources:
            row[s] = tokens_by_src[s].get(w, 0)
        rows.append(row)
    out(pd.DataFrame(rows).to_string(index=False))

    # ---------- 7. Token phổ biến ----------
    title("7. TOKEN PHỔ BIẾN THEO NGUỒN")
    for s in sources:
        out(f"\n[{s}] Top 30 token:")
        out("   " + ", ".join(f"{w}({c})" for w, c in tokens_by_src[s].most_common(30)))

    title("8. TOKEN NGẮN (<=3 ký tự) PHỔ BIẾN: ỨNG VIÊN VIẾT TẮT/SLANG")
    out("(Đọc danh sách này để quyết định từ điển slang; bỏ qua các từ bình thường)")
    for s in sources:
        short_tokens = Counter({w: c for w, c in tokens_by_src[s].items()
                                if len(w) <= 3 and w.isalpha()})
        out(f"\n[{s}] Top 40:")
        out("   " + ", ".join(f"{w}({c})" for w, c in short_tokens.most_common(40)))

    out("\nTần suất các viết tắt/slang đã biết (theo nguồn):")
    rows = []
    for w in SLANG_CANDIDATES:
        row = {"từ": w}
        total = 0
        for s in sources:
            c = tokens_by_src[s].get(w, 0)
            row[s] = c
            total += c
        if total > 0:
            rows.append(row)
    out(pd.DataFrame(rows).to_string(index=False) if rows else "(không thấy)")

    # ---------- 9. Mẫu ngẫu nhiên ----------
    title("9. 10 CÂU NGẪU NHIÊN MỖI NGUỒN (đọc bằng mắt)")
    for s in sources:
        sub = df[df["source"] == s]
        out(f"\n[{s}]")
        for _, r in sub.sample(min(10, len(sub)), random_state=7).iterrows():
            out(f"   label={r['label']}: {r['text'][:140]!r}")

    # ---------- Lưu báo cáo ----------
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n>>> Đã lưu báo cáo: {args.report}")


if __name__ == "__main__":
    main()