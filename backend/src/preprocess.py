"""
preprocess.py - hàm làm sạch văn bản DÙNG CHUNG cho cả nhóm
(train SVM, train Transformer, và lúc dự đoán/inference).

Các quy tắc đều dựa trên kết quả EDA của merged_raw.csv:
  - neu_esc, vsfc: đã lowercase, dấu câu đã tách rời (" ."), không có slang.
  - hieu_sa: chữ gốc (hoa/thường), có URL, shortcode emoji (":red_heart:"),
    HTML entity (&amp;), ký tự kéo dài, viết tắt (ko, dc, đc, j, vs...).
=> Phải đưa 3 nguồn về CÙNG một "phong cách", nếu không model sẽ học cách viết
   của từng nguồn (ví dụ: có dấu "!" => hieu_sa) thay vì học cảm xúc.

Dùng:
    from preprocess import clean_text
    clean_text("Ko hiểu bài, quááááá chán :(")   # -> "không hiểu bài quá chán emoneg"
"""
import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# Từ điển viết tắt / slang. CHỈ gồm các từ có trong EDA và nghĩa khá chắc chắn.
# Có thể bổ sung thêm sau khi xem danh sách token chỉ có ở hieu_sa.
# ---------------------------------------------------------------------------
SLANG = {
    # phủ định -> "không" (rất quan trọng, đừng bỏ)
    "ko": "không", "k": "không", "hok": "không", "khg": "không", "kh": "không",
    "hông": "không", "hổng": "không",
    # các từ thông dụng
    "dc": "được", "đc": "được", "dk": "được",   # lưu ý: "dk" đôi khi là "điều kiện"
    "j": "gì", "ntn": "như thế nào", "vs": "với",
    "r": "rồi", "rùi": "rồi", "roi": "rồi",
    "oke": "ok", "okê": "ok",
    "mn": "mọi người", "ae": "anh em",
    "bt": "bình thường", "bth": "bình thường",
    "cx": "cũng", "cũg": "cũng",
    "wa": "quá", "wá": "quá",
    "thik": "thích", "iu": "yêu",
    "mik": "mình", "mk": "mình", "tui": "tôi", "bít": "biết", "ng": "người",
}

# Từ khóa để phân loại shortcode emoji (":red_heart:") thành cảm xúc dương/âm.
# Kiểm tra NEG trước POS (vì "broken_heart" chứa "heart").
EMO_NEG_KEYS = (
    "broken_heart", "angry", "pout", "rage", "thumbs_down", "cry", "sob", "sad",
    "disappoint", "unamused", "worried", "confound", "weary", "tired", "sick",
    "vomit", "nauseat", "anguish", "fear", "frown", "persever", "expressionless",
    "rolling_eyes", "pensive", "steam",
)
EMO_POS_KEYS = (
    "heart", "kiss", "smil", "grin", "laugh", "joy", "thumbs_up", "clap", "ok_hand",
    "star", "sparkl", "blush", "sunglasses", "party", "tada", "muscle", "pray",
    "love", "wink", "relieved", "yum",
)

# Token thay thế (viết liền, không dấu gạch dưới để không bị xóa như dấu câu)
TOK_POS, TOK_NEG, TOK_NUM = "emopos", "emoneg", "xnum"

# ---------------------------------------------------------------------------
# Các regex (biên dịch sẵn)
# ---------------------------------------------------------------------------
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_URL = re.compile(r"(?:https?:/*|www\.)\S+", re.I)          # cả "http:wwwlazada..." bị lỗi
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_MENTION = re.compile(r"@\w+")
_SHORTCODE = re.compile(r":[a-z][a-z0-9_]{2,}:")             # ":red_heart:"
_EMO_POS_TXT = re.compile(r"(?<!\w)(?::-?\)+|=\)+|:d|\^_\^|<3)(?!\w)")
_EMO_NEG_TXT = re.compile(r"(?<!\w)(?::-?\(+|:'\(|t_t|-_-)(?!\w)")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U0000FE0F\u263a]"
)
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")       # "10k", "2700mah" được giữ nguyên
_REPEAT = re.compile(r"([^\W\d_])\1{2,}")                    # "quááááá" -> "quá"
_PUNCT = re.compile(r"[^\w\s]|_")
_PUNCT_KEEP_QM = re.compile(r"[^\w\s!?]|_")
_MULTI_NUM = re.compile(rf"(?:\b{TOK_NUM}\b\s*){{2,}}")
_SPACES = re.compile(r"\s+")


def shortcode_to_token(code: str) -> str:
    """Đổi một shortcode emoji (có dấu ':') thành token cảm xúc, hoặc '' nếu không rõ."""
    name = code.strip(":")
    if any(k in name for k in EMO_NEG_KEYS):
        return f" {TOK_NEG} "
    if any(k in name for k in EMO_POS_KEYS):
        return f" {TOK_POS} "
    return " "


def clean_text(text, keep_punct: bool = False) -> str:
    """
    Làm sạch một câu tiếng Việt về dạng thống nhất cho mọi nguồn.

    keep_punct=False (mặc định): xóa toàn bộ dấu câu.
    keep_punct=True: giữ lại '!' và '?' (tách thành token riêng). Chỉ nên dùng nếu
    cả nhóm thống nhất, vì neu_esc/vsfc gần như không có hai dấu này.

    KHÔNG xóa dấu tiếng Việt và KHÔNG xóa từ phủ định (không, chẳng, chưa, chả...).
    """
    if not isinstance(text, str):
        return ""

    t = html.unescape(text)                       # &amp; -> &
    t = unicodedata.normalize("NFC", t)           # thống nhất Unicode (168 dòng hieu_sa lệch)
    t = _ZERO_WIDTH.sub("", t)                    # ký tự vô hình \u200b
    t = t.lower()

    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _MENTION.sub(" ", t)

    # emoji: shortcode trước, rồi emoticon dạng chữ, cuối cùng là emoji unicode
    t = _SHORTCODE.sub(lambda m: shortcode_to_token(m.group(0)), t)
    t = _EMO_POS_TXT.sub(f" {TOK_POS} ", t)
    t = _EMO_NEG_TXT.sub(f" {TOK_NEG} ", t)
    t = _EMOJI.sub(" ", t)

    t = _NUMBER.sub(f" {TOK_NUM} ", t)            # số đứng riêng -> token (giá, ngày, thông số)
    t = _REPEAT.sub(r"\1", t)                     # ký tự chữ lặp >= 3 lần -> 1 lần

    if keep_punct:
        t = _PUNCT_KEEP_QM.sub(" ", t)
        t = re.sub(r"([!?])", r" \1 ", t)
    else:
        t = _PUNCT.sub(" ", t)

    # chuẩn hóa viết tắt theo từng token
    t = " ".join(SLANG.get(tok, tok) for tok in t.split())

    t = _MULTI_NUM.sub(f"{TOK_NUM} ", t)          # "xnum xnum xnum" -> "xnum"
    t = _SPACES.sub(" ", t).strip()
    return t


if __name__ == "__main__":
    demo = [
        "Ko hiểu bài , quááááá chán :(",
        "Thật tuyệt vời. Tôi đã dùng rất nhiều sản phẩm của Xiaomi &amp; rất hài lòng",
        "chất lượng sản phẩm tuyệt vời :red_heart::red_heart: giao nhanh",
        "giá hạ nhiệt rồi lazada đang bán 820000đ http:wwwlazadavn.com/abc vòng đeo tay",
        "Giá 700k, bác nào dc thì mua, ngày 28 9 2023 !!! ok",
        "thầy dạy rất dễ hiểu , nhiệt tình .",
    ]
    for d in demo:
        print(repr(d), "\n   ->", repr(clean_text(d)))