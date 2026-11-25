"""Nguồn cấu hình nhãn dùng chung cho toàn hệ thống."""

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}

LABELS = [ID_TO_LABEL[label_id] for label_id in sorted(ID_TO_LABEL)]
