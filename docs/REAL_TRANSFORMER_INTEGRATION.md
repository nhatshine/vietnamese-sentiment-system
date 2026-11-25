# Real Transformer và Real Ensemble – Bước 8

## Artifact và tính toàn vẹn

Nguồn: `C:/Users/Admin/train_Transformer.zip` do người dùng cung cấp.
Chỉ đưa các artifact inference vào `backend/models/transformer/`:

```text
best_transformer.pth
model.py
tokenizer.py
predict_transformer.py
vocab.json
config.json
label_map.json
preprocess.py
```

Không giải nén dataset, report, notebook hoặc script train/evaluate vào
project. Checkpoint, model.py, vocab, config, label map và preprocessing giữ
nguyên byte. Chỉ đổi ba dòng import trong tokenizer.py và predict_transformer.py
để chạy trong package backend; thuật toán tokenization/inference không đổi.

```text
SHA-256 best_transformer.pth:
60dd4f48f879b75a759161b9149235cae9f5c03ab4c5d08d4085074680f08118
SHA-256 vocab.json:
fad68bf01d7364545b14ca660ddb71e5ddc664b12d56e673f13dbb677db0f7b6
SHA-256 model.py:
82e39d1cf46a033054746bac6383817664a73b87378ca477cace38b525329783
```

## Kiến trúc thực tế và checkpoint

Encoder-only Transformer tự xây trong source model.py, không pretrained:

```text
vocab_size = 9169       max_len = 128
d_model = 128          num_heads = 4
num_layers = 2         d_ff = 256
dropout = 0.2          num_classes = 3
```

Giữ nguyên token embedding cộng **Learned Positional Embedding**
`nn.Embedding(128,128)`, hai block Pre-LayerNorm/residual, self-attention Q/K/V,
FFN GELU, final LayerNorm và head Dropout + Linear lấy CLS tại vị trí 0.
Không thay bằng sinusoidal positional encoding hoặc đổi hyperparameter.

Checkpoint thực tế là OrderedDict state_dict gồm 38 tensor, không chứa
optimizer metadata. Đã kiểm tra token embedding `(9169,128)`, position
embedding `(128,128)` và head weight `(3,128)`.

Wrapper `backend/src/real_transformer.py` tái sử dụng constructor gốc:
`torch.load(..., map_location="cpu", weights_only=True)`,
`load_state_dict(..., strict=True)`, `model.eval()`.
Resources được cache theo device; inference dùng `torch.inference_mode()`.
Không load checkpoint lại từng câu và không tạo gradient.

CPU là mặc định. Có thể truyền `device="cuda"` nếu CUDA có sẵn, nếu không
sẽ báo lỗi rõ ràng. Môi trường kiểm chứng là Python 3.13.13,
PyTorch **2.14.0+cpu**, NumPy 2.3.5, sklearn 1.9.1, joblib 1.6.0;
CUDA không có sẵn. Source requirements ghi torch>=2.2, không xác định chính
xác phiên bản torch khi train. Project pin release torch==2.14.0 đã kiểm chứng.
Không cài hoặc thay package Python toàn hệ thống trong bước này.

## Tokenizer, vocabulary và preprocessing

Tokenizer gốc là whitespace/syllable tokenizer. Source build_vocab dùng
TRAIN, min_freq=2, max_vocab_size=25000, tần suất giảm dần và thứ tự chữ cái
khi hòa; inference chỉ load vocab 9169 ID liên tục, không rebuild vocab.

```text
PAD <pad> = 0; UNK <unk> = 1; CLS <cls> = 2
```

`encode_text(raw_text)` gọi `tokenize()`; **tokenize đã gọi clean_text()**.
Giữ tối đa 127 token nội dung, thêm CLS ở đầu rồi PAD đến 128. Từ ngoài vocab
thành UNK. Mask bằng 1 cho CLS/token thật, bằng 0 cho PAD. Input IDs/mask là
torch.long, shape `(1,128)` khi inference một câu.

Preprocess.py trong ZIP trùng nguyên byte và SHA-256 với bản SVM dùng chung:

```text
20d8e57b3aa014f5c4cab733560971b5dea4aca813c0c7637dee7a0620fa5eae
```

Tokenizer chỉnh import để tái sử dụng `backend.src.preprocess.clean_text`.
Wrapper đối chiếu cả bản dùng chung và bản trong folder artifact với hash
config trước khi load, không ghi đè preprocessing SVM.

```text
Raw paragraph → Sentence Splitter → Raw sentence
→ Real Transformer predictor → encode_text()
→ clean_text() đúng một lần bên trong tokenizer
→ vocabulary / CLS / truncation / padding / mask
→ Transformer → logits → softmax
```

Không clean đoạn trước splitter; analyzer và wrapper không clean thêm trước
encode_text. Test kiểm tra mỗi model được gọi và preprocess đúng một lần cho
mỗi raw sentence theo đúng thứ tự.

## Label mapping và prediction contract

Đã đọc label_map.json: `0=negative`, `1=neutral`, `2=positive`. Wrapper đọc
từng cột theo mapping đó, kiểm tra đủ index, nhãn lowercase và không trùng;
không dựa vào thứ tự class ngầm định.

`predict_transformer(text: str, *, device="cpu") -> dict[str,float]`
trả đúng LABELS với Python float, softmax không round và tái sử dụng
validate_prediction_output() với tolerance 1e-5. Non-string gây TypeError;
raw rỗng/whitespace gây ValueError. Nonempty raw text chỉ có punctuation vẫn
theo tokenizer gốc với CLS/PAD, không tạo output giả.

## Predictor injection

```python
from backend.src.real_svm import predict_svm
from backend.src.real_transformer import predict_transformer
from backend.src.sentence_analyzer import analyze_sentence, analyze_sentences
from backend.src.document_aggregator import analyze_document

result = analyze_sentence(
    "Sản phẩm này rất tốt.", alpha=0.5,
    svm_predictor=predict_svm, transformer_predictor=predict_transformer,
)
sentences = analyze_sentences(
    "Camera rất đẹp. Pin yếu.",
    svm_predictor=predict_svm, transformer_predictor=predict_transformer,
)
document = analyze_document(
    "Camera rất đẹp. Pin yếu.", alpha=0.5,
    svm_predictor=predict_svm, transformer_predictor=predict_transformer,
)
```

Không truyền predictor vẫn mock; chỉ truyền một predictor thật vẫn là hybrid
integration test. Truyền cả hai mới dùng hai output thật cho Real Ensemble.
Giữ nguyên công thức Ensemble và ba phương pháp Document Aggregation.
Alpha 0.5 chỉ kiểm thử tích hợp, không phải alpha tối ưu.

Tính tay từ probability thật câu đầu, theo thứ tự negative/neutral/positive:

```text
SVM         = (0.003743699698, 0.003862124286, 0.992394176016)
Transformer = (0.000862488872, 0.000714108406, 0.998423457146)
0.5*SVM + 0.5*Transformer
            = (0.002303094285, 0.002288116346, 0.995408816581)
```

## Numerical consistency và demo

Unit test đối chiếu wrapper với source predictor CPU trong tolerance 1e-7.
Ngoài ra đã chạy source **chưa chỉnh sửa** từ ZIP trong process riêng, chỉ
đưa tám artifact inference vào thư mục tạm. Sai khác tuyệt đối lớn nhất trên
năm câu demo là **0.0**. Không chạy script train/evaluate hoặc dùng CSV.

Mỗi ô dưới là nhãn / confidence, số được định dạng lúc hiển thị:

| Input | Real SVM | Real Transformer | Real Ensemble |
|---|---|---|---|
| Sản phẩm này rất tốt. | positive / 0.992394176 | positive / 0.998423457 | positive / 0.995408817 |
| Tôi rất thất vọng vì sản phẩm quá tệ. | negative / 0.981958446 | negative / 0.992833614 | negative / 0.987396030 |
| Sản phẩm này bình thường. | neutral / 0.603974161 | negative / 0.465832800 | neutral / 0.490721627 |
| Camera rất đẹp nhưng pin quá yếu. | positive / 0.669723754 | positive / 0.862620831 | positive / 0.766172292 |
| Giao hàng chậm và đóng gói cẩu thả. | negative / 0.895132288 | negative / 0.947759390 | negative / 0.921445839 |

Không sửa nhãn câu thứ ba để làm đẹp demo. Đây không phải phép đo metric.

Đoạn `Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!` tạo ba câu;
cả ba phương án lần lượt dự đoán positive, negative, positive. Chín kết quả
aggregation đều positive, với confidence:

| Phương án | Majority Voting | Mean Probability | Confidence Weighted |
|---|---:|---:|---:|
| SVM | 0.666666667 | 0.660364731 | 0.677700517 |
| Transformer | 0.666666667 | 0.720698441 | 0.788866988 |
| Ensemble | 0.666666667 | 0.690531586 | 0.734440501 |

Majority confidence là tỷ lệ phiếu, không phải probability model.

```powershell
python -m unittest discover backend/tests
python -m backend.demo_real_transformer
```

Demo in đầy đủ probability của cả ba phương án trên năm câu và đoạn văn.
Sáu demo cũ tiếp tục chạy. Các 205 test cũ không sửa; test mới kiểm tra load,
tokenizer, CLS/PAD/UNK, mask/shape/dtype, cache, eval/inference mode,
preprocessing, label map, numerical consistency và pipeline hai model thật.

## Giới hạn hiện tại

CPU đã kiểm chứng, CUDA chưa chạy trên môi trường CPU này. Giữ nguyên các
giới hạn của splitter. Không retrain/fine-tune, thay checkpoint/vocab/kiến
trúc hoặc dataset. Chưa tuning alpha, đánh giá lại Macro-F1 trên Test, chọn
BEST_METHOD/BEST_AGGREGATION, khẳng định Ensemble tốt hơn hoặc xây FastAPI/React.
Dừng tại Bước 8.

## Danh sách thay đổi

Tạo `backend/src/real_transformer.py`, `backend/tests/test_real_transformer.py`,
`backend/demo_real_transformer.py`, tài liệu này và tám file artifact liệt kê
ở đầu tài liệu trong `backend/models/transformer/`.

Sửa `backend/src/sentence_analyzer.py`, `backend/src/document_aggregator.py`
để truyền predictor; thêm torch vào `backend/requirements.txt`. Đồng bộ
README.md và các tài liệu MODEL_INTEGRATION_CONTRACT.md,
SENTENCE_LEVEL_ANALYSIS.md, DOCUMENT_LEVEL_AGGREGATION.md,
REAL_SVM_INTEGRATION.md, WEIGHTED_ENSEMBLE.md, SENTENCE_SEGMENTATION.md.
Không sửa test cũ, công thức Ensemble/aggregation hoặc artifact SVM.
