# Calculation - tài liệu tham chiếu có AI hỗ trợ

Theo W8 mục 24, sinh viên cần tự trình bày phép tính MLM và giải thích bằng lời
của mình. Các số sau là kết quả kiểm chứng, không phải bản tính tay cá nhân.

## Exercise 1: MLM cross-entropy

Với một vị trí được mask và token thật là `eats`, loss dùng natural logarithm:

| Xác suất token đúng | Phép tính | Loss tham chiếu |
|---|---|---|
| 0.70 | -ln(0.70) | 0.356674944 |
| 0.95 | -ln(0.95) | 0.051293294 |
| 0.01 | -ln(0.01) | 4.605170186 |

Xác suất token đúng càng cao thì loss càng thấp. Cross-entropy tương ứng negative
log-likelihood của nhãn token, phù hợp bài toán phân phối xác suất trên vocabulary.
Chỉ vị trí được chọn để dự đoán đóng góp MLM loss. Đề viết tổng trên các vị trí
mask; framework có thể lấy mean trên số vị trí hợp lệ. Một vị trí thì hai cách
giống nhau. Không tính tổng loss cho mọi token không được chọn làm mục tiêu.

Notebook `tokenization.ipynb` kiểm chứng phép tính bằng NumPy/PyTorch và chạy
MLM inference thật với BERT pretrained trên câu `The cat [MASK] fish.`. Các xác
suất 0.70/0.95/0.01 ở bài tính là giả thiết đề, không phải output BERT đo được.

## Exercise 2: tokenization

Tokenizer: `BertTokenizerFast`, vocabulary uncased WordPiece đi cùng
`google/bert_uncased_L-2_H-128_A-2`. Kết quả thực tế được lưu trong
`tokenization_results.json`; không đoán cách tách từ trước khi chạy.

Với `The cats are playing.`:

```text
Không padding:
tokens = [CLS] the cats are playing . [SEP]
IDs    = 101   1996 8870 2024 2652 1012 102
mask   = 1     1    1    1    1    1    1

Pad đến 12:
tokens = [CLS] the cats are playing . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]
mask   = 1     1    1    1    1    1    1     0     0     0     0     0
```

- Words theo whitespace: 4 (`playing.` còn dấu chấm).
- Content WordPiece tokens: 5, bao gồm dấu chấm tách riêng.
- Special tokens hợp lệ: 2, CLS/SEP; tổng trước padding: 7.
- Padding tokens: 5; tổng tensor length: 12.
- Continuation pieces có tiền tố `##`: 0 trong chính câu này. Không được suy
  rằng không có subword vocabulary; 5 content token đều là token WordPiece.
- `token_type_ids` đều 0 cho một sequence. Pair input có thể chứa segment 0/1.

`input_ids` tra cứu embedding; `attention_mask` dùng 1 cho nội dung/special token
hợp lệ, 0 cho PAD. Đây là quy ước Hugging Face, khác mask bool True=PAD ở LAB7.
`token_type_ids` nhận diện segment của sentence pair, không phải attention mask.

## Special tokens

| Token | Chức năng |
|---|---|
| [CLS] | Hidden state ở vị trí đầu dùng cho classification head |
| [SEP] | Kết thúc/phân tách sequence |
| [PAD] | Đồng bộ độ dài batch; phải được mask |
| [MASK] | Token che trong MLM, không dùng thay cho attention mask |
| [UNK] | Nội dung không biểu diễn được bằng vocabulary |

Classification head trong bài là Linear(hidden_size=128, classes=2) trên
`last_hidden_state[:,0]`; đây là CLS contextual state, không phải mean pooling
hay BERT pooler. CrossEntropyLoss nhận logits trực tiếp; softmax chỉ để lấy
probability khi evaluation/inference.

## Câu hỏi sinh viên tự giải thích

1. Vì sao pretraining khác training from scratch và fine-tuning?
2. Vì sao frozen encoder vẫn tạo được feature cho một head mới?
3. Khi full fine-tuning, những tensor nào thay đổi? Frozen có tensor nào thay đổi?
4. Contextual embedding khác static Word2Vec ở đâu?
5. Attention mask giúp gì khi các input trong batch dài ngắn khác nhau?
