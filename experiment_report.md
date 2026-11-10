# W8: báo cáo bằng chứng thực nghiệm

Tài liệu kỹ thuật có AI hỗ trợ, từ 11 lượt chạy thật. Các kết luận diễn giải cá
nhân theo W8 mục 24, Q1-Q6 và reflection cần sinh viên tự viết. Không giả kết
quả và không gán các nhận xét AI thành phát biểu của sinh viên.

## Thiết kế

BERT-Tiny: 2 layers, hidden 128, 2 heads, FFN 512. Model/tokenizer:
`google/bert_uncased_L-2_H-128_A-2`; revision `30b0a37ccaaa32f332884b96992754e246e48c5f`.
Cả pretrained và scratch dùng cùng BERT architecture, tokenizer WordPiece,
max length 128 kể cả CLS/SEP, head Linear(128,2), split IMDb 20000/5000/25000,
batch 64, AdamW LR 1e-4, weight decay 0.01, 4 epochs, clip-grad 1.0, seed 42.
Scratch nghĩa là weights encoder ngẫu nhiên nhưng **vẫn dùng tokenizer đã học**
của BERT; không phải toàn pipeline đều được học từ 0 trên IMDb.

Không copy trực tiếp LAB7 vào so sánh kiểm soát: LAB7 có tokenizer, pooling,
architecture và lịch train khác. LAB7 chỉ là bối cảnh lịch sử. Main primary là
pretrained/full-data/T128, chọn trước chạy. Best epoch mỗi run chọn bằng
validation cross-entropy, không chọn bằng test. LR cố định kiểm soát một yếu tố,
nhưng chưa tối ưu riêng từng chế độ; kết quả không phải trần hiệu năng của scratch.

## Experiment 1: pretrained và scratch

| initialization | parameters | accuracy | precision | recall | f1 | train_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | 4369666 | 0.8161 | 0.8171 | 0.8146 | 0.8158 | 34.9974 |
| pretrained | 4369666 | 0.8169 | 0.8099 | 0.8282 | 0.8190 | 35.4287 |

Chênh lệch F1 pretrained - scratch: +0.0031 trên lần
chạy này. Cả hai có cùng số tham số; pretrained thêm tri thức từ giai đoạn
pretraining ngoài IMDb. Thời gian bảng chỉ gồm downstream training, không tính
chi phí pretraining lịch sử hay tải model/tokenize. Không suy từ accuracy thành
khẳng định model hiểu mọi cấu trúc ngôn ngữ.

## Experiment 2: frozen và full fine-tuning

| mode | parameters | trainable_parameters | accuracy | f1 | train_seconds |
| --- | --- | --- | --- | --- | --- |
| frozen | 4369666 | 258 | 0.6755 | 0.6607 | 14.8057 |
| finetune | 4369666 | 4369666 | 0.8169 | 0.8190 | 35.4287 |

Frozen chỉ cập nhật head; encoder ở eval mode, requires_grad=False và no_grad
trong forward. Encoder không cache features nên thời gian frozen vẫn gồm
encoder forward mỗi batch. Kiểm tra `frozen_verification.json` xác nhận encoder
weights không đổi, số tham số encoder trainable bằng 0. Full fine-tuning cập
nhật encoder và head. Giữ cùng LR để kiểm soát; frozen head có thể hưởng lợi
từ lịch train/LR riêng, chưa được tune ở bài này.

## Experiment 3: lượng dữ liệu

| initialization | fraction | train_samples | accuracy | f1 | train_seconds |
| --- | --- | --- | --- | --- | --- |
| pretrained | 0.1000 | 2000 | 0.7393 | 0.7411 | 5.0226 |
| pretrained | 0.2500 | 5000 | 0.7786 | 0.7648 | 8.9280 |
| pretrained | 0.5000 | 10000 | 0.7962 | 0.8002 | 17.7041 |
| pretrained | 1.0000 | 20000 | 0.8169 | 0.8190 | 35.4287 |
| scratch | 0.1000 | 2000 | 0.5429 | 0.6829 | 3.5638 |
| scratch | 0.2500 | 5000 | 0.7340 | 0.6803 | 8.7612 |
| scratch | 0.5000 | 10000 | 0.8024 | 0.7989 | 17.8792 |
| scratch | 1.0000 | 20000 | 0.8161 | 0.8158 | 34.9974 |

`data_efficiency.png` có đủ đường accuracy và F1 cho cả pretrained/scratch trên
10/25/50/100%. Subset phân tầng, cân bằng, lồng nhau và dùng cùng indices cho
hai chế độ. Validation/test không đổi. Mỗi run khởi tạo lại độc lập, không tiếp
tục từ checkpoint subset nhỏ. Số epoch cố định nhưng tổng optimizer steps tăng
theo số mẫu, vì vậy ảnh hưởng lượng dữ liệu và tổng bước train chưa tách rời.

Sinh viên cần dùng **toàn bộ đường cong**, không chọn một con số đơn lẻ để
trả lời Q1. Một seed không cho confidence interval hay bảo đảm monotonicity.

## Experiment 4: maximum sequence length

| max_length | accuracy | f1 | train_seconds | peak_allocated_MiB |
| --- | --- | --- | --- | --- |
| 64 | 0.7638 | 0.7609 | 26.0425 | 138.6763 |
| 128 | 0.8169 | 0.8190 | 35.4287 | 222.9575 |
| 256 | 0.8459 | 0.8434 | 61.9354 | 411.3960 |

T=64/128/256 là tổng tokens gồm CLS/SEP. Head và dữ liệu giữ nguyên, chỉ mức
truncation thay đổi. Tensor cache 256 được cắt đúng và khôi phục SEP tại biên;
notebook đối chiếu với tokenizer trực tiếp. Peak memory là
torch.cuda.max_memory_allocated trong train+validation; không phải RAM CPU hay
tổng bộ nhớ GPU báo bởi nvidia-smi. CPU run sẽ ghi rỗng memory thay vì bịa số.

Dense attention có T×T quan hệ nên tăng T làm tăng compute/memory; backend
SDPA có thể tránh materialize toàn bộ scores, vì vậy peak GPU thực đo không
buộc tăng đúng 4 lần khi T gấp đôi. Chỉ so sánh thời gian downstream cùng máy.

## Experiment 5: domain shift

| domain | n | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| IMDb movie reviews | 25000 | 0.8169 | 0.8099 | 0.8282 | 0.8190 |
| Yelp business reviews | 4000 | 0.7910 | 0.7483 | 0.8770 | 0.8076 |

Train chỉ IMDb (movie reviews). Không train/tune trên Yelp. Đánh giá Yelp là
4000 mẫu cố định, phân tầng từ official test của `fancyzhx/yelp_polarity`,
revision `bbf1c97a1f0cf005e5aded43839fd814654a1557`. Dữ liệu gồm business/service/restaurant
reviews; nhãn đều ánh xạ 0=negative/1=positive. Lưu sample indices riêng.
Khác domain còn đi kèm khác độ dài/style/phân bố nhãn và cách gán nhãn, nên
chênh lệch metric không cô lập nguyên nhân duy nhất. Không giả định out-of-domain
luôn phải kém hơn in-domain. Pretrained không bảo đảm hiệu năng mọi domain.

## Confidence analysis

Confidence = max softmax(logits), không phải probability đã calibration.
Ngưỡng định trước: low [0.5,0.7), medium [0.7,0.9), high [0.9,1].

| bucket | n | errors | mean_confidence | accuracy |
| --- | --- | --- | --- | --- |
| low_0.50_0.70 | 4896 | 1991 | 0.6027 | 0.5933 |
| medium_0.70_0.90 | 7396 | 1846 | 0.8123 | 0.7504 |
| high_0.90_1.00 | 12708 | 740 | 0.9624 | 0.9418 |

Hai dự đoán sai tự tin nhất được chọn từ toàn test bằng max softmax và lưu
trong error_examples.json; tách biệt với 3 FP + 3 FN chọn theo index.
Histogram và bucket chart ở confidence_analysis.png là mô tả calibration ban
đầu, không phải bằng chứng đã calibration hay uncertainty toàn diện.

## Learning curves

| run | best_epoch | train_first | train_last | best_val | val_last |
| --- | --- | --- | --- | --- | --- |
| pretrained_finetune_p10_t128 | 4 | 0.6758 | 0.4330 | 0.5242 | 0.5242 |
| pretrained_finetune_p25_t128 | 3 | 0.6448 | 0.3197 | 0.4614 | 0.4817 |
| pretrained_finetune_p50_t128 | 2 | 0.5944 | 0.2963 | 0.4292 | 0.4456 |
| pretrained_finetune_p100_t128 | 2 | 0.5310 | 0.2596 | 0.3819 | 0.4183 |
| scratch_finetune_p10_t128 | 4 | 0.7173 | 0.6859 | 0.6723 | 0.6723 |
| scratch_finetune_p25_t128 | 3 | 0.7071 | 0.2833 | 0.5258 | 0.5328 |
| scratch_finetune_p50_t128 | 3 | 0.7010 | 0.2120 | 0.4077 | 0.4554 |
| scratch_finetune_p100_t128 | 2 | 0.6015 | 0.1975 | 0.3699 | 0.4319 |
| pretrained_frozen_p100_t128 | 4 | 0.6883 | 0.6151 | 0.6056 | 0.6056 |
| pretrained_finetune_p100_t64 | 2 | 0.5874 | 0.3441 | 0.4718 | 0.4956 |
| pretrained_finetune_p100_t256 | 4 | 0.4778 | 0.1934 | 0.3403 | 0.3403 |

Ảnh learning_curves.png giữ train/validation loss từng epoch của cả 11 run.
Train loss giảm mà val cuối cao hơn best là dấu hiệu cần kiểm tra overfit;
không sửa log, không dùng test chọn epoch. 4 epochs không chứng minh đã hội tụ.

## Prediction verification và giới hạn

Prediction cá nhân chưa được cung cấp trước thí nghiệm; không thể chấm Đúng/Sai
một cách trung thực. Xem prediction_verification.md. Chưa đánh giá nhiều seed,
chưa tuning LR riêng, chưa đo catastrophic forgetting qua MLM trước/sau training.
Không kết luận forgetting chỉ từ domain shift. BERT nhỏ có giới hạn capacity,
WordPiece/ngôn ngữ, truncation, sarcasm, label ambiguity và computational cost.

## Nguồn

- [Google BERT Miniatures model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- [Hugging Face BERT API](https://huggingface.co/docs/transformers/model_doc/bert).
- [IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb).
- [Yelp polarity dataset](https://huggingface.co/datasets/fancyzhx/yelp_polarity).
