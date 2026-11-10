# LAB08 - Pretrained Transformers and BERT Fine-Tuning

Thực hiện theo `../W8.pdf`; lưu `lab8/` theo yêu cầu người dùng.

## Bản mã nguồn trên GitHub

Repository chứa nội dung thư mục `lab8` ở thư mục gốc. Dataset tải về, weights
pretrained và checkpoint `.pt` được giữ cục bộ, không đưa vào Git. Sau khi clone,
chạy `python download_assets.py` để tải model/Yelp, chuẩn bị IMDb cache như bên
dưới, rồi chạy `python -u run_experiments.py` để tạo lại checkpoint. Các lệnh có
tiền tố `lab8/` bên dưới áp dụng khi đứng ở thư mục cha; khi đứng trong repository,
bỏ tiền tố đó. `W8.pdf` ở thư mục cha và `lab7/` không nằm trong repository này.

## Kết quả và trạng thái

Đã chạy **11 cấu hình × 4 epoch** trên IMDb: train 20000,
validation 5000, test 25000. GPU: NVIDIA GeForce RTX 4060 Laptop GPU.
Primary fine-tuned BERT/T128: accuracy **0.8169**,
precision **0.8099**, recall **0.8282**,
F1 **0.8190**. Scratch cùng kiến trúc: F1 **0.8158**;
frozen: F1 **0.6607**. Có đánh giá domain shift trên 4000 Yelp test
reviews và phân tích confidence. Số liệu đều từ lần chạy thật.

**Chưa phải bản nộp cá nhân hoàn tất:** theo W8 mục 24, sinh viên còn tự trình
bày MLM calculation, prediction trước experiment, giải thích pretraining/
fine-tuning, phân tích kết quả, error analysis và reflection 300-500 từ.
Các file cung cấp tài liệu tham chiếu, dữ liệu thực và câu hỏi hỗ trợ, đánh dấu
rõ phần chưa có câu trả lời cá nhân. Không bịa prediction sau khi xem kết quả.

## Chạy lại

Từ thư mục chứa W8.pdf:

```powershell
python -m pip install -r lab8/requirements.txt
python -u lab8/run_experiments.py
python lab8/make_reports.py
python lab8/pretrained_classification.py --text "This movie is not good."
```

`run_experiments.py` ghi đè kết quả hiện tại khi chạy lại. Dùng cache IMDb
Arrow hiện có của máy ở ~/.cache/huggingface/datasets/stanfordnlp___imdb.
Model và Yelp parquet đã nằm trong lab8 nên training không cần mạng.
Nếu cần tải lại model/Yelp: `python lab8/download_assets.py` (cần internet).
Downloader lấy revision hiện tại rồi pin trong manifest; để tái lập lần này,
giữ nguyên assets và kiểm chứng SHA256 trong assets_manifest.json.

Trên máy mới thiếu IMDb, tải bằng datasets về cache người dùng hoặc workspace,
rồi đặt Arrow cache đúng vị trí script đọc (xem load_data). Không có fallback
dữ liệu giả nếu thiếu cache. CPU được hỗ trợ nhưng chậm hơn; memory GPU ghi
rỗng nếu chạy CPU. Mở hai notebook bằng JupyterLab/VS Code để tương tác; cài
thêm jupyterlab/ipykernel nếu cần. Notebook dùng output đã lưu, có công tắc
RUN_TRAINING=False để tránh train lại ngoài ý muốn.

## Cấu hình tái lập

- Model/tokenizer: `google/bert_uncased_L-2_H-128_A-2`, revision `30b0a37ccaaa32f332884b96992754e246e48c5f`;
  uncased WordPiece, CLS contextual state + Linear(128,2), 2 classes.
- Random initialization baseline: cùng BERT config và tokenizer, weights ngẫu nhiên.
- IMDb train/val split seed42 stratified 80/20, giữ official 25000 test; cùng
  indices LAB7; nested subsets 10/25/50/100% từ training only.
- Max length=128 mặc định, kể cả CLS/SEP; length experiment 64/128/256.
- Batch 64; AdamW; learning rate=0.0001 ở mọi run; weight decay=0.01;
  epochs=4; clip gradient=1.0; không scheduler; seed42.
- Frozen encoder eval/no_grad, chỉ head trainable; full tuning train encoder+head.
- Best validation CE chọn epoch từng run. Primary pretrained/100%/T128 chọn
  trước chạy. Test không dùng chọn checkpoint. Single seed, chưa tune LR riêng.
- Training time chỉ tổng train epochs, synchronize CUDA, bỏ validation/test,
  tải/tokenization; frozen không cache features. GPU peak=PyTorch allocated
  train+validation, không phải tổng nvidia-smi memory. Pretraining cost không đo.
- Yelp official test subset stratified4000, seed42, chưa train hoặc tune trên
  Yelp; positive/negative mapping khớp IMDb. Domain indices lưu riêng.

## Tệp

| Tệp | Nội dung |
|---|---|
| calculations.md | Phép tính MLM tham chiếu, tokenization thật, quy ước counts/mask |
| prediction.md / prediction_verification.md | Prediction cá nhân chưa cung cấp, bảng evidence sau chạy |
| tokenization.ipynb | Token IDs, special tokens, masks, sentence pair, MLM inference |
| pretrained_classification.py | Encoder, custom head, frozen mode, CLI inference |
| fine_tuning.ipynb | Setup, training tùy chọn, kiểm tra, bảng thí nghiệm, inference |
| results.csv / history/ / run.log | 11 cấu hình và log/loss thật |
| experiment_report.md | So sánh scratch/pretrained, frozen, data, length, domain, confidence |
| data_efficiency.png | Accuracy/F1 cho cả pretrained và scratch ở 4 mức dữ liệu |
| confusion_matrix.png / confidence_analysis.png / learning_curves.png | Biểu đồ từ dự đoán/loss thật |
| error_analysis.md / error_examples.json | 3 FP, 3 FN, 2 sai tự tin cao và bằng chứng |
| reflection.md | Dàn ý và Q1-Q6, phần reflection cá nhân 300-500 từ chưa cung cấp |
| assets_manifest.json / environment.json / dataset_provenance.json | Revision, hash, versions, cấu hình |
| split_indices.json / domain_indices.json | Split/subsets để tái lập |
| model/ / data/ | Weights pretrained, vocabulary và Yelp test parquet |
| best_pretrained.pt / best_frozen.pt / predictions/ | Best checkpoint và probability/label thật |
| frozen_verification.json / validation.json / notebook_validation.json | Bằng chứng kiểm tra |

## AI-assisted code

- Tool: OpenAI Codex, Python/PyTorch/Transformers/datasets/NumPy/scikit-learn/Matplotlib.
- What was generated: code, notebooks, plots, tài liệu kỹ thuật tham chiếu và
  các khung bài cá nhân trong lab8.
- What was modified: tự viết classification head/frozen logic, thiết kế 11
  cấu hình, nested balanced subsets, SEP-safe truncation, confidence buckets,
  đánh giá IMDb -> Yelp; chưa có chỉnh sửa cá nhân của sinh viên.
- What was verified: chạy 11 cấu hình thật, kiểm tra frozen weights không đổi,
  tính lại metrics/confusion matrix từ probability, split/subset không rò rỉ,
  checkpoint/source/assets hashes; thực thi code cells của hai notebook trong
  cùng namespace bằng Python (không giả execution count/output).

## Nguồn

- [Google BERT Miniatures](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2).
- [BERT API](https://huggingface.co/docs/transformers/model_doc/bert).
- [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb).
- [Yelp polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity).
