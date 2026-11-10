"""Create W8 reports from completed experiment outputs, then execute notebooks."""
from pathlib import Path
import csv
import json
import re

ROOT=Path(__file__).resolve().parent


def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')


def table(rows,cols):
    def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+\
           '\n'.join('| '+' | '.join(fmt(r.get(c,'')) for c in cols)+' |' for r in rows)


def main():
    rows=list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8')))
    if len(rows)!=11:raise RuntimeError('Need all 11 completed runs before generating final reports')
    for r in rows:
        for k in ['fraction','train_seconds','val_loss','test_loss','accuracy','precision','recall','f1','peak_allocated_MiB']:
            r[k]=float(r[k]) if r[k] else None
    lookup={r['run']:r for r in rows}
    primary=lookup['pretrained_finetune_p100_t128']
    scratch=lookup['scratch_finetune_p100_t128']
    frozen=lookup['pretrained_frozen_p100_t128']
    data=[r for r in rows if r['mode']=='finetune' and r['max_length']=='128']
    lengths=[lookup[f'pretrained_finetune_p100_t{t}'] for t in [64,128,256]]
    env=load('environment.json');manifest=load('assets_manifest.json')
    domain=load('domain_results.json');buckets=load('confidence_buckets.json')
    curves=[]
    for r in rows:
        h=load(f'history/{r["run"]}.json')
        curves.append({'run':r['run'],'best_epoch':r['best_epoch'],'train_first':h[0]['train_loss'],
                       'train_last':h[-1]['train_loss'],'best_val':min(x['val_loss'] for x in h),
                       'val_last':h[-1]['val_loss']})
    write('experiment_report.md',f'''
# W8: báo cáo bằng chứng thực nghiệm

Tài liệu kỹ thuật có AI hỗ trợ, từ 11 lượt chạy thật. Các kết luận diễn giải cá
nhân theo W8 mục 24, Q1-Q6 và reflection cần sinh viên tự viết. Không giả kết
quả và không gán các nhận xét AI thành phát biểu của sinh viên.

## Thiết kế

BERT-Tiny: 2 layers, hidden 128, 2 heads, FFN 512. Model/tokenizer:
`{manifest['model_id']}`; revision `{manifest['model_revision']}`.
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

{table([scratch,primary],['initialization','parameters','accuracy','precision','recall','f1','train_seconds'])}

Chênh lệch F1 pretrained - scratch: {primary['f1']-scratch['f1']:+.4f} trên lần
chạy này. Cả hai có cùng số tham số; pretrained thêm tri thức từ giai đoạn
pretraining ngoài IMDb. Thời gian bảng chỉ gồm downstream training, không tính
chi phí pretraining lịch sử hay tải model/tokenize. Không suy từ accuracy thành
khẳng định model hiểu mọi cấu trúc ngôn ngữ.

## Experiment 2: frozen và full fine-tuning

{table([frozen,primary],['mode','parameters','trainable_parameters','accuracy','f1','train_seconds'])}

Frozen chỉ cập nhật head; encoder ở eval mode, requires_grad=False và no_grad
trong forward. Encoder không cache features nên thời gian frozen vẫn gồm
encoder forward mỗi batch. Kiểm tra `frozen_verification.json` xác nhận encoder
weights không đổi, số tham số encoder trainable bằng 0. Full fine-tuning cập
nhật encoder và head. Giữ cùng LR để kiểm soát; frozen head có thể hưởng lợi
từ lịch train/LR riêng, chưa được tune ở bài này.

## Experiment 3: lượng dữ liệu

{table(data,['initialization','fraction','train_samples','accuracy','f1','train_seconds'])}

`data_efficiency.png` có đủ đường accuracy và F1 cho cả pretrained/scratch trên
10/25/50/100%. Subset phân tầng, cân bằng, lồng nhau và dùng cùng indices cho
hai chế độ. Validation/test không đổi. Mỗi run khởi tạo lại độc lập, không tiếp
tục từ checkpoint subset nhỏ. Số epoch cố định nhưng tổng optimizer steps tăng
theo số mẫu, vì vậy ảnh hưởng lượng dữ liệu và tổng bước train chưa tách rời.

Sinh viên cần dùng **toàn bộ đường cong**, không chọn một con số đơn lẻ để
trả lời Q1. Một seed không cho confidence interval hay bảo đảm monotonicity.

## Experiment 4: maximum sequence length

{table(lengths,['max_length','accuracy','f1','train_seconds','peak_allocated_MiB'])}

T=64/128/256 là tổng tokens gồm CLS/SEP. Head và dữ liệu giữ nguyên, chỉ mức
truncation thay đổi. Tensor cache 256 được cắt đúng và khôi phục SEP tại biên;
notebook đối chiếu với tokenizer trực tiếp. Peak memory là
torch.cuda.max_memory_allocated trong train+validation; không phải RAM CPU hay
tổng bộ nhớ GPU báo bởi nvidia-smi. CPU run sẽ ghi rỗng memory thay vì bịa số.

Dense attention có T×T quan hệ nên tăng T làm tăng compute/memory; backend
SDPA có thể tránh materialize toàn bộ scores, vì vậy peak GPU thực đo không
buộc tăng đúng 4 lần khi T gấp đôi. Chỉ so sánh thời gian downstream cùng máy.

## Experiment 5: domain shift

{table(domain,['domain','n','accuracy','precision','recall','f1'])}

Train chỉ IMDb (movie reviews). Không train/tune trên Yelp. Đánh giá Yelp là
4000 mẫu cố định, phân tầng từ official test của `{manifest['domain_dataset']}`,
revision `{manifest['domain_revision']}`. Dữ liệu gồm business/service/restaurant
reviews; nhãn đều ánh xạ 0=negative/1=positive. Lưu sample indices riêng.
Khác domain còn đi kèm khác độ dài/style/phân bố nhãn và cách gán nhãn, nên
chênh lệch metric không cô lập nguyên nhân duy nhất. Không giả định out-of-domain
luôn phải kém hơn in-domain. Pretrained không bảo đảm hiệu năng mọi domain.

## Confidence analysis

Confidence = max softmax(logits), không phải probability đã calibration.
Ngưỡng định trước: low [0.5,0.7), medium [0.7,0.9), high [0.9,1].

{table(buckets,['bucket','n','errors','mean_confidence','accuracy'])}

Hai dự đoán sai tự tin nhất được chọn từ toàn test bằng max softmax và lưu
trong error_examples.json; tách biệt với 3 FP + 3 FN chọn theo index.
Histogram và bucket chart ở confidence_analysis.png là mô tả calibration ban
đầu, không phải bằng chứng đã calibration hay uncertainty toàn diện.

## Learning curves

{table(curves,['run','best_epoch','train_first','train_last','best_val','val_last'])}

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
''')
    errors=load('error_examples.json')
    parts=['# Error analysis: bằng chứng test thật\n',
           'W8 mục 24 yêu cầu sinh viên tự error analysis. Bảng dưới lưu 3 false positives, '
           '3 false negatives và 2 ca sai tự tin nhất; có thể trùng index giữa các nhóm. '
           'Các dấu hiệu tự động và hypothesis là gợi ý kiểm chứng, không phải kết luận nguyên nhân. '
           'Sinh viên cần tự bổ sung phân loại và giải thích.\n']
    summary=[]
    for e in errors:
        lower=e['text'].lower()
        signs=[];hypotheses=[]
        if re.search(r"\b(not|never|no|isn't|wasn't|don't|didn't)\b",lower):
            signs.append('negation cue');hypotheses.append('Kiểm tra scope phủ định và sentiment trong đoạn được giữ')
        if re.search(r'\b(but|although|however|despite)\b',lower):
            signs.append('contrast cue');hypotheses.append('Kiểm tra mệnh đề tương phản và nhãn tổng thể')
        if e['truncated']:
            signs.append('truncation');hypotheses.append('Sentiment quan trọng có thể ở phần bị cắt; so sánh T256')
        if e['unknown_tokens']:
            signs.append('UNK');hypotheses.append('Kiểm tra token lạ, không mặc định mọi rare word đều thành UNK')
        if not signs:
            signs=['context / label ambiguity cần xem thủ công']
            hypotheses=['Đối chiếu ngữ cảnh giữ lại, sarcasm và tính rõ ràng của nhãn trước khi gán nguyên nhân']
        summary.append({'index':e['test_index'],'group':e['selection'],'true':e['true'],'pred':e['pred'],
                        'confidence':e['confidence'],'error type (candidate)':', '.join(signs),
                        'hypothesis (unverified)':' ; '.join(hypotheses)})
        parts.append(f"## Test #{e['test_index']} - {e['selection']}\n\n"
                     f"True={e['true']}, Pred={e['pred']}, confidence={e['confidence']:.6f}, "
                     f"P(positive)={e['positive_probability']:.6f}.\n\n"
                     f"Tokens={e['full_token_count']}, giữ={e['retained_token_count']}, "
                     f"truncated={e['truncated']}, UNK={e['unknown_tokens']}.\n\n"
                     'Text gốc:\n\n> '+e['text'].replace('\n',' ')+'\n\n'
                     'Token input thực tế:\n\n```text\n'+' '.join(e['retained_tokens'])+'\n```\n\n'
                     'Candidate types: '+', '.join(signs)+'.\n\n'
                     'Hypothesis chưa kiểm chứng: '+'; '.join(hypotheses)+'.\n\n'
                     '**Sinh viên tự phân tích:** kiểm tra negation/sarcasm/long-range/ambiguous sentiment/'
                     'domain expression/rare word/tokenization/context/label ambiguity; '
                     'trích evidence cụ thể, nêu cách kiểm tra giả thuyết. Confidence cao có giải thích '
                     'được vì sao sai không?\n')
    parts.insert(2,table(summary,['index','group','true','pred','confidence','error type (candidate)','hypothesis (unverified)'])+'\n')
    write('error_analysis.md','\n'.join(parts))
    write('prediction_verification.md',f'''
# Đối chiếu prediction

Prediction gốc chưa được cung cấp. Không giả prediction hoặc chấm Đúng/Sai sau
khi đã xem kết quả. Sinh viên dùng bằng chứng để tự đối chiếu câu trả lời gốc.

| # | Prediction | Evidence | Đúng/Sai | Giải thích cá nhân |
|---|---|---|---|---|
| 1 | Chưa cung cấp | Scratch F1={scratch['f1']:.4f}; pretrained F1={primary['f1']:.4f} | Chưa xác định | Sinh viên bổ sung |
| 2 | Chưa cung cấp | Frozen F1={frozen['f1']:.4f}; fine-tuned F1={primary['f1']:.4f} | Chưa xác định | Sinh viên bổ sung |
| 3 | Chưa cung cấp | data_efficiency.png và bảng 10/25/50/100% | Chưa xác định | Sinh viên bổ sung |
''')
    write('reflection.md','''
# Reflection cá nhân - cần sinh viên viết

W8 mục 24 không cho AI thay reflection. Viết **300-500 từ** bằng lời của mình:

> Sau LAB07 đã biết xây dựng Transformer. Sau LAB08, tại sao thực tế thường
> không bắt đầu bằng train Transformer từ đầu?

Bắt buộc đề cập data, pretrained representation, computational cost, downstream
task, fine-tuning và limitation/domain shift. Trích số liệu thật từ results.csv
và domain_results.json; phân biệt observation -> evidence -> interpretation.

## Dàn ý hỗ trợ

1. Bối cảnh LAB7 và bài toán IMDb, giải thích pretraining/fine-tuning bằng lời
   của bạn; phân biệt tokenizer đã học với encoder random initialization.
2. Đối chiếu đường 10/25/50/100% của cả hai model, số liệu frozen/full tuning;
   nêu giới hạn do cùng epoch nhưng khác số optimizer step.
3. So sánh chi phí downstream với chi phí pretraining lịch sử không đo ở lab;
   xem T64/128/256 có đáng chi phí tăng thêm trong kết quả của bạn không.
4. Phân tích domain shift IMDb -> Yelp, một lỗi tự tin cao, các giới hạn về
   truncation/tokenization/labels. Không suy từ accuracy thành hiểu ngôn ngữ.
5. Hướng thực nghiệm tiếp theo và cầu nối sequence classification -> token
   classification/NER ở LAB09.

## Q1-Q6 bắt buộc (sinh viên tự trả lời)

1. Model nào hưởng lợi hơn khi ít training data? Dùng cả đường cong, không chỉ 10%.
2. Nếu pretrained tốt hơn scratch, kết quả chứng minh điều gì và chưa chứng minh gì?
3. Nếu frozen gần bằng fine-tuned, điều đó có thể có ý nghĩa gì?
4. Nếu fine-tuning hơn frozen nhiều, điều đó gợi ý điều gì? Có confound từ LR không?
5. Nếu performance giảm trên domain khác, những nguyên nhân nào cần kiểm tra?
6. Nếu tăng length gần như không đổi performance nhưng tốn thời gian, có nên tăng tiếp?

## Individual learning check

Tự trả lời 12 câu W8 mục 27: pretraining/fine-tuning; MLM; tokenizer; subword;
CLS; attention mask; frozen encoder; tham số thay đổi khi fine-tuning; dữ liệu
downstream nhỏ; domain shift; chi phí sequence length; confidence/correctness.

Nội dung reflection cá nhân: **chưa được cung cấp**. Không coi dàn ý này là bài
reflection 300-500 từ đã hoàn thành.
''')
    write('README.md',f'''
# LAB08 - Pretrained Transformers and BERT Fine-Tuning

Thực hiện theo `../W8.pdf`; lưu `lab8/` theo yêu cầu người dùng.

## Kết quả và trạng thái

Đã chạy **11 cấu hình × {env['epochs']} epoch** trên IMDb: train 20000,
validation 5000, test 25000. GPU: {env['hardware']}.
Primary fine-tuned BERT/T128: accuracy **{primary['accuracy']:.4f}**,
precision **{primary['precision']:.4f}**, recall **{primary['recall']:.4f}**,
F1 **{primary['f1']:.4f}**. Scratch cùng kiến trúc: F1 **{scratch['f1']:.4f}**;
frozen: F1 **{frozen['f1']:.4f}**. Có đánh giá domain shift trên 4000 Yelp test
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

- Model/tokenizer: `{manifest['model_id']}`, revision `{manifest['model_revision']}`;
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
''')
    from build_notebooks import build_all
    build_all()
    print('Reports written; both notebooks executed successfully.')


if __name__=='__main__':main()
