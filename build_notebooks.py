"""Fill scaffolded notebooks and execute ordinary Python cells with real outputs."""
from pathlib import Path
import base64
import contextlib
import io
import json
import os
import uuid

ROOT=Path(__file__).resolve().parent


class Notebook:
    def __init__(self,name):
        self.path=ROOT/name
        self.doc=json.loads(self.path.read_text(encoding='utf-8'))
        self.cells=[]

    def add(self,kind,source):
        cell={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        self.cells.append(cell)

    def md(self,source):self.add('markdown',source)
    def code(self,source):self.add('code',source)

    def image(self,name,caption):
        self.md(f'{caption}\n\n![{caption}](attachment:{name})')
        self.cells[-1]['attachments']={name:{'image/png':base64.b64encode((ROOT/name).read_bytes()).decode()}}

    def save_executed(self):
        scope={'__name__':'__notebook__'};count=0
        previous=Path.cwd()
        try:
            os.chdir(ROOT)
            for c in self.cells:
                if c['cell_type']!='code':continue
                count+=1
                output=io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(''.join(c['source']),f'{self.path.name}:cell{count}','exec'),scope)
                c['execution_count']=count
                c['outputs']=[{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(keepends=True)}] if output.getvalue() else []
        finally:
            os.chdir(previous)
        self.doc['cells']=self.cells
        self.path.write_text(json.dumps(self.doc,ensure_ascii=False,indent=1),encoding='utf-8')
        return count


SETUP='''from pathlib import Path
import sys, json, csv
import numpy as np
import torch
ROOT=Path.cwd() if (Path.cwd()/"pretrained_classification.py").exists() else Path.cwd()/"lab8"
assert (ROOT/"model/config.json").exists()
sys.path.insert(0,str(ROOT))
from pretrained_classification import get_tokenizer, load_classifier
torch.set_num_threads(4)
tokenizer=get_tokenizer()
print("Tokenizer:",type(tokenizer).__name__,"vocab:",len(tokenizer))
'''


def tokenization():
    n=Notebook('tokenization.ipynb')
    n.md('# LAB08 - Tokenization, attention mask và MLM\n\n'
         'Kiểm tra tokenizer thật của BERT-Tiny. Các phép tính là tham chiếu kiểm '
         'chứng có AI hỗ trợ; sinh viên tự trình bày MLM calculation theo W8 mục 24. '
         'Không đoán tokenization từ ví dụ minh họa trong đề.')
    n.code(SETUP)
    n.md('## Exercise 2: The cats are playing.\n\n'
         'So sánh words theo whitespace, content WordPiece tokens, continuation '
         'pieces (##) và special tokens. Không đồng nhất các cách đếm.')
    n.code('''text="The cats are playing."
batch=tokenizer([text,"A good movie."],padding="max_length",max_length=12,truncation=True,return_tensors="pt")
print("keys:",list(batch.keys()))
for key,value in batch.items(): print(key,value.tolist())
tokens=tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
print("tokens:",tokens)
active=[int(i) for i,m in zip(batch["input_ids"][0],batch["attention_mask"][0]) if m]
content=[i for i in active if i not in tokenizer.all_special_ids]
print("whitespace words:",len(text.split()),"content WordPiece:",len(content),"valid incl specials:",len(active),"padded length:",len(tokens))
print("continuation pieces:",sum(t.startswith("##") for t in tokens))
print("special tokens:",tokenizer.special_tokens_map)
assert batch["attention_mask"][0].tolist()==[1]*7+[0]*5
assert batch["input_ids"][0].tolist()==[101,1996,8870,2024,2652,1012,102,0,0,0,0,0]
''')
    n.md('`input_ids` là chỉ số vocabulary; `attention_mask`: 1=valid, 0=padding. '
         '`token_type_ids` đánh dấu segment cho sentence pair. [MASK] và attention '
         'mask có chức năng khác nhau. Một câu có thể không có continuation piece '
         '## nhưng vẫn được tokenize bằng subword vocabulary.')
    n.code('''pair=tokenizer("The movie is good.","I recommend it.")
print(tokenizer.convert_ids_to_tokens(pair["input_ids"]))
print("token_type_ids:",pair["token_type_ids"])
for text in ["unbelievably entertaining", "antidisestablishmentarianism", "NLP 😊"]:
    print(text,"->",tokenizer.tokenize(text))
''')
    n.md('## Exercise 1: kiểm chứng MLM loss\n\n'
         'Đây là ba xác suất giả định đề cho, không phải probability BERT. '
         'Với một masked position, tổng và mean trên vị trí mask bằng nhau.')
    n.code('''for p in [0.70,0.95,0.01]:
    print(f"P(correct)={p:.2f}: -ln(p)={-np.log(p):.9f}")
logits=torch.log(torch.tensor([[.10,.05,.70,.15]]))
loss=torch.nn.functional.cross_entropy(logits,torch.tensor([2]))
print("PyTorch CE target=eats:",loss.item())
assert abs(loss.item()+np.log(.70))<1e-6
''')
    n.md('## MLM inference trên checkpoint pretrained thật\n\n'
         'Chạy forward với MLM head từ model đã tải. Không fine-tune MLM và '
         'không gán probability đo được thành số giả định 0.70 của đề.')
    n.code('''from transformers import AutoModelForMaskedLM
mlm,loading=AutoModelForMaskedLM.from_pretrained(str(ROOT/"model"),local_files_only=True,output_loading_info=True)
assert not loading["missing_keys"], "MLM weights must come from the pretrained checkpoint"
mlm.eval()
masked=tokenizer("The cat [MASK] fish.",return_tensors="pt")
position=(masked["input_ids"][0]==tokenizer.mask_token_id).nonzero().item()
with torch.inference_mode(): probs=mlm(**masked).logits[0,position].softmax(-1)
values,ids=probs.topk(5)
print("top5:",[(tokenizer.convert_ids_to_tokens(int(i)),round(float(v),5)) for i,v in zip(ids,values)])
print("P(eats) thực đo:",probs[tokenizer.convert_tokens_to_ids("eats")].item())
del mlm
''')
    n.md('## Attention mask và invariance đối với padding\n\n'
         'Cùng câu được infer riêng và trong batch có câu dài; valid logits '
         'phải gần bằng nhau khi eval mode và mask đúng.')
    n.code('''model,config=load_classifier()
short="A good movie."
single=tokenizer([short],return_tensors="pt")
padded=tokenizer([short,"The acting is wonderful and the story is memorable."],padding=True,return_tensors="pt")
with torch.inference_mode():
    alone=model(**single)[0]
    together=model(**padded)[0]
print("max logits delta:",(alone-together).abs().max().item())
torch.testing.assert_close(alone,together,atol=1e-5,rtol=1e-5)
print("PASS padding invariance")
''')
    n.md('Tự giải thích bằng lời: tokenization liên hệ embedding ra sao; vì sao '
         'cần attention mask; subword giúp gì với rare word; tại sao CLS chỉ '
         'trở thành useful representation thông qua pretraining/task training. '
         'Đọc `calculations.md` và tự hoàn thiện phần cá nhân theo đề.')
    return n


def fine_tuning():
    n=Notebook('fine_tuning.ipynb')
    n.md('# LAB08 - Pretrained BERT fine-tuning\n\n'
         'Mục tiêu: so sánh pretrained/random-init với cùng kiến trúc; frozen/full '
         'tuning; data efficiency; sequence length; IMDb -> Yelp; confidence. '
         'Đọc prediction.md trước kết quả. Sinh viên tự phân tích/reflection theo W8 mục 24.')
    n.code(SETUP+'''environment=json.loads((ROOT/"environment.json").read_text(encoding="utf-8"))
print(json.dumps(environment,ensure_ascii=False,indent=2))
''')
    n.md('## Kế hoạch và controls\n\n'
         '11 run × 4 epoch, seed42, batch64, AdamW1e-4, weight_decay0.01. '
         'Pretrained/scratch dùng cùng tokenizer, BERT config và nested subsets '
         '10/25/50/100%; cố định validation/test. Primary là pretrained100%/T128, '
         'không chọn theo test. Giữ LR chung để kiểm soát, chưa tune từng mode.')
    n.code('''from pretrained_classification import BertSentimentClassifier
import inspect
print(inspect.getsource(BertSentimentClassifier))
''')
    n.md('## Training\n\nMặc định dùng kết quả thật đã lưu. Đặt RUN_TRAINING=True '
         'để train lại toàn bộ; sau đó gọi make_reports.py từ terminal để cập nhật báo cáo/hình.')
    n.code('''RUN_TRAINING=False
if RUN_TRAINING:
    import subprocess
    subprocess.run([sys.executable,"-u",str(ROOT/"run_experiments.py")],check=True)
else:
    print("Đọc 11 run đã lưu; không train lại trong phiên review.")
rows=list(csv.DictReader((ROOT/"results.csv").open(encoding="utf-8")))
assert len(rows)==11
def show(selected,cols):
    print(" | ".join(cols))
    for r in selected: print(" | ".join(str(r[c]) for c in cols))
''')
    n.md('## Kiểm tra split và truncation\n\n'
         'Cắt tensor cache phải phục hồi SEP tại boundary; đối chiếu với tokenizer '
         'trực tiếp. Training subsets lồng nhau, chỉ chứa training indices.')
    n.code('''from run_experiments import encode,loader
splits=json.loads((ROOT/"split_indices.json").read_text())
assert not set(splits["train"]).intersection(splits["validation"])
if (ROOT.parent/"lab7/split_indices.json").exists():
    previous=json.loads((ROOT.parent/"lab7/split_indices.json").read_text())
    assert splits["train"]==previous["train"] and splits["validation"]==previous["validation"]
subsets=splits["nested_subsets_local"]
for a,b in [("0.1","0.25"),("0.25","0.5"),("0.5","1.0")]:
    assert set(subsets[a]).issubset(subsets[b])
examples=["A good movie.","The actors played their parts well. "*100]
cached=encode(examples,tokenizer)
for length in [64,128,256]:
    ids,mask,types,y=next(iter(loader(cached,np.array([1,1]),length,2,tokenizer)))
    direct=tokenizer(examples,padding="max_length",truncation=True,max_length=length,return_tensors="pt")
    assert torch.equal(ids,direct["input_ids"])
    assert torch.equal(mask,direct["attention_mask"])
print("PASS: disjoint/nested splits, LAB7 split equality, SEP-safe truncation at 64/128/256")
''')
    n.md('## Experiment 1-2: initialization và frozen encoder')
    n.code('''selected=[r for r in rows if r["fraction"]=="1.0" and r["max_length"]=="128"]
show(selected,["initialization","mode","parameters","trainable_parameters","accuracy","precision","recall","f1","train_seconds"])
frozen=json.loads((ROOT/"frozen_verification.json").read_text())
assert frozen["max_encoder_parameter_change"]==0 and frozen["encoder_trainable_parameters"]==0
print("Frozen verification:",frozen)
''')
    n.md('## Experiment 3: data efficiency\n\n'
         'Xem cả accuracy và F1: một classifier thiên một lớp có thể F1 positive '
         'tương đối cao dù accuracy thấp. Các run không tiếp tục train từ subset nhỏ.')
    n.code('''data=[r for r in rows if r["mode"]=="finetune" and r["max_length"]=="128"]
show(data,["initialization","fraction","train_samples","accuracy","f1"])
''')
    n.image('data_efficiency.png','Accuracy và F1 theo phần trăm dữ liệu downstream')
    n.md('## Experiment 4: sequence length\n\n'
         'T gồm CLS/SEP. Memory là peak PyTorch allocated; không phải toàn GPU '
         'memory. SDPA backend có thể tối ưu lưu trữ tensor attention.')
    n.code('''lengths=[r for r in rows if r["initialization"]=="pretrained" and r["mode"]=="finetune" and r["fraction"]=="1.0"]
show(lengths,["max_length","accuracy","f1","train_seconds","peak_allocated_MiB"])
''')
    n.image('learning_curves.png','Train và validation loss của 11 run')
    n.md('## Experiment 5: domain shift\n\n'
         'Train IMDb; evaluate official IMDb test và 4000 mẫu Yelp official test '
         'cố định. Không fine-tune/tune trên Yelp. Chênh lệch còn gồm style, độ dài, '
         'labeling; không xem đó là đo catastrophic forgetting.')
    n.code('''domains=json.loads((ROOT/"domain_results.json").read_text())
show(domains,["domain","n","accuracy","precision","recall","f1"])
''')
    n.image('confusion_matrix.png','Confusion matrix primary model trên IMDb test')
    n.md('## Confidence\n\n'
         'Low [0.5,0.7), medium [0.7,0.9), high [0.9,1]; threshold định trước. '
         'Max softmax không phải bảo đảm prediction đúng hay probability đã calibration.')
    n.code('''buckets=json.loads((ROOT/"confidence_buckets.json").read_text())
show(buckets,["bucket","n","errors","mean_confidence","accuracy"])
errors=json.loads((ROOT/"error_examples.json").read_text())
assert sum(e["selection"]=="false_positive" for e in errors)==3
assert sum(e["selection"]=="false_negative" for e in errors)==3
assert sum(e["selection"]=="most_confident_wrong" for e in errors)==2
for e in errors: print("test",e["test_index"],e["selection"],"true",e["true"],"pred",e["pred"],"confidence",round(e["confidence"],6),"truncated",e["truncated"])
''')
    n.image('confidence_analysis.png','Confidence buckets và phân phối prediction đúng/sai')
    n.md('## Application\n\n'
         'Pipeline: text -> WordPiece IDs/mask -> contextual CLS -> Linear2 '
         '-> softmax. Các câu tự tạo không dùng làm test accuracy.')
    n.code('''model,config=load_classifier()
for text in ["I really enjoyed this movie.","This movie is not good.","Great, another two hours of my life wasted."]:
    encoded=tokenizer(text,truncation=True,max_length=config["max_length"],return_tensors="pt")
    with torch.inference_mode(): p=model(**encoded).softmax(-1)[0]
    print(text,"=>",["negative","positive"][p.argmax().item()],"confidence",round(p.max().item(),5))
''')
    n.md('## Phần cá nhân và giới hạn\n\n'
         'Hoàn thiện error_analysis.md, prediction_verification.md, Q1-Q6 và '
         'reflection300-500 từ theo W8. Báo cáo kỹ thuật và số liệu: experiment_report.md. '
         'Single seed; cùng LR chưa tối ưu riêng; khác data size đồng thời khác '
         'optimizer steps; chưa đo catastrophic forgetting; không suy từ metric '
         'thành hiểu ngôn ngữ toàn diện. AI assistance statement ở README.md.')
    return n


def build_all():
    counts={}
    for notebook in [tokenization(),fine_tuning()]:
        counts[notebook.path.name]=notebook.save_executed()
    (ROOT/'notebook_validation.json').write_text(json.dumps({'code_cells_executed':counts,'errors':0,
        'method':'Actual ordinary Python exec in a fresh shared namespace per notebook; training previously run by run_experiments.py; embedded PNG attachments'},indent=2),encoding='utf-8')


if __name__=='__main__':build_all()
