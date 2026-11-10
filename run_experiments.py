"""W8: 11 controlled BERT experiments, IMDb -> Yelp domain shift, real outputs."""
from pathlib import Path
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import random
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import transformers
from datasets import Dataset
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from pretrained_classification import BertSentimentClassifier,get_tokenizer

ROOT=Path(__file__).resolve().parent
SEED=42


def dump(name,data):
    path=ROOT/name
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def metrics(y,probs):
    pred=probs.argmax(1)
    return {'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
            'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0)}


def encode(texts,tokenizer):
    batches={}
    for start in range(0,len(texts),512):
        encoded=tokenizer(texts[start:start+512],padding='max_length',truncation=True,
                          max_length=256,return_tensors='pt')
        for key,value in encoded.items():batches.setdefault(key,[]).append(value)
    return {key:torch.cat(values) for key,values in batches.items()}


def load_data(tokenizer):
    cache=Path.home()/'.cache/huggingface/datasets/stanfordnlp___imdb/plain_text/0.0.0'
    paths=sorted(cache.glob('*/imdb-train.arrow'))
    if not paths:raise FileNotFoundError('IMDb Arrow cache missing; see README for download instructions')
    folder=paths[-1].parent
    train=Dataset.from_file(str(folder/'imdb-train.arrow'))
    test=Dataset.from_file(str(folder/'imdb-test.arrow'))
    all_text=list(train['text']);all_y=np.asarray(train['label'])
    train_idx,val_idx=train_test_split(np.arange(len(all_y)),test_size=.2,random_state=SEED,stratify=all_y)
    texts=[[all_text[int(i)] for i in train_idx],[all_text[int(i)] for i in val_idx],list(test['text'])]
    ys=[all_y[train_idx],all_y[val_idx],np.asarray(test['label'])]
    # Nested class-balanced subsets: same indices for scratch and pretrained.
    rng=np.random.default_rng(SEED)
    per_class=[rng.permutation(np.flatnonzero(ys[0]==label)) for label in [0,1]]
    subsets={}
    for fraction in [.1,.25,.5,1.]:
        selected=np.concatenate([indices[:round(len(indices)*fraction)] for indices in per_class])
        subsets[fraction]=np.sort(selected)
    dump('split_indices.json',{'train':train_idx.tolist(),'validation':val_idx.tolist(),
         'nested_subsets_local':{str(f):idx.tolist() for f,idx in subsets.items()}})
    arrays=[encode(texts_split,tokenizer) for texts_split in texts]
    yelp=pq.read_table(ROOT/'data/yelp-test-0.parquet').to_pydict()
    yelp_y=np.asarray(yelp['label'])
    domain_idx=train_test_split(np.arange(len(yelp_y)),train_size=4000,random_state=SEED,stratify=yelp_y)[0]
    domain_text=[yelp['text'][int(i)] for i in domain_idx]
    dump('domain_indices.json',{'split':'Yelp official test','sample_size':len(domain_idx),
                               'indices':domain_idx.tolist(),'seed':SEED})
    domain=(encode(domain_text,tokenizer),yelp_y[domain_idx],domain_text)
    dump('dataset_provenance.json',{'imdb_arrow_folder':str(folder),
          'imdb_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [folder/'imdb-train.arrow',folder/'imdb-test.arrow']},
          'imdb_sizes':[len(x) for x in ys],'yelp_original_test_size':len(yelp_y),
          'yelp_evaluated_size':len(domain_idx),'labels':'0=negative, 1=positive for both datasets'})
    return texts,ys,arrays,subsets,domain


def loader(encoded,ys,length,batch,tokenizer,indices=None,shuffle=False):
    if indices is None:indices=np.arange(len(ys))
    ids=encoded['input_ids'][indices,:length].clone()
    mask=encoded['attention_mask'][indices,:length].clone()
    # Slicing a 256-token cache must restore SEP at the truncation boundary.
    last=mask.sum(1)-1
    ids[torch.arange(len(ids)),last]=tokenizer.sep_token_id
    type_ids=encoded['token_type_ids'][indices,:length].clone()
    data=TensorDataset(ids,mask,type_ids,torch.tensor(ys[indices],dtype=torch.long))
    gen=torch.Generator().manual_seed(SEED)
    return DataLoader(data,batch_size=batch,shuffle=shuffle,generator=gen,num_workers=0)


def evaluate(model,batches,device):
    model.eval();total,n=0.,0;probs=[]
    with torch.inference_mode():
        for ids,mask,types,y in batches:
            ids,mask,types,y=[t.to(device) for t in [ids,mask,types,y]]
            logits=model(ids,mask,types)
            total+=nn.functional.cross_entropy(logits,y).item()*len(y);n+=len(y)
            probs.append(logits.softmax(-1).cpu().numpy())
    return total/n,np.concatenate(probs)


def train_one(initialization,mode,fraction,length,args,arrays,ys,subsets,tokenizer,device):
    seed_all()
    name=f'{initialization}_{mode}_p{int(fraction*100)}_t{length}'
    model=BertSentimentClassifier(initialization,frozen=mode=='frozen').to(device)
    if model.frozen:
        frozen_before={k:v.cpu().clone() for k,v in model.encoder.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    tr=loader(arrays[0],ys[0],length,args.batch_size,tokenizer,subsets[fraction],True)
    va=loader(arrays[1],ys[1],length,args.batch_size,tokenizer)
    te=loader(arrays[2],ys[2],length,args.batch_size,tokenizer)
    best_loss,best_state,best_epoch=float('inf'),None,0
    history=[];seconds=0.
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for ep in range(1,args.epochs+1):
        model.train();sync(device);start=time.perf_counter();total,n=0.,0
        for ids,mask,types,y in tr:
            ids,mask,types,y=[t.to(device) for t in [ids,mask,types,y]]
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(ids,mask,types),y)
            loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step();total+=loss.item()*len(y);n+=len(y)
        sync(device);elapsed=time.perf_counter()-start;seconds+=elapsed
        vl,vp=evaluate(model,va,device)
        history.append({'epoch':ep,'train_loss':total/n,'val_loss':vl,
                        'val_f1':metrics(ys[1],vp)['f1'],'train_seconds':elapsed})
        print(f'{name} ep {ep}/{args.epochs}: train={total/n:.4f} val={vl:.4f} '
              f'valF1={history[-1]["val_f1"]:.4f} time={elapsed:.1f}s',flush=True)
        if vl<best_loss:
            best_loss,best_epoch=vl,ep
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
    if model.frozen:
        delta=max((v.cpu()-frozen_before[k]).abs().max().item() for k,v in model.encoder.state_dict().items())
        assert delta==0,'Frozen encoder weights changed'
        dump('frozen_verification.json',{'max_encoder_parameter_change':delta,
             'encoder_trainable_parameters':sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
             'head_trainable_parameters':sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)})
        del frozen_before
    model.load_state_dict(best_state)
    tl,prob=evaluate(model,te,device)
    row={'run':name,'initialization':initialization,'mode':mode,'fraction':fraction,
         'train_samples':len(subsets[fraction]),'max_length':length,'epochs':args.epochs,
         'best_epoch':best_epoch,'learning_rate':args.lr,'batch_size':args.batch_size,
         'parameters':sum(p.numel() for p in model.parameters()),
         'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'train_seconds':seconds,'peak_allocated_MiB':peak,'val_loss':best_loss,'test_loss':tl,
         **metrics(ys[2],prob)}
    dump(f'history/{name}.json',history)
    np.savez_compressed(ROOT/'predictions'/f'{name}.npz',probabilities=prob,labels=ys[2])
    if initialization=='pretrained' and fraction==1 and length==128:
        checkpoint='best_frozen.pt' if mode=='frozen' else 'best_pretrained.pt'
        torch.save({'state_dict':best_state,'config':row},ROOT/checkpoint)
    print('RESULT '+json.dumps(row),flush=True)
    return row,model,prob


def savefig(name):
    plt.tight_layout();plt.savefig(ROOT/name,dpi=160,bbox_inches='tight');plt.close()


def primary_artifacts(model,prob,texts,ys,tokenizer,domain,args,device):
    domain_encoded,domain_y,domain_text=domain
    dl=loader(domain_encoded,domain_y,128,args.batch_size,tokenizer)
    _,dp=evaluate(model,dl,device)
    np.savez_compressed(ROOT/'predictions/yelp.npz',probabilities=dp,labels=domain_y)
    dump('domain_results.json',[{'domain':'IMDb movie reviews','n':len(ys[2]),**metrics(ys[2],prob)},
                                {'domain':'Yelp business reviews','n':len(domain_y),**metrics(domain_y,dp)}])
    confidence=prob.max(1);pred=prob.argmax(1);correct=pred==ys[2]
    buckets=[]
    # Explicit, fixed thresholds; no bin thresholds chosen from accuracy.
    for name,mask in [('low_0.50_0.70',confidence<.7),
                       ('medium_0.70_0.90',(confidence>=.7)&(confidence<.9)),
                       ('high_0.90_1.00',confidence>=.9)]:
        buckets.append({'bucket':name,'n':int(mask.sum()),'errors':int((~correct&mask).sum()),
                        'accuracy':float(correct[mask].mean()) if mask.any() else None,
                        'mean_confidence':float(confidence[mask].mean()) if mask.any() else None})
    dump('confidence_buckets.json',buckets)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3)
    axes[0].bar(x-.18,[b['mean_confidence'] or 0 for b in buckets],width=.36,label='Mean confidence')
    axes[0].bar(x+.18,[b['accuracy'] or 0 for b in buckets],width=.36,label='Accuracy')
    axes[0].set_xticks(x,['Low','Medium','High']);axes[0].set_ylim(0,1);axes[0].legend()
    axes[0].set_title('Confidence is not correctness')
    axes[1].hist([confidence[correct],confidence[~correct]],bins=np.linspace(.5,1,21),label=['Correct','Incorrect'])
    axes[1].set_xlabel('Max softmax probability');axes[1].set_ylabel('Count');axes[1].legend()
    savefig('confidence_analysis.png')
    cm=confusion_matrix(ys[2],pred)
    dump('confusion_matrix.json',cm.tolist())
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set_xticks([0,1],['Negative','Positive']);ax.set_yticks([0,1],['Negative','Positive'])
    ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title('Fine-tuned BERT / IMDb test')
    savefig('confusion_matrix.png')
    selections=[]
    for label,mask in [('false_positive',(pred==1)&(ys[2]==0)),('false_negative',(pred==0)&(ys[2]==1))]:
        # First three test indices, independent of confidence; top two errors additionally.
        selections += [(int(i),label) for i in np.flatnonzero(mask)[:3]]
    wrong=np.flatnonzero(~correct)
    selections += [(int(i),'most_confident_wrong') for i in wrong[np.argsort(-confidence[wrong])[:2]]]
    errors=[]
    for i,kind in selections:
        raw=texts[2][i]
        token_ids=tokenizer(raw,add_special_tokens=True,truncation=False)['input_ids']
        enc=tokenizer(raw,truncation=True,max_length=128)
        errors.append({'test_index':i,'selection':kind,'text':raw,'true':int(ys[2][i]),
                       'pred':int(pred[i]),'confidence':float(confidence[i]),'positive_probability':float(prob[i,1]),
                       'full_token_count':len(token_ids),'retained_token_count':len(enc['input_ids']),
                       'truncated':len(token_ids)>128,'unknown_tokens':enc['input_ids'].count(tokenizer.unk_token_id),
                       'retained_tokens':tokenizer.convert_ids_to_tokens(enc['input_ids'])})
    dump('error_examples.json',errors)
    cases=[('short','I really enjoyed this movie.'),('negation','This movie is not good.'),
           ('contrast','The acting is poor but the story is wonderful.'),
           ('ambiguous','A strange film with memorable moments and dull stretches.'),
           ('sarcasm','Great, another two hours of my life wasted.')]
    outputs=[]
    for kind,text in cases:
        enc=tokenizer(text,return_tensors='pt')
        with torch.inference_mode():p=model(**{k:v.to(device) for k,v in enc.items()}).softmax(-1)[0].cpu()
        outputs.append({'kind':kind,'text':text,'prediction':int(p.argmax()),'confidence':p.max().item(),
                        'positive_probability':p[1].item()})
    dump('application_predictions.json',outputs)


def tokenizer_artifact(tokenizer):
    text='The cats are playing.'
    batch=tokenizer([text,'A good movie.'],padding='max_length',max_length=12,truncation=True)
    demo={'text':text,'words_whitespace':len(text.split()),'word_definition':'whitespace; punctuation remains attached',
          'tokens':tokenizer.convert_ids_to_tokens(batch['input_ids'][0]),
          'input_ids':batch['input_ids'][0],'attention_mask':batch['attention_mask'][0],
          'token_type_ids':batch['token_type_ids'][0],
          'special_tokens':tokenizer.special_tokens_map,'keys':list(batch.keys())}
    active=[i for i,m in zip(demo['input_ids'],demo['attention_mask']) if m]
    demo['content_tokens']=len([i for i in active if i not in tokenizer.all_special_ids])
    demo['subword_continuations']=sum(t.startswith('##') for t in demo['tokens'])
    demo['subword_definition']='All WordPiece content tokens are subword-vocabulary tokens; ## counts continuation pieces only'
    dump('tokenization_results.json',demo)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--lr',type=float,default=1e-4)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0:parser.error('positive epochs, batch size and LR required')
    torch.set_num_threads(4);seed_all()
    for folder in ['history','predictions']:(ROOT/folder).mkdir(exist_ok=True)
    tokenizer=get_tokenizer();tokenizer_artifact(tokenizer)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE',device,flush=True)
    texts,ys,arrays,subsets,domain=load_data(tokenizer)
    env={'started_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'model_id':'google/bert_uncased_L-2_H-128_A-2',
         'tokenizer':'BertTokenizerFast / uncased WordPiece','torch':torch.__version__,'transformers':transformers.__version__,
         'python':platform.python_version(),'numpy':np.__version__,'device':str(device),
         'hardware':torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor(),
         'train_validation_test':[len(y) for y in ys],'epochs':args.epochs,'batch_size':args.batch_size,
         'optimizer':'AdamW','lr':args.lr,'weight_decay':.01,'gradient_clip':1.,'scheduler':None,
         'primary':'pretrained_finetune_p100_t128; fixed before experiments',
         'checkpoint_selection':'minimum validation cross entropy per run',
         'timing':'summed train epochs only; CUDA synchronized; validation/test excluded; frozen includes encoder forward each batch, no feature cache',
         'max_length_definition':'total WordPiece tokens including CLS and SEP',
         'determinism':'fixed seeds and deterministic cuDNN; no bitwise guarantee across machines/versions',
         'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ['run_experiments.py','pretrained_classification.py']}}
    dump('environment.json',env)
    # Same architecture, tokenizer, LR, epochs and nested subset in each pair.
    configs=[(initialization,'finetune',fraction,128)
             for initialization in ['pretrained','scratch'] for fraction in [.1,.25,.5,1.]]
    configs += [('pretrained','frozen',1.,128),('pretrained','finetune',1.,64),('pretrained','finetune',1.,256)]
    rows=[]
    for initialization,mode,fraction,length in configs:
        row,model,prob=train_one(initialization,mode,fraction,length,args,arrays,ys,subsets,tokenizer,device)
        rows.append(row)
        with (ROOT/'results.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        if initialization=='pretrained' and mode=='finetune' and fraction==1 and length==128:
            primary_artifacts(model,prob,texts,ys,tokenizer,domain,args,device)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for initialization in ['pretrained','scratch']:
        selected=[r for r in rows if r['initialization']==initialization and r['mode']=='finetune' and r['max_length']==128]
        for ax,metric in zip(axes,['accuracy','f1']):
            ax.plot([r['fraction']*100 for r in selected],[r[metric] for r in selected],marker='o',label=initialization)
            ax.set_xticks([10,25,50,100]);ax.set_xlabel('% of training split');ax.set_ylabel(metric);ax.legend()
    savefig('data_efficiency.png')
    fig,axes=plt.subplots(4,3,figsize=(14,13))
    for ax,r in zip(axes.flat,rows):
        h=json.loads((ROOT/'history'/f'{r["run"]}.json').read_text())
        ax.plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label='train')
        ax.plot([x['epoch'] for x in h],[x['val_loss'] for x in h],label='validation')
        ax.set_title(r['run'],fontsize=9);ax.set_xlabel('Epoch');ax.set_ylabel('Cross entropy');ax.legend()
    axes.flat[-1].axis('off');savefig('learning_curves.png')
    env['finished_utc']=datetime.now(timezone.utc).isoformat();dump('environment.json',env)
    print('COMPLETE: 11 configurations, real IMDb and Yelp evaluation.',flush=True)


if __name__=='__main__':main()
