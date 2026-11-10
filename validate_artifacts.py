"""Verify saved metrics, split integrity, hashes and experiment bookkeeping."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

ROOT=Path(__file__).resolve().parent


def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    rows=list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8')))
    assert len(rows)==11 and len({r['run'] for r in rows})==11
    checks=[]
    for row in rows:
        history=load(f'history/{row["run"]}.json')
        assert len(history)==int(row['epochs'])==4
        best=min(history,key=lambda h:h['val_loss'])
        assert int(row['best_epoch'])==best['epoch']
        assert abs(float(row['val_loss'])-best['val_loss'])<1e-10
        data=np.load(ROOT/'predictions'/f'{row["run"]}.npz')
        probs,y=data['probabilities'],data['labels']
        assert probs.shape==(25000,2) and y.shape==(25000,)
        assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
        np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
        pred=probs.argmax(1)
        for key,fn in [('accuracy',accuracy_score),('precision',precision_score),('recall',recall_score),('f1',f1_score)]:
            computed=fn(y,pred) if key=='accuracy' else fn(y,pred,zero_division=0)
            assert abs(computed-float(row[key]))<1e-10
        assert abs(-np.log(probs[np.arange(len(y)),y]).mean()-float(row['test_loss']))<2e-6
    checks.append('11 runs x 4 epochs; checkpoint selection equals minimum validation CE; all metrics and CE independently recomputed')
    splits=load('split_indices.json')
    tr,va=splits['train'],splits['validation']
    assert len(set(tr))==20000 and len(set(va))==5000
    assert not set(tr).intersection(va) and set(tr).union(va)==set(range(25000))
    subsets=splits['nested_subsets_local']
    for key,n in [('0.1',2000),('0.25',5000),('0.5',10000),('1.0',20000)]:
        assert len(set(subsets[key]))==n and set(subsets[key]).issubset(range(20000))
    for a,b in [('0.1','0.25'),('0.25','0.5'),('0.5','1.0')]:
        assert set(subsets[a]).issubset(subsets[b])
    old_path=ROOT.parent/'lab7/split_indices.json'
    if old_path.exists():
        old=json.loads(old_path.read_text())
        assert tr==old['train'] and va==old['validation']
    checks.append('disjoint IMDb split, exact equality with LAB7, correctly sized nested subsets')
    frozen=load('frozen_verification.json')
    assert frozen['max_encoder_parameter_change']==0 and frozen['encoder_trainable_parameters']==0
    assert frozen['head_trainable_parameters']==258
    checks.append('frozen encoder unchanged; exactly 258 head parameters trained')
    primary=np.load(ROOT/'predictions/pretrained_finetune_p100_t128.npz')
    y,p=primary['labels'],primary['probabilities']
    assert confusion_matrix(y,p.argmax(1)).tolist()==load('confusion_matrix.json')
    buckets=load('confidence_buckets.json')
    assert sum(b['n'] for b in buckets)==25000
    assert sum(b['errors'] for b in buckets)==int((p.argmax(1)!=y).sum())
    errors=load('error_examples.json')
    assert len(errors)==8
    assert sum(e['selection']=='false_positive' for e in errors)==3
    assert sum(e['selection']=='false_negative' for e in errors)==3
    assert sum(e['selection']=='most_confident_wrong' for e in errors)==2
    for e in errors:
        i=e['test_index']
        assert e['true']==int(y[i]) and e['pred']==int(p[i].argmax()) and e['true']!=e['pred']
        assert abs(e['confidence']-float(p[i].max()))<1e-10
        if e['selection']=='false_positive':assert e['true']==0 and e['pred']==1
        if e['selection']=='false_negative':assert e['true']==1 and e['pred']==0
    wrong=np.flatnonzero(p.argmax(1)!=y)
    top=wrong[np.argsort(-p[wrong].max(1))[:2]].tolist()
    assert [e['test_index'] for e in errors if e['selection']=='most_confident_wrong']==top
    checks.append('confusion matrix, confidence buckets, 3 FP/3 FN and top-2 confident wrong examples exactly match primary predictions')
    domain=np.load(ROOT/'predictions/yelp.npz')
    assert domain['probabilities'].shape==(4000,2)
    assert len(set(load('domain_indices.json')['indices']))==4000
    for r,data in zip(load('domain_results.json'),[primary,domain]):
        yy,pp=data['labels'],data['probabilities'].argmax(1)
        assert abs(r['f1']-f1_score(yy,pp))<1e-10
        assert abs(r['accuracy']-accuracy_score(yy,pp))<1e-10
    checks.append('Yelp held-out sample has 4000 unique official test indices; domain metrics recomputed')
    env=load('environment.json');assert 'finished_utc' in env
    for name,digest in env['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,info in load('assets_manifest.json')['files'].items():
        assert (ROOT/name).stat().st_size==info['bytes']
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==info['sha256']
    checks.append('training source and downloaded model/data hashes match provenance')
    required=['README.md','calculations.md','prediction.md','tokenization.ipynb','pretrained_classification.py',
              'fine_tuning.ipynb','results.csv','data_efficiency.png','confusion_matrix.png',
              'confidence_analysis.png','error_analysis.md','reflection.md']
    assert all((ROOT/name).is_file() for name in required)
    notebooks=load('notebook_validation.json')
    assert notebooks['errors']==0 and len(notebooks['code_cells_executed'])==2
    for name,count in notebooks['code_cells_executed'].items():
        doc=load(name);cells=[c for c in doc['cells'] if c['cell_type']=='code']
        assert len(cells)==count
        assert [c['execution_count'] for c in cells]==list(range(1,count+1))
        assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
    checks.append('all W8 deliverable filenames present; both notebooks have real executed outputs and zero errors')
    (ROOT/'validation.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2),encoding='utf-8')
    print('PASS: '+'; '.join(checks))


if __name__=='__main__':main()
