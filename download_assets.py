"""Download pinned public model/data into this workspace, never global HF cache."""
from pathlib import Path
import hashlib
import json
import requests

ROOT=Path(__file__).resolve().parent


def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(url,timeout=120,stream=True) as r:
        r.raise_for_status()
        with path.open('wb') as f:
            for chunk in r.iter_content(1024*1024):f.write(chunk)
    print(path.name,path.stat().st_size,flush=True)
    return {'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    model='google/bert_uncased_L-2_H-128_A-2'
    response=requests.get(f'https://huggingface.co/api/models/{model}',timeout=30)
    response.raise_for_status()
    meta=response.json(); revision=meta['sha']
    names={f['rfilename'] for f in meta['siblings']}
    weights='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
    files={}
    for name in ['config.json','vocab.txt',weights]:
        files['model/'+name]=fetch(f'https://huggingface.co/{model}/resolve/{revision}/{name}',ROOT/'model'/name)
    dataset='fancyzhx/yelp_polarity'
    response=requests.get(f'https://huggingface.co/api/datasets/{dataset}',timeout=30)
    response.raise_for_status(); meta=response.json(); data_revision=meta['sha']
    test_files=[f['rfilename'] for f in meta['siblings'] if 'test' in f['rfilename'] and f['rfilename'].endswith('.parquet')]
    if not test_files:raise RuntimeError('No Yelp test parquet found')
    for i,name in enumerate(sorted(test_files)):
        local=f'data/yelp-test-{i}.parquet'
        files[local]=fetch(f'https://huggingface.co/datasets/{dataset}/resolve/{data_revision}/{name}',ROOT/local)
    manifest={'model_id':model,'model_revision':revision,'domain_dataset':dataset,
              'domain_revision':data_revision,'files':files}
    (ROOT/'assets_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
