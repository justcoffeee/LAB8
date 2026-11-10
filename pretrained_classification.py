"""BERT encoder + task-specific linear head; CLI for saved-model inference."""
from pathlib import Path
import argparse
import json
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, BertTokenizerFast

ROOT=Path(__file__).resolve().parent


def get_tokenizer():
    return BertTokenizerFast.from_pretrained(str(ROOT/'model'),local_files_only=True)


class BertSentimentClassifier(nn.Module):
    def __init__(self,initialization='pretrained',frozen=False):
        super().__init__()
        config=AutoConfig.from_pretrained(str(ROOT/'model'),local_files_only=True)
        if initialization=='pretrained':
            self.encoder=AutoModel.from_pretrained(str(ROOT/'model'),local_files_only=True,
                                                   add_pooling_layer=False)
        elif initialization=='scratch':
            self.encoder=AutoModel.from_config(config,add_pooling_layer=False)
        else:
            raise ValueError('initialization must be pretrained or scratch')
        self.frozen=frozen
        for p in self.encoder.parameters():p.requires_grad_(not frozen)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,2)

    def train(self,mode=True):
        super().train(mode)
        # Frozen features should not be randomized by encoder dropout.
        if self.frozen:self.encoder.eval()
        return self

    def forward(self,input_ids,attention_mask,token_type_ids=None):
        args={'input_ids':input_ids,'attention_mask':attention_mask}
        if token_type_ids is not None:args['token_type_ids']=token_type_ids
        if self.frozen:
            with torch.no_grad():hidden=self.encoder(**args).last_hidden_state
        else:
            hidden=self.encoder(**args).last_hidden_state
        # Raw contextual CLS state; deliberately not the optional BERT pooler.
        return self.classifier(self.dropout(hidden[:,0]))


def load_classifier(checkpoint=None,device='cpu'):
    checkpoint=Path(checkpoint) if checkpoint else ROOT/'best_pretrained.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=BertSentimentClassifier('scratch',frozen=saved['config']['mode']=='frozen')
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(),saved['config']


def predict(texts,checkpoint=None,device='cpu'):
    model,config=load_classifier(checkpoint,device)
    tokens=get_tokenizer()(texts,padding=True,truncation=True,max_length=config['max_length'],return_tensors='pt')
    with torch.inference_mode():prob=model(**{k:v.to(device) for k,v in tokens.items()}).softmax(-1).cpu()
    return [{'text':t,'prediction':'positive' if p.argmax().item()==1 else 'negative',
             'positive_probability':p[1].item(),'confidence':p.max().item()} for t,p in zip(texts,prob)]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--text',default='I really enjoyed this movie.')
    parser.add_argument('--checkpoint',default=None)
    args=parser.parse_args()
    print(json.dumps(predict([args.text],args.checkpoint),ensure_ascii=False,indent=2))
