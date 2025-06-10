# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:58:01 2024

"""

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import torch
import json
import pickle
import tqdm
import argparse
import nltk
argparser=argparse.ArgumentParser()
argparser.add_argument('--input_dataset',type=str)
argparser.add_argument('--output_summary',type=str)
argparser.add_argument('--model_path',type=str)
args = argparser.parse_args()
PRIMER_path=args.model_path
TOKENIZER = PegasusTokenizer.from_pretrained(PRIMER_path)
MODEL = PegasusForConditionalGeneration.from_pretrained(PRIMER_path).to(device='cuda')
MODEL.gradient_checkpointing_enable()
#PAD_TOKEN_ID = TOKENIZER.pad_token_id
#DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")
with open(args.input_dataset+'.pickle','rb') as file:
    data=pickle.load(file)
def process_document(documents):
    input_ids_all=[]
    for data in documents:
        all_docs =[' '.join(j['sentences']) for j in data['reviews']]
        for i, doc in enumerate(all_docs):
            doc = doc.replace("\n", " ")
            doc = " ".join(doc.split())
            all_docs[i] = doc
        #print(all_docs[0][:500])
        #### concat with global attention on doc-sep
        input_ids = []
        for doc in all_docs:
            input_ids.extend(
                TOKENIZER.encode(
                    doc,
                    truncation=True,
                    max_length=1024 // len(all_docs),
                )[1:-1]
            )
            #input_ids.append(DOCSEP_TOKEN_ID)
        input_ids=TOKENIZER.decode(input_ids)
        input_ids_all.append(input_ids)
    
    return TOKENIZER(input_ids_all, truncation=True, padding=True, return_tensors="pt").to('cuda')


def batch_process(batch):
    inputs=process_document(batch)
    # get the input ids and attention masks together
    #global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
    # put global attention on <s> token

    #global_attention_mask[:, 0] = 1
    #global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
    generated_ids = MODEL.generate(**inputs,max_length=256)
    generated_str = TOKENIZER.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
    #result={}
    #result['generated_summaries'] = generated_str
    #result['ids']=[i['entity_id'] for i in batch]
    return generated_str
batch_size=8
result={}
with torch.no_grad():
    for i in tqdm.tqdm(range(0,len(data),batch_size)):
        media_batch=data[i:(i+batch_size)]
        media_str=batch_process(media_batch)
        #print(media_str[0])
        for j in range(len(media_batch)):
            result[media_batch[j]['entity_id']]=media_str[j]
            #print(media_str[j])
            #print(len(media_str[j].split()))
data1={}
if 'article_bias' in args.input_dataset or 'news_stance' in args.input_dataset:
    max_len=200
    media_len=150
    min_len=100
else:
    max_len=100
    media_len=75
    min_len=50
data=result
for i in data.keys():
    data[i]=data[i].strip()
    if data[i].startswith('–'):
        data[i]=data[i][1:].strip()
    if len(data[i].split())>=max_len:
        media_sents=nltk.sent_tokenize(data[i])
        media_seq=''
        for j in media_sents:
            media_media_seq=media_seq+j
            if len(media_media_seq.split())>=media_len:
                break
            else:
                media_seq=media_media_seq
        data[i]=media_seq
    if len(data[i])>min_len:
        data1[i]=data[i]
with open(args.output_summary,'w') as file:
    json.dump(data1,file)