# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:40:30 2024

@author: jacki
"""

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle,json
from torch.utils.data import Dataset,DataLoader
import torch
import tqdm
from huggingface_hub import login
import argparse
import numpy as np
import nltk

argparser=argparse.ArgumentParser()
argparser.add_argument('--input_file',type=str)
argparser.add_argument('--seed',type=int,default=0)
argparser.add_argument('--input_model',type=str)
args = argparser.parse_args()
login(token='')
llm = LLM(model=args.input_model, 
          tensor_parallel_size=1,
          dtype=torch.float16,
          max_model_len=4096,
          gpu_memory_utilization = 0.8)
tokenizer = AutoTokenizer.from_pretrained(args.input_model, token='')
with open(args.input+'.pickle','rb') as file:
    media=pickle.load(file)
sent_list={}
for i in media:
    sent_list[i['entity_id']]=[]
    for j in i['reviews']:
        sent_list[i['entity_id']].append([k for k in j['sentences'] if len(k)>1])
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
def format_instruction(documents):
    if 'llama' in args.input_model.lower():
    #print(documents)
        message = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Always generate a required summary by the user."
            },
            {
                "role": "user", 
                "content": documents
            },
            {
                'role': 'assistant'
            }
        ]
        return tokenizer.apply_chat_template(message, tokenize=False)[:-10].strip()
    elif 'gemma' in args.input_model.lower():
        message = [
            {
                "role": "user", 
                "content": documents
            }
        ]
        return tokenizer.apply_chat_template(message, tokenize=False).strip()
    elif 'mistral' in args.input_model.lower():
        message = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Always generate a required summary by the user."
            },
            {
                "role": "user", 
                "content": documents
            }
        ]
        return tokenizer.apply_chat_template(message, tokenize=False).strip()
    else:
        raise ValueError('Model is not supported')
pairs={}
for i in sent_list.keys():
    #random.shuffle(sent_list[i])
    if 'llama' in args.input_model.lower():
        if 'tweetstance' in args.input_file:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words about "+i.split('_')[0].strip()+" for all the above tweets. Do not list sources of contents in the summary."
        elif 'tweetideology' in args.input_file:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above tweets. Do not list sources of contents in the summary."
        elif 'amazon' in args.input_file:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above reviews. Do not list sources of contents in the summary."
        else:
            raise ValueError('Dataset is not supported')
    elif 'mistral' in args.input_model.lower():
        if 'tweetstance' in args.input_file:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary less than 50 words about "+i.split('_')[0].strip()+" that summarizes the above tweets. Do not list sources of contents in the summary."
        elif 'tweetideology' in args.input_file:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary less than 40 words that summarizes the above tweets. Do not list sources of contents in the summary."
        elif 'amazon' in args.input_file:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary with 30 words that summarizes the above reviews."
        else:
            raise ValueError('Dataset is not supported')
    elif 'gemma' in args.input_model.lower():
        if 'tweetstance' in args.input_file:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary about "+i.split('_')[0].strip()+" with 65 words that summarizes the above tweets."
        elif 'tweetideology' in args.input_file:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary with 65 words that summarizes the above tweets."
        elif 'amazon' in args.input_file:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary with 65 words that summarizes the above reviews."
        else:
            raise ValueError('Dataset is not supported')
    pairs[i]=format_instruction(media_prompts)
pairs={i:pairs[i] for i in pairs.keys()}
pairs=[[i,pairs[i]] for i in pairs.keys()]
pairs=sorted(pairs, key=lambda x:tokenizer(x[1],return_tensors='pt').input_ids.size(1))
class MyDataset(Dataset):
    def __init__(self,dataset):
        self.data=dataset
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {'ids':self.data[i][0],'text_inputs':self.data[i][1]}
#assert 0==1
data=MyDataset(pairs)
dataloader=DataLoader(data,batch_size=200)
sampling_params = SamplingParams(temperature=0.6,top_p=0.9,max_tokens=512,seed=int(args.seed))
result={}
for i in dataloader:
    outputs = llm.generate(i['text_inputs'], sampling_params)
    outputs=[o.outputs[0].text for o in outputs]
    for j in range(len(i['ids'])):
        result[i['ids'][j]]=outputs[j]
result1={}
fail_words=['cannot fulfill','i cannot','document 1','documents 1']
not_in_key=[]
for i in result.keys():
    media=result[i].strip()
    fail_word=False
    for j in fail_words:
        if j in media.lower():
            print("**********************")
            #print(data[i])
            print(media)
            fail_word=True
            break
        #media=''
    if not fail_word:
        media=media.split('\n')
        media=[j for j in media if len(j)>1]
        if len(media)>2:
            media=' '.join(media[1:])
        elif len(media)==2:
            if len(media[0])<len(media[1]):
                media=media[-1]
            else:
                media=' '.join(media)
        else:
            media=media[0]
        if len(media.split())>=75 or len(media.split())<=35:
            not_in_key.append(i)
        result1[i]=media
    else:
        not_in_key.append(i)
        #data1[i]=media
    #print(media)
print(len(not_in_key))
not_in_pair=[i for i in pairs if i[0] in not_in_key]
sampling_params = SamplingParams(temperature=0.6,top_p=0.9,max_tokens=256)
if len(not_in_pair)>0:
   data=MyDataset(not_in_pair)
   dataloader=DataLoader(data,batch_size=200)
   result_test={}
   for i in dataloader:
       outputs = llm.generate(i['text_inputs'], sampling_params)
       outputs=[o.outputs[0].text for o in outputs]
       for j in range(len(i['ids'])):
           result_test[i['ids'][j]]=outputs[j]
   result1_test={}
   fail_words=['cannot fulfill','i cannot','document 1','documents 1']
   for i in result_test.keys():
       media=result_test[i].strip()
       fail_word=False
       for j in fail_words:
           if j in media.lower():
               print("**********************")
               print(i)
               print(result[i])
               fail_word=True
               break
           #media=''
       if not fail_word:
           media=media.split('\n')
           media=[j for j in media if len(j)>1]
           if len(media)>2:
               media=' '.join(media[1:])
           elif len(media)==2:
               if len(media[0])<len(media[1]):
                   media=media[-1]
               else:
                   media=' '.join(media)
           else:
               media=media[0]
           result1_test[i]=media
   #print(result1_test)
   for i in result1_test.keys():
       if abs(50-len(result1_test[i].split()))<abs(50-len(result1[i].split())):
           result1[i]=result1_test[i]
for i in result1.keys():
    if result1[i].startswith('"'):
        result1[i]=result1[i][1:]
    if result1[i].endswith('"'):
        result1[i]=result1[i][:-1]
count=0
for i in result1.keys():
    if len(result1[i])>=400:
        count+=1
        media=nltk.sent_tokenize(result1[i])
        counter=0
        for j in range(len(media)):
            if len(' '.join(media[:j+1]))>=400:
                counter=j
                break
        if len(' '.join(media[:counter]))>=200:
            result1[i]=' '.join(media[:counter])

if 'llama' in args.input_model.lower():
    with open(args.input+'_llama3proc'+str(args.seed)+'.json','w') as file:
        json.dump(result1,file)
elif 'mistral' in args.input_model.lower():
    with open(args.input+'_mistralproc'+str(args.seed)+'.json','w') as file:
        json.dump(result1,file)
elif 'gemma' in args.input_model.lower():
    with open(args.input+'_gemmaproc'+str(args.seed)+'.json','w') as file:
        json.dump(result1,file)
