# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:40:30 2024

"""

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle,json
from torch.utils.data import Dataset,DataLoader
import torch
import tqdm
from huggingface_hub import login
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('--model',type=str)
argparser.add_argument('--input_dataset',type=str)
argparser.add_argument('--num_gpu',type=int)
argparser.add_argument('--output_summary',type=str)
args = argparser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model, token='') #fill in huggingface token
with open(args.input_dataset+'.pickle','rb') as file:
    media=pickle.load(file)
sent_list={}
claims={}
for i in media:
    sent_list[i['entity_id']]=[]
    if 'news_stance' in args.input_dataset:
        claims[i['entity_id']]=i['claim']
    for j in i['reviews']:
        sent_list[i['entity_id']].append([k for k in j['sentences'] if len(k)>1])
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
if 'gemma' in args.model.lower():
    def format_instruction(documents):
        message = [
            {
                "role": "user", 
                "content": documents
            }
        ]
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
elif 'llama-2' in args.model.lower():
    def format_instruction(documents):
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
        return tokenizer.apply_chat_template(message, tokenize=False)
elif 'llama-3' in args.model.lower():
    def format_instruction(documents):
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
elif 'mistral' in args.model.lower():
    def format_instruction(documents):
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
        return tokenizer.apply_chat_template(message, tokenize=False)
else:
    print('not supported')
    assert 0==1
pairs={}
max_len=0
max_id=0
for i in sent_list.keys():
    #random.shuffle(sent_list[i])
    #print(i)
    if 'gemma' in args.model.lower():
        if 'article_bias' in args.input_dataset:
            media_prompts="Below is a list of news of a certain topic:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary with 110 words that summarizes the above news. Summary:\n"
        elif 'news_stance' in args.input_dataset:
            media_prompts='Below is a list of documents that support or against a claim, "'+claims[i]+'":\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary with 100 words for all the above documents in the form of consecutive texts. The summary should focus on information that supports or against the claim, "+'"'+claims[i]+'" . Do not specify the source of information in the summary. Do not write it as bullet points.\n Summary:'
        elif 'amazon' in args.input_dataset:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary with 65 words that summarizes the above reviews. Summary:\n"
        elif 'mitweet' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary with 65 words that summarizes the above tweets. Summary:\n"
        elif 'semeval' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary about "+i.split('_')[0].strip()+" with 65 words that summarizes the above tweets. Summary:"
    elif 'llama-3' in args.model.lower():
        if 'article_bias' in args.input_dataset:
            media_prompts="Below is a list of news of a certain topic:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary around 100 words for all the above news. Do not list sources of contents in the summary."
        elif 'news_stance' in args.input_dataset:
            media_prompts='Below is a list of documents that support or against a claim, "'+claims[i]+'":\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+='**********************\n'
            media_prompts+="Please write a single summary with 100 words for all the above documents in the form of consecutive texts. The summary should focus on information that supports or against the claim, "+'"'+claims[i]+'" . Do not specify the source of information in the summary. Do not write it as bullet points.\n Summary:'
        elif 'amazon' in args.input_dataset:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above reviews. Do not list sources of contents in the summary."
        elif 'mitweet' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above tweets. Do not list sources of contents in the summary."
        elif 'semeval' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words about "+i.split('_')[0].strip()+" for all the above tweets. Do not list sources of contents in the summary."
    elif 'llama-2' in args.model.lower():
        if 'article_bias' in args.input_dataset:
            media_prompts="Below is a list of news of a certain topic:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 100 words for all the above news."
        elif 'news_stance' in args.input_dataset:
            media_prompts='Below is a list of documents that support or against a claim, "'+claims[i]+'":\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 100 words for all the above documents in the form of consecutive texts. The summary should focus on information that supports or against the claim, "+'"'+claims[i]+'" . Do not specify the source of information in the summary. Do not write it as bullet points.'
        elif 'amazon' in args.input_dataset:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above reviews."
        elif 'mitweet' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above tweets."
        elif 'semeval' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words focusing on "+i.split('_')[0].strip()+" for all the above tweets.\n"
    elif 'mistral':
        if 'article_bias' in args.input_dataset:
            media_prompts="Below is a list of news of a certain topic:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 100 words for all the above news.\n Summary:"
        elif 'news_stance' in args.input_dataset:
            media_prompts='Below is a list of documents that support or against a claim, "'+claims[i]+'":\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 100 words for all the above documents in the form of consecutive texts. The summary should focus on information that supports or against the claim, "+'"'+claims[i]+'" . Do not specify the source of information in the summary. Do not write it as bullet points.\n Summary:'
        elif 'amazon' in args.input_dataset:
            media_prompts="Below is a list of product reviews:\n"
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 50 words for all the above reviews.\n Summary:"
        elif 'mitweet' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 40 words for all the above tweets.\n Summary:"
        elif 'semeval' in args.input_dataset:
            media_prompts="Below is a list of tweets about "+i.split('_')[0].strip()+':\n'
            for k in range(len(sent_list[i])):
                media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
            media_prompts+="Please write a single summary around 40 words focusing on "+i.split('_')[0].strip()+" for all the above tweets.\n Summary:"
    pairs[i]=format_instruction(media_prompts)
    media=tokenizer(pairs[i]).input_ids
    #max_len.append(len(media))
    if len(media)>max_len:
        max_id=i
        max_len=len(media)
    #print(len(media))
    #assert 0==1
pairs={i:pairs[i] for i in pairs.keys()}
pairs=[[i,pairs[i]] for i in pairs.keys()]
print(pairs[0][1])
pairs=sorted(pairs, key=lambda x:tokenizer(x[1],return_tensors='pt').input_ids.size(1))
class MyDataset(Dataset):
    def __init__(self,dataset):
        self.data=dataset
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {'ids':self.data[i][0],'text_inputs':self.data[i][1]}
data=MyDataset(pairs)
dataloader=DataLoader(data,batch_size=300)
llm = LLM(model=args.model, 
          tensor_parallel_size=args.num_gpu,
          gpu_memory_utilization = 0.8)
sampling_params = SamplingParams(temperature=0.6,top_p=0.9,max_tokens=512,seed=0)
result={}
for i in dataloader:
    outputs = llm.generate(i['text_inputs'], sampling_params)
    outputs=[o.outputs[0].text for o in outputs]
    for j in range(len(i['ids'])):
        result[i['ids'][j]]=outputs[j]
result1={}
fail_words=['cannot fulfill','i cannot','document 1','documents 1']
for i in result.keys():
    media=result[i].strip()
    fail_word=False
    for j in fail_words:
        if j in media.lower():
            print("**********************")
            print(data[i])
            fail_word=True
            break
        #media=''
    if not fail_word:
        if 'article_bias' in args.input_dataset or 'news_stance' in args.input_dataset:
            if len(media.split())<=200 and len(media.split())>=40:
                result1[i]=media
        else:
            if len(media.split())<=100 and len(media.split())>=20:
                result1[i]=media
        #data1[i]=media
    #print(media)
not_in_key=[]
for i in result.keys():
    if i not in result1.keys():
        not_in_key.append(i)
not_in_pair=[i for i in pairs if i[0] in not_in_key]
if len(not_in_pair)>0:
    data=MyDataset(not_in_pair)
    dataloader=DataLoader(data,batch_size=300)
    for i in dataloader:
        outputs = llm.generate(i['text_inputs'], sampling_params)
        outputs=[o.outputs[0].text for o in outputs]
        for j in range(len(i['ids'])):
            result[i['ids'][j]]=outputs[j]
    result1={}
    fail_words=['cannot fulfill','i cannot','document 1','documents 1']
    for i in result.keys():
        media=result[i].strip()
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
            if 'article_bias' in args.input_dataset or 'news_stance' in args.input_dataset:
                if len(media.split())<=200 and len(media.split())>=40:
                    result1[i]=media
                else:
                    print(i)
                    print(media)
            else:
                if len(media.split())<=100 and len(media.split())>=20:
                    result1[i]=media
                else:
                    print(i)
                    print(media)
for i in result1.keys():
    if result1[i].startswith('"'):
        result1[i]=result1[i][1:]
    if result1[i].endswith('"'):
        result1[i]=result1[i][:-1]
with open(args.output_summary,'w') as file:
    json.dump(result1,file)