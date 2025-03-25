# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 00:07:13 2023
"""
import anthropic
import tiktoken
import json
import openai
import pickle
import random
import time
import argparse

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)
argparser=argparse.ArgumentParser()
argparser.add_argument('--model',type=str)
argparser.add_argument('--input_dataset',type=str)
argparser.add_argument('--output_summary',type=str)
args = argparser.parse_args()
with open(args.input_dataset+'.pickle','rb') as file:
    media=pickle.load(file)
mapping={}
sent_list={}
claims={}
for i in media:
    if 'news_stance' in args.input_dataset:
        claims[i['entity_id']]=i['claim']
    sent_list[i['entity_id']]=[]
    for j in i['reviews']:
        sent_list[i['entity_id']].append([k for k in j['sentences'] if len(k)>1])
#prompt="Below is a list of tweets of"
pairs={}
for i in sent_list.keys():
    if 'article_bias' in args.input_dataset:
        media_prompts="Below is a list of news of a certain topic:\n"
        for k in range(len(sent_list[i])):
            media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
        media_prompts+="Please write a single summary around 100 words for all the above news."
    elif 'news_stance' in args.input_dataset:
        media_prompts='Below is a list of documents that support or against a claim, "'+claims[i]+'":\n'
        for k in range(len(sent_list[i])):
            media_prompts+=str(k+1)+'.'+' '.join(sent_list[i][k])+'\n'
        media_prompts+="\nPlease write a single summary around 100 words for all the above documents in the form of consecutive texts. The summary should focus on information that supports or against the claim, "+'"'+claims[i]+'" . Do not specify the source of information in the summary. Do not write it as bullet points.'
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
        media_prompts+="Please write a single summary around 50 words focusing on "+i.split('_')[0].strip()+" for all the above tweets."
    #random.shuffle(sent_list[i])
    pairs[i]=media_prompts
pairs={i:pairs[i] for i in pairs.keys()}
result={}
#assert 0==1
for i in pairs.keys():
    success=False
    while not success:
        try:
           messages=[]
           messages.append({'role':'user','content':pairs[i]})
           message = client.messages.create(
               model=args.model,
               max_tokens=1000,
               temperature=1.0,
               system="You are a helpful assistant.",
               messages=messages
           )
           result[i]=message.content[0].text
           success=True
           time.sleep(0.5)
        except:
           time.sleep(1)
           print(i)
with open(args.output,'w') as file:
   json.dump(result,file)
