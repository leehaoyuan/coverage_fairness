# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:50:10 2024

@author: jacki
"""

import pickle
import json
import random
import numpy as np
from huggingface_hub import login
import math
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
argparser=argparse.ArgumentParser()
argparser.add_argument('--model',type=str)
argparser.add_argument('--summary1',type=str)
argparser.add_argument('--summary2',type=str)
argparser.add_argument('--summary3',type=str)
argparser.add_argument('--data',type=str)
argparser.add_argument('--output_path',type=str)
args = argparser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model, token='')
np.random.seed(1)
random.seed(0)
with open(args.data+'.pickle','rb') as file:
    dataset=pickle.load(file)
batch1_key=[i['entity_id'] for i in dataset]
with open(args.summary1+'_ecv2.pickle','rb') as file:
    middle=pickle.load(file)
with open(args.summary2+'_ecv2.pickle','rb') as file:
    shuffle=pickle.load(file)
with open(args.summary3+'_ecv2.pickle','rb') as file:
    order=pickle.load(file)
with open(args.summary1+'.json','r') as file:
    middle_sum=json.load(file)
with open(args.summary2+'.json','r') as file:
    shuffle_sum=json.load(file)
with open(args.summary3+'.json','r') as file:
    order_sum=json.load(file)

fair={}
unfair={}
diff={}
sample=[order_sum,middle_sum,shuffle_sum]
selected_cp={}
rejected_cp={}
same_max=0
drag={}
#assert 0==1
if 'tweetstance' in args.data:
    num_label=2
else:
    num_label=3
for i in middle.keys():
    if i in order.keys() and i in shuffle.keys() and i in batch1_key:
        media=[order[i][0],middle[i][0],shuffle[i][0]]
        min_idx=np.argmin(media)
        max_idx=np.argmax(media)
        media1=[order[i][1],middle[i][1],shuffle[i][1]]
        if np.argmax(media1[max_idx][0])==np.argmax(media1[min_idx][0]):
            same_max+=1
        diff[i]=(np.max(media)-np.min(media))
        fair[i]=sample[min_idx][i]
        unfair[i]=sample[max_idx][i]
        media_cp=[order[i][1],middle[i][1],shuffle[i][1]]
        selected_cp[i]=media_cp[min_idx]
        rejected_cp[i]=media_cp[max_idx]
        media_drag=[0,0]
        for j in range(num_label):
            if order[i][1][1][j]>0:
                media_drag[j]=media_cp[min_idx][0][j]-media_cp[max_idx][0][j]
        drag[i]=media_drag

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
sent_list={}
for i in dataset:
    sent_list[i['entity_id']]=[]
    for j in i['reviews']:
        sent_list[i['entity_id']].append([k for k in j['sentences'] if len(k)>1])
pairs={}

#random.shuffle(sent_list[i])
for i in sent_list.keys():
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
result=[]
for i in fair.keys():
    media={'prompt':pairs[i],'chosen':fair[i],'rejected':unfair[i],'diff':diff[i],'id':i}
    result.append(media)
random.shuffle(result)
train_group={}
for i in dataset:
    count=np.zeros(num_label)
    for j in i['reviews']:
        count[j['label']]+=1
    train_group[i['entity_id']]=int(np.argmax(count))
train_group1=[[] for _ in range(num_label)]
for i in result:
    train_group1[train_group[i['id']]].append(i)
diff_cat_chosen=[[] for _ in range(num_label*2)]
diff_cat_rejected=[[] for _ in range(num_label*2)]

for i in result:
    media_cat_chosen=np.zeros(num_label*2)
    media_weight_chosen=np.zeros(num_label*2)
    media_cat_rejected=np.zeros(num_label*2)
    media_weight_rejected=np.zeros(num_label*2)
    media_cp1=selected_cp[i['id']][0]*selected_cp[i['id']][1]
    media_cp2=rejected_cp[i['id']][0]*rejected_cp[i['id']][1]
    media_num_chosen=np.zeros(num_label*2)
    media_num_rejected=np.zeros(num_label*2)
    for j in range(num_label):
        media_num_chosen[2*j]=selected_cp[i['id']][1][j]
        media_num_chosen[2*j+1]=selected_cp[i['id']][1][j]
        media_num_rejected[2*j]=rejected_cp[i['id']][1][j]
        media_num_rejected[2*j+1]=rejected_cp[i['id']][1][j]
        if media_cp1[j]>0.0:
            media_cat_chosen[2*j]=1
            media_weight_chosen[2*j]=media_cp1[j]
            diff_cat_chosen[2*j].append(math.log(abs(media_cp1[j])))
        elif media_cp1[j]<0.0:
            media_cat_chosen[2*j+1]=1
            media_weight_chosen[2*j+1]=abs(media_cp1[j])
            diff_cat_chosen[2*j+1].append(math.log(abs(media_cp1[j])))
        if media_cp2[j]>0.0:
            media_cat_rejected[2*j]=1
            media_weight_rejected[2*j]=media_cp2[j]
            diff_cat_rejected[2*j].append(math.log(abs(media_cp2[j])))
        elif media_cp2[j]<0.0:
            media_cat_rejected[2*j+1]=1
            media_weight_rejected[2*j+1]=abs(media_cp2[j])
            diff_cat_rejected[2*j+1].append(math.log(abs(media_cp2[j])))
    i['cat_chosen']= list(media_cat_chosen)
    i['cat_rejected'] = list(media_cat_rejected)
    i['weight_chosen'] = list(media_weight_chosen)
    i['weight_rejected'] = list(media_weight_rejected)
    i['num_chosen']=list(media_num_chosen)
    i['num_rejected']=list(media_num_rejected)
for i in result:
    media_weight=[0 for _ in range(num_label*2)]
    for k in range(len(media_weight)):
        if i['cat_chosen'][k]>0:
            media_weight[k]=math.exp((math.log(i['weight_chosen'][k])-np.mean(diff_cat_chosen[k]))/2/np.std(diff_cat_chosen[k]))
    i['pow_chosen']=media_weight
for i in result:
    media_weight=[0 for _ in range(num_label*2)]
    for k in range(len(media_weight)):
        if i['cat_rejected'][k]>0:
            media_weight[k]=math.exp((math.log(i['weight_rejected'][k])-np.mean(diff_cat_rejected[k]))/2/np.std(diff_cat_rejected[k]))
    i['pow_rejected']=media_weight
#train=result
diff_cat=0
diff_cat_count=0
for i in result:
    diff_cat+=diff[i['id']]**0.5
    diff_cat_count+=1
for i in result:
    i['diff']=i['diff']**0.5/diff_cat*diff_cat_count
with open(args.output+'.json','w') as file:
    for i in result:
        file.write(json.dumps(i)+'\n')


