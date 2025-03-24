# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:09:53 2023

@author: jacki
"""

import torch
import itertools
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import argparse
import networkx as nx
import copy
argparser=argparse.ArgumentParser()
argparser.add_argument('--model',type=str)
argparser.add_argument('--input_dataset',type=str)
argparser.add_argument('--input_acu',type=str)
argparser.add_argument('--block_len',type=int,default=100)
argparser.add_argument('--random_time',type=int,default=1000)
args = argparser.parse_args()

device=torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained(args.model)
nli_model = AutoModelForSequenceClassification.from_pretrained(args.model)
nli_model=nli_model.to(device)
nli_model.eval()
with open(args.input_acu+'.json','r') as file:
    acus=json.load(file)
with open(args.input_dataset+'.pickle','rb') as file:
    data=pickle.load(file)

block_len=args.block_len
sents={}
ids={}
for i in data:
    sents[i['entity_id']]=[]
    ids[i['entity_id']]=[]
    counter=0
    for j in i['reviews']:
        for k in range(len(j['sentences'])):
            media_block=[j['sentences'][k]]
            dist=1
            max_len=False
            while len(' '.join(media_block).split())<block_len and (k-dist>=0 or k+dist<len(j['sentences'])):
                candidate=[]
                if k-dist>=0:
                    candidate.append(k-dist)
                if k+dist<len(j['sentences']):
                    candidate.append(k+dist)
                min_idx=np.argsort([len(j['sentences'][s].split()) for s in candidate])
                for idx in min_idx:
                    if len(j['sentences'][candidate[idx]].split())+len(' '.join(media_block).split())<block_len:
                        if candidate[idx]<k:
                            media_block=[j['sentences'][candidate[idx]]]+media_block
                        else:
                            media_block.append(j['sentences'][candidate[idx]])
                    else:
                        max_len=True
                        break
                if max_len:
                    break
                dist+=1
            #print(media_block)
            media_block=' '.join(media_block)
            if media_block not in sents[i['entity_id']]:
                sents[i['entity_id']].append(media_block)
                ids[i['entity_id']].append(counter)
        counter+=1
result={} 
batch_size=128

with torch.no_grad():
    for i in acus.keys():
        #print(i)
        entailment_map=np.zeros((len(acus[i]),len(sents[i]),3),dtype=float)
        pairs=list(itertools.product(list(range(len(acus[i]))),list(range(len(sents[i])))))
        #pairs=pairs+list(itertools.product(neg_pair,neg_sent))
        #print(pairs)
        counter=0
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            #print(media_pair)
            pre=[sents[i][j[1]] for j in media_pair]
            hyp=[acus[i][j[0]] for j in media_pair]
            #hyp=[media[j[0]] for j in media_pair]
            sample=tokenizer(pre,hyp,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            #print(logits.size())
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs.detach().cpu().numpy()
            for j in range(len(media_pair)):
                entailment_map[media_pair[j][0],media_pair[j][1]]=prob_label_is_true[j]
            #break
            counter+=batch_size
        result[i]=(entailment_map,ids[i])
        #break
###obtain entailment map between summary itself
result_self={}
with torch.no_grad():
    for i in acus.keys():
        #print(i)
        entailment_map=np.zeros((len(acus[i]),len(acus[i]),3),dtype=float)
        pairs=list(itertools.product(list(range(len(acus[i]))),list(range(len(acus[i])))))
        counter=0
        while counter<len(pairs):
            media_pair=pairs[counter:counter+batch_size]
            pre=[acus[i][j[1]] for j in media_pair]
            hyp=[acus[i][j[0]] for j in media_pair]
            #hyp=[media[j[0]] for j in media_pair]
            sample=tokenizer(pre,hyp,truncation=True,padding=True,return_tensors='pt')
            logits=nli_model(**sample.to(device))[0]
            #print(logits.size())
            probs = logits.softmax(dim=1)
            prob_label_is_true = probs.detach().cpu().numpy()
            for j in range(len(media_pair)):
                entailment_map[media_pair[j][0],media_pair[j][1]]=prob_label_is_true[j]
            #break
            counter+=batch_size
        result_self[i]=entailment_map
        #break
##evaluate_fairness
data1={}
for i in data:
    if i['entity_id'] in result.keys():
        data1[i['entity_id']]=i['reviews']
entail_idx=2
ent_diff=[]
not_cover=0
all_count=0
if 'semeval'  in args.input_dataset or 'mitweet' in args.input_dataset:
    num_label=2
else:
    num_label=3
all_ratio_list=[[] for i in range(num_label)]
ec_result={}
for i in result.keys():
    self_entailment_map=np.zeros((result[i][0].shape[0],result[i][0].shape[0]))
    for j in range(self_entailment_map.shape[0]):
        for k in range(self_entailment_map.shape[1]):
            if result_self[i][j,k,entail_idx]>0.95:
                self_entailment_map[j,k]=result_self[i][j,k,entail_idx]
    counter=0
    graph=nx.Graph()
    for j in range(self_entailment_map.shape[0]):
        for k in range(j+1,self_entailment_map.shape[1]):
            if self_entailment_map[j,k]>0 and self_entailment_map[k,j]>0:
                graph.add_edge(j,k)
                counter+=1
    in_cc=np.zeros(len(acus[i]))
    cc_list=[]
    
    if counter>0:
        #print(i)
        ccs=nx.connected_components(graph)
        for j in ccs:
            #print([acu[i][k] for k in j])
            cc_list.append(list(j))
            for k in j:
                in_cc[k]=1
    for j in range(in_cc.shape[0]):
        if in_cc[j]==0:
            cc_list.append([j])

    entailment_map=np.zeros((len(cc_list),result[i][0].shape[1]))
    for j in range(entailment_map.shape[0]):
        for k in range(entailment_map.shape[1]):
            for l in cc_list[j]:
                if np.argmax(result[i][0][l,k])==entail_idx and result[i][0][l,k,entail_idx]>entailment_map[j,k]:
                    entailment_map[j,k]=result[i][0][l,k,entail_idx]
                    #entailment_map[j,k]=1
    
    media=np.max(entailment_map,axis=1)
    not_in_idx=[]
    in_count=0
    for j in range(media.shape[0]):
        all_count+=1
        if media[j]==0.0:
            #print(acu[i][j])
            not_cover+=1
            not_in_idx.append(j)
        else:
            in_count+=1
    if in_count==0:
        print(i)
        #assert 0==1
        continue
    entailment_map_doc=np.zeros((entailment_map.shape[0],len(data1[i])))
    entailment_map_doc_count=np.zeros((entailment_map.shape[0],len(data1[i])))
    sent_pos_idx=result[i][1]
    bias_idx=[]
    for j in data1[i]:
        bias_idx.append(j['label'])
    for j in range(entailment_map.shape[0]):
        for k in range(entailment_map.shape[1]):
            if entailment_map[j,k]>entailment_map_doc[j,sent_pos_idx[k]]:
                entailment_map_doc[j,sent_pos_idx[k]]=entailment_map[j,k]
            if entailment_map[j,k]>0:
                entailment_map_doc_count[j,sent_pos_idx[k]]+=1
    media_mean=np.sum(entailment_map_doc,axis=0)
    media_mean=media_mean/in_count
    #print(media_mean)
    overall_mean=np.mean(media_mean)
    bias_ratios=np.zeros(num_label)
    bias_count=np.zeros(num_label)
    for j in range(media_mean.shape[0]):
        bias_ratios[bias_idx[j]]+=media_mean[j]
        bias_count[bias_idx[j]]+=1
    for j in range(num_label):
        if bias_count[j]!=0:
            bias_ratios[j]=bias_ratios[j]/bias_count[j]
    media_diff=0.0
    counter=0
    for j in range(num_label):
        if bias_count[j]!=0:
            media_diff+=abs(bias_ratios[j]-overall_mean)
            counter+=1
    #if media_diff>1e-5:
    if np.max(bias_count)!=media_mean.shape[0]:
       media_diff=media_diff/counter
       ent_diff.append(media_diff)
       for j in range(media_mean.shape[0]):
           all_ratio_list[bias_idx[j]].append(media_mean[j]-overall_mean)
    else:
        print(i)
      # assert 0==1

#print('no'not_cover/all_count)
#print(all_count)
print('Equal Coverage: '+str(round(np.mean(ent_diff),4)))
all_ratio_list=[i for i in all_ratio_list if len(i)>0]
all_mean=[]
all_ratio_list1=np.zeros(num_label)
for i in range(len(all_ratio_list)):
    all_mean.extend(all_ratio_list[i])
    all_ratio_list1[i]=np.mean(all_ratio_list[i])
all_mean=np.mean(all_mean)
max_idx=np.argmax(all_ratio_list1)
min_idx=np.argmin(all_ratio_list1)
print('Coverage Parity: '+str(round(np.mean(np.abs(all_ratio_list1)),4)))
attribute_list=None
if 'amazon' in args.input_dataset:
    attribute_list=['negative','neutral','positive']
elif 'mitweet' in args.input_dataset:
    attribute_list=['left','center','right']
elif 'article_bias' in args.input_dataset:
    attribute_list=['left','center','right']
elif 'news_stance' in args.input_dataset:
    attribute_list=['support','against']
elif 'semeval' in args.input_dataset:
    attribute_list=['support','against']
media_sequence=''
for i in range(len(attribute_list)):
    media_sequence+=attribute_list[i]+': '+str(round(all_ratio_list1[i],4))+' '
print('Coverage Probability Difference: '+media_sequence)
#print(all_ratio_list1)
greater_time=0
random_time=args.random_time
for i in range(random_time):
    sample=np.random.choice(all_ratio_list[max_idx],size=len(all_ratio_list[max_idx]))
    if np.sum(sample)>0:
        greater_time+=1
print('Most overrepresented social attribute value: '+attribute_list[max_idx]+' significance: '+str(round(1.0-greater_time/random_time,4)))
less_time=0
for i in range(random_time):
    sample=np.random.choice(all_ratio_list[min_idx],size=len(all_ratio_list[min_idx]))
    if np.sum(sample)<0:
        less_time+=1
print('Most underrepresented social attribute value: '+attribute_list[min_idx]+' significance: '+str(round(1.0-less_time/random_time,4)))
