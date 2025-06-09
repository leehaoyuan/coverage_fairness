# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:28:58 2024

@author: jacki
"""

import json
import random
import copy
import numpy as np
import pickle
import argparse
random.seed(0)
argparser=argparse.ArgumentParser()
argparser.add_argument('--input_dataset',type=str)
argparser.add_argument('--seed',type=int,default=0)
argparser.add_argument('--input_ec_file',type=str)
argparser.add_argument('--perturbation_rate',type=float,default=0.1)
args = argparser.parse_args()
with open(args.input_dataset+'.pickle','rb') as file:
    data=pickle.load(file)
#data=data[0]+data[1]+data[2]
random.shuffle(data)
#data=data[:100]
with open(args.input_ec_file+'.pickle','rb') as file:
    shuffle=pickle.load(file)
for i in range(len(data)):
    random.shuffle(data[i]['reviews'])
data_order=copy.deepcopy(data)
data_middle=copy.deepcopy(data)
#can_count
not_in_key=[]
same_id=0
for i in range(len(data_order)):
    if data_order[i]['entity_id'] in shuffle.keys():
        perturb_num_doc=round(len(data_order[i]['reviews'])*args.perturbation_rate)
        count=np.zeros(3)
        for j in data_order[i]['reviews']:
            count[j['label']]+=1
        arg_idx=np.argsort(-shuffle[data_order[i]['entity_id']][1][0])
        for j in arg_idx:
            if count[j]>3:
                can=j
                break
        can_media=can
        #candidates=[j for j in range(count.shape[0]) if count[j]>0 and count[j]<len(data_order[i]['reviews'])-1]
        #can=random.choice(candidates)
        can_review=[j for j in data_order[i]['reviews'] if j['label']==can]
        other_review=[j for j in data_order[i]['reviews'] if j['label']!=can]
        random.shuffle(can_review)
        random.shuffle(other_review)
        can_review=can_review[:-perturb_num_doc]
        data_order[i]['reviews']=can_review+other_review
        random.shuffle(data_order[i]['reviews'])
        arg_idx=np.argsort(shuffle[data_order[i]['entity_id']][1][0])
        for j in arg_idx:
            if count[j]>perturb_num_doc:
                can=j
                break
        if can==can_media:
            same_id+=1
        can_review_id=set([j['review_id'] for j in can_review])
        can_review1=[j for j in data_middle[i]['reviews'] if j['label']==can]
        other_review=[j for j in data_middle[i]['reviews'] if j['label']!=can]
        random.shuffle(can_review1)
        random.shuffle(other_review)
        can_review1=can_review1[:-perturb_num_doc]
        can_review1_id=set([j['review_id'] for j in can_review1])
        while can_review_id==can_review1_id:
            #print('a')
            can_review1=[j for j in data_middle[i]['reviews'] if j['label']==can]
            random.shuffle(can_review1)
            can_review1=can_review1[perturb_num_doc:]
            can_review1_id=set([j['review_id'] for j in can_review1])
        data_middle[i]['reviews']=can_review1+other_review
        random.shuffle(data_middle[i]['reviews'])
        assert len(data_middle[i]['reviews'])==np.sum(count)-perturb_num_doc
    else:
        not_in_key.append(i)
        
data_order=[data_order[i] for i in range(len(data_order)) if i not in not_in_key]
data_middle=[data_middle[i] for i in range(len(data_middle)) if i not in not_in_key]
print(same_id)
#assert 0==1
with open(args.input_dataset+'_order.pickle','wb') as file:
    pickle.dump(data_order,file)
with open(args.input_dataset+'_middle.pickle','wb') as file:
    pickle.dump(data_middle,file)