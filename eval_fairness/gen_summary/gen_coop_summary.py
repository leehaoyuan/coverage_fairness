# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:34:41 2024

"""

import json
from typing import List
import pandas as pd
import torch
from coop import VAE, util
import pickle

model_name: str = f"megagonlabs/bimeanvae-amzn"  # or f"megagonlabs/optimus-{task}"
vae = VAE(model_name)
sent_dict={}
with open('amazon300v3.pickle','rb') as file:
    data=pickle.load(file)
result={}
for i in data:
    sent_dict[i['entity_id']]=[' '.join(j['sentences']) for j in i['reviews']]
for i in sent_dict.keys():
    reviews: List[str] = sent_dict[i]
    z_raw: torch.Tensor = vae.encode(reviews)
    idxes: List[List[int]] = util.powerset(len(reviews))
    zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes])
    outputs: List[str] = vae.generate(zs, bad_words=util.BAD_WORDS,max_tokens=128)
    best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
    result[i]=best
with open('amazon300v3_coop_summ.pickle','wb') as file:
    pickle.dump(result,file)