# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:07:21 2024

@author: jacki
"""

import os
import gc
import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer,DPOConfig
import bitsandbytes as bnb
from huggingface_hub import login
import argparse
argparser=argparse.ArgumentParser()
argparser.add_argument('model_path')
argparser.add_argument('adapter_path')
args = argparser.parse_args()
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, # location of saved SFT model
    torch_dtype=torch.float16,
    token=''
)
tokenizer=AutoTokenizer.from_pretrained(args.adapter_path)
model = PeftModel.from_pretrained(model, args.adapter_path)
model.eval()
model = model.merge_and_unload()
model.save_pretrained('merged')
tokenizer.save_pretrained('merged')
