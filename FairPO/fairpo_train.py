# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:25:55 2024

@author: jacki
"""

import os
import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer,DPOConfig
from huggingface_hub import login
import argparse
from torch.utils.data import DataLoader
from transformers.utils import is_datasets_available
import datasets
import tqdm
from transformers.integrations.tpu import tpu_spmd_dataloader
import math
import sys
import torch.nn.functional as F
from trainer_fairpo import FairPOTrainer

argparser=argparse.ArgumentParser()
argparser.add_argument('--train_data',type=str)
argparser.add_argument('--model',type=str)
argparser.add_argument('--output_path',type=str)
argparser.add_argument('--batch_size',type=int)
argparser.add_argument('--gradient_accumulation_steps',type=int)
argparser.add_argument('--num_bias_label',type=int)
argparser.add_argument('--weight_schema',type=str)
argparser.add_argument('--weight_step',type=float)
argparser.add_argument('--temperature',type=float)

args = argparser.parse_args()


train_data=load_dataset('json',data_files=args.train_data)['train']
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",],
    bias="none",
    task_type="CAUSAL_LM",
)
tokenizer = AutoTokenizer.from_pretrained(args.model, token='')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
max_len=-1
model = AutoModelForCausalLM.from_pretrained(
    args.model, 
    torch_dtype=torch.float16,
    load_in_8bit=True,
    token=''
)
model.config.use_cache = False
training_args = DPOConfig(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=2,
    save_strategy="no",
    logging_steps=5,
    output_dir='test_dpo',
    optim="paged_adamw_32bit",
    bf16=True,
    warmup_ratio=0.5,
    seed=int(args.output_path[-1])
)


dpo_trainer = FairPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    num_bias_label=args.num_bias_label, 
    weight_schema=args.weight_schema,
    weight_step=args.weight_step, 
    temperature=args.temperature,
    peft_config=peft_config,
    beta=0.1,
    max_length=1536,
    max_prompt_length=1024,
    
)
dpo_trainer.train()
dpo_trainer.model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)