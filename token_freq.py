import os
from transformers import AutoTokenizer
import torch
from dataloader import DiffusionLoader
import numpy as np
import diffusion_token_freq
import math
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('/data/luoyc/MolGen-large')
train_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='zinc250k', splits=['train'])[0]
token_freq = torch.zeros((len(tokenizer),), dtype=torch.int64)
print(len(tokenizer))

for data in tqdm(train_data):
    for iid in data['input_ids']:
        token_freq[iid] += 1

if not os.path.exists('./token_freq'):
    os.mkdir('token_freq')

torch.save(token_freq, f'./token_freq/MolGen-large_zinc250k.pt')
