"""
Codes are modified from 
@article{xu2024llm,
  title={LLM Jailbreak Attack versus Defense Techniques--A Comprehensive Study},
  author={Xu, Zihao and Liu, Yi and Deng, Gelei and Li, Yuekang and Picek, Stjepan},
  journal={arXiv preprint arXiv:2402.13457},
  year={2024}
}

"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name_list = {
    'llama':"meta-llama/Llama-2-7b-chat-hf",
    'vicuna':"lmsys/vicuna-7b-v1.5"}

model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_path ="./models/meta-llama/Llama-2-7b-chat-hf"

def download(name):
    model_name = model_name_list[name]
    base_model_path =  f"./models/{model_name_list[name]}"
    print("making directory")
    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True, use_cache=False)
    #Save the model and the tokenizer
    model.save_pretrained(base_model_path, from_pt=True)
    tokenizer.save_pretrained(base_model_path, from_pt=True)
    print("done")

download('llama')
