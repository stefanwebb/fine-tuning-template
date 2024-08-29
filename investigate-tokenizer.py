import os
os.environ['HF_HOME'] = '/home/stefanwebb/models/hf'
import torch
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

torch.random.manual_seed(0)

# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/stefanwebb/models/llm/google_gemma-2b")



prompt = "<start_of_turn>Testing<eos>"
inputs = tokenizer(prompt)

print(inputs, tokenizer.special_tokens_map)

tokenizer.decode()