access_token = "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"
import os
os.environ['HF_HOME'] = '/home/stefanwebb/models/hf'
import torch
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

torch.random.manual_seed(0)

# model_id = "/home/stefanwebb/models/llm/microsoft_Phi-3-small-8k-instruct"
# model_id = "/home/stefanwebb/models/llm/google_gemma-2b"
# model_id = 'google/gemma-7b'

model_ids = [
    # Yy'google/gemma-7b-it',
    'microsoft/Phi-3-small-8k-instruct',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-v0.3',
    'mistralai/Mistral-7B-Instruct-v0.3'
    ]

for model_id in model_ids:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token, trust_remote=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='cuda',
        torch_dtype="auto", 
        trust_remote_code=True,
        token=access_token
    )
    del base_model
    del tokenizer

# peft_model_id = "/home/stefanwebb/code/python/test-qwen2/stefans-gemma-2b-instruct-third-attempt/checkpoint-3250"
# model = PeftModel.from_pretrained(base_model, peft_model_id)

# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.padding_side = 'right'
