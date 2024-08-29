"""
This is a sample script to attempt to train a raw model to be instruction
following using QLoRA

Raw language models to experiment with:

    google/gemma-2b
    google/gemma-7b

    Qwen/Qwen2-0.5B
    Qwen/Qwen2-1.5B
    Qwen/Qwen2-7B

    tiiuae/falcon-11B

    meta-llama/Meta-Llama-3-8B

"""

access_token = "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"
import os
os.environ['HF_HOME'] = '/home/stefanwebb/models/hf'
import torch
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

torch.random.manual_seed(1)

# model_id = "/home/stefanwebb/models/llm/microsoft_Phi-3-small-8k-instruct"
# model_id = "/home/stefanwebb/models/llm/google_gemma-2b"
model_id = "/home/stefanwebb/models/llm/meta_llama3-8b"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='cuda',
    torch_dtype="auto", 
    # trust_remote_code=True,
    token=access_token
)

# peft_model_id = "/home/stefanwebb/code/python/test-qwen2/stefans-gemma-2b-instruct-third-attempt/checkpoint-3250"
# peft_model_id = "/home/stefanwebb/code/python/test-qwen2/stefans-debug-llama3-chat/checkpoint-2500"
peft_model_id = "/home/stefanwebb/code/python/test-qwen2/stefans-debug-llama3-chat-bs-16-eval/checkpoint-6876"
model = PeftModel.from_pretrained(base_model, peft_model_id)
# model = base_model

# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

tokenizer.pad_token = "<|reserved_special_token_0|>"
tokenizer.pad_token_id = 128002

# tokenizer.eos_token = "<|end_of_text|>"
# tokenizer.eos_token_id = 128001

tokenizer.eos_token = "<|eot_id|>"
tokenizer.eos_token_id = 128009

# tokenizer.padding_side = 'right'

# conversation = Conversation("Who are you? Answer in a single sentence.")
# conversation = Conversation("Explain the difference between voluntary and involuntary manslaughter.")
# conversation = Conversation("What is your name?")

prompt = """<start_of_turn>system
You are Gemma.<end_of_turn>
<start_of_turn>user
What is an turnip?<end_of_turn>
<start_of_turn>assistant
"""

prompt2 = """<start_of_turn>system
You are Gemma.<end_of_turn>
<start_of_turn>user
What is an apple?<end_of_turn>
<start_of_turn>assistant
An apple is a type of fruit that is typically round or oval in shape, with a crunchy texture and a sweet or tart flavor. Apples can be eaten fresh, baked, or used in recipes.<end_of_turn>
<start_of_turn>user
Why was the last question I asked?<end_of_turn>
<start_of_turn>assistant
"""

prompt3 = """<start_of_turn>system
You are Gemma.<end_of_turn>
<start_of_turn>user
Calculate 50427950 + 107962.<end_of_turn>
<start_of_turn>assistant
"""

prompt4 = """<start_of_turn>system
You are Gemma, a helpful assistant. You reply concisely and truthfully.<end_of_turn>
<start_of_turn>user
Calculate 50427950 + 107962.<end_of_turn>
<start_of_turn>assistant
"""

prompt5 = """<start_of_turn>system
You are Gemma, a helpful assistant. You reply concisely and truthfully.<end_of_turn>
<start_of_turn>user
What is an emergent behavior?<end_of_turn>
<start_of_turn>assistant
"""

prompt6 = """<start_of_turn>system
You are Gemma, a helpful assistant. You reply concisely and truthfully.<end_of_turn>
<start_of_turn>user
How are you feeling?<end_of_turn>
<start_of_turn>assistant
I'm feeling great!<end_of_turn>
<start_of_turn>user
Why is that?<end_of_turn>
<start_of_turn>assistant
"""

def llama3_format_prompt(s):
        return f"<|start_header_id|>user<|end_header_id|>\n\n{s.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

prompt_llama3 = llama3_format_prompt("How many numbers are in the list 25, 26, ..., 100? (A) 75 (B) 76 (C) 22 (D) 23. Answer with a single letter.")

inputs = tokenizer(prompt_llama3, return_tensors="pt")
input_ids = inputs["input_ids"].cuda()

# print(input_ids)

# print(tokenizer(tokenizer.bos_token+tokenizer.eos_token))

generation_output = model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(do_sample=True, temperature=0.1, top_p=0.75, num_beams=4),
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=512,
    tokenizer=tokenizer,
    stop_strings=["<end_of_text>", "<|eot_id|>"],
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
    
)

print(len(generation_output.sequences))
for seq in generation_output.sequences:
    output = tokenizer.decode(seq)
    print(output)
    # print(seq)
    # print("Respuesta:", output.split("### Response:")[1].strip())