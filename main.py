import os
os.environ['HF_HOME'] = '/home/stefanwebb/models/hf'
import torch
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM

torch.random.manual_seed(0)
# model_id = "microsoft/Phi-3-small-8k-instruct"
# model="/home/stefanwebb/models/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6", 
# model_id = "/home/stefanwebb/models/llm/Qwen_Qwen2-7B-Instruct"
model_id = "/home/stefanwebb/models/llm/microsoft_Phi-3-small-8k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='cuda',
    torch_dtype="auto", 
    trust_remote_code=True
)

# assert torch.cuda.is_available(), "This model needs a GPU to run ..."
# device = torch.cuda.current_device()
# model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# conversation = Conversation("Who are you? Answer in a single sentence.")
# conversation = Conversation("Explain the difference between voluntary and involuntary manslaughter.")
conversation = Conversation("What are some differences between common law in the US and UK?")

# model="Qwen/Qwen2-7B-Instruct"
pipe = pipeline("conversational", 
                model=model,
                tokenizer=tokenizer #,
                #framework ="pt",              
                )

conversation = pipe(conversation, max_new_tokens=1024)
print(conversation.messages[-1])