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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer

torch.random.manual_seed(0)

# model_id = "/home/stefanwebb/models/llm/microsoft_Phi-3-small-8k-instruct"
MODEL_ID = "google/gemma-2b"
NEW_MODEL_NAME = "stefans-gemma-2b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='cuda',
    torch_dtype="auto", 
    # trust_remote_code=True,
    token=access_token
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=access_token)

DATASET_NAME = "tatsu-lab/alpaca"
SPLIT = "train"
MAX_SEQ_LENGTH = 8192 # 2048

num_train_epochs = 1
learning_rate = 1.41e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 1

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

dataset = load_dataset(DATASET_NAME, split="train")

EOS_TOKEN = tokenizer.eos_token_id

# Select a subset of the data for faster processing
dataset = dataset.select(range(100))

"""
<start_of_turn>system
You are Gemma.<end_of_turn>
<start_of_turn>user
Give three tips for staying healthy.<end_of_turn>
<start_of_turn>assistant\n1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.<end_of_turn>
<eos>']

"""

# Define a function to format the prompts in the dataset.
# This function takes a batch of examples and returns a dictionary with the key 'text' and the value being a list of formatted texts.
def formatting_prompts_func(examples):
    # Extract the conversations from the examples.
    convos = examples["conversations"]
    # Initialize an empty list to store the formatted texts.
    texts = []
    # Define a dictionary to map the 'from' field in the conversation to a prefix.
    mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
    # Define a dictionary to map the 'from' field in the conversation to a suffix.
    end_mapper = {"system": "", "human": "", "gpt": ""}
    # Iterate over each conversation.
    for convo in convos:
        # Format the conversation by joining each turn with its corresponding prefix and suffix.
        # Append the EOS token to the end of the conversation.
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
        texts.append(f"{text}{EOS_TOKEN}")
    # Return the formatted texts.
    return {"text": texts}

# Apply the formatting function to the dataset using the map method.
# The 'batched=True' argument means that the function is applied to batches of examples.
dataset = dataset.map(formatting_prompts_func, batched=True)

# Print the 9th example from the 'text' field of the dataset to check the result.
print(dataset['text'][8])