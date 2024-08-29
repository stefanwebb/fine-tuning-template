from datasets import load_dataset
import multiprocessing

ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",   # test_prefs
        # num_proc=8,
        # batch_size=32
        # num_workers=4
        )

def format_prompt(s):
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{s.strip()}<|eot_id|>"

def format_completion(s):
        return f"<|start_header_id|>assistant<|end_header_id|>\n\n{s.strip()}<|eot_id|><|end_of_text|>"

def process(row):
        return {
                "prompt": format_prompt(row["prompt"]),
                "chosen": format_completion([x["content"] for x in row["chosen"] if x["role"] == "assistant"][0]),
                "rejected": format_completion([x["content"] for x in row["rejected"] if x["role"] == "assistant"][0]),
        }

ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
        remove_columns=["prompt_id", "messages", "score_chosen", "score_rejected"],
        # batched=True
)

print(ds)

# for x in ds:
#     print(x)
#     break

print(ds[0]) # ['chosen'][1]['content'])
print(ds[1])