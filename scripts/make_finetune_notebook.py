"""Write finetune/slm-supervisor/finetune_qwen35_4b.ipynb (the Colab training notebook).

    python scripts/make_finetune_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "finetune" / "slm-supervisor" / "finetune_qwen35_4b.ipynb"

CELLS: list[tuple[str, str]] = [
    ("markdown", """# Fine-tune the supervisor SLM (Qwen3.5 4B) on the fair-campaign decisions

Runs on a Colab GPU (T4 16 GB is enough). Steps:

1. install Unsloth, clone the public repo and build the dataset with `scripts/slm_finetune_dataset.py`
   (CIFAR-10 is the held-out test set; its runs never inform the teacher);
2. score the **base** model on held-out decisions;
3. LoRA fine-tune in 16-bit (Unsloth advises against 4-bit QLoRA for Qwen3.5);
4. score the **fine-tuned** model on the same decisions;
5. save the LoRA adapter and a merged GGUF (`q4_k_m`) and upload them to a private Hugging Face repo.

Before running, add a Hugging Face **write** token as the Colab secret `HF_TOKEN` (key icon on the left,
with notebook access on). Every result is uploaded as soon as it exists, so a Colab disconnect never
loses the trained adapter. Run the cells top to bottom."""),
    ("code", """!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """%%capture
!pip install --upgrade --no-cache-dir unsloth unsloth_zoo
# Qwen3.5 needs Transformers v5 (Unsloth docs); Colab ships an older one.
!pip install --upgrade "transformers>=5.0.0" jsonschema"""),
    ("code", """import importlib.metadata as md
for package in ("unsloth", "unsloth_zoo", "transformers", "trl", "torch"):
    print(package, md.version(package))"""),
    ("code", """import os
if not os.path.isdir("dfl-slm-dtn"):
    !git clone --depth 1 https://github.com/Hammadwakeel/dfl-slm-dtn.git
%cd dfl-slm-dtn
!python scripts/slm_finetune_dataset.py
!cat finetune/slm-supervisor/README.md"""),
    ("code", """# Settings
MODEL = "unsloth/Qwen3.5-4B"
MAX_SEQ_LENGTH = 3072
N_TRAIN = 2000       # stratified sample of train.jsonl (each example is ~1.3k tokens)
N_VAL = 200
N_EVAL = 150         # held-out CIFAR-10 decisions scored before and after training
EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_RANK = 16
SEED = 3407
HF_REPO_NAME = "qwen35-4b-dfl-supervisor"   # private repo under your Hugging Face account"""),
    ("code", """# Log in with the Colab secret HF_TOKEN and create the private repo before the long steps.
from google.colab import userdata
from huggingface_hub import HfApi, login
login(token=userdata.get("HF_TOKEN"))
api = HfApi()
REPO = api.whoami()["name"] + "/" + HF_REPO_NAME
api.create_repo(REPO, private=True, exist_ok=True)

def upload(path, path_in_repo=None):
    \"\"\"Upload a file or folder to the private repo right away.\"\"\"
    if os.path.isdir(path):
        api.upload_folder(repo_id=REPO, folder_path=path, path_in_repo=path_in_repo or path)
    else:
        api.upload_file(repo_id=REPO, path_or_fileobj=path, path_in_repo=path_in_repo or os.path.basename(path))
    print("uploaded", path, "to", REPO)
print("Hugging Face repo:", REPO)"""),
    ("code", """import json, random, re
from collections import Counter, defaultdict

def load(name):
    with open(f"finetune/slm-supervisor/{name}.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]

def stratified(rows, n, seed=SEED):
    \"\"\"Sample n rows with every (mode, label) group represented in proportion.\"\"\"
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["meta"]["mode"], row["meta"]["label"])].append(row)
    picked = []
    for key, items in groups.items():
        rng.shuffle(items)
        picked += items[: max(1, round(n * len(items) / len(rows)))]
    rng.shuffle(picked)
    return picked[:n]

train_rows = stratified(load("train"), N_TRAIN)
val_rows = stratified(load("val"), N_VAL)
eval_rows = stratified(load("test"), N_EVAL)
print(len(train_rows), len(val_rows), len(eval_rows))
print(Counter(r["meta"]["label"] for r in train_rows))"""),
    ("code", """from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)"""),
    ("code", """def prompt_text(row):
    return text_tokenizer.apply_chat_template(
        row["messages"][:-1], tools=row["tools"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )

def completion_text(row):
    \"\"\"The assistant turn exactly as the chat template renders the tool call.\"\"\"
    full = text_tokenizer.apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False, enable_thinking=False,
    )
    prompt = prompt_text(row)
    assert full.startswith(prompt), "chat template renders the prompt differently in the full conversation"
    # "<tool_call>\\n<function=...>...</tool_call><|im_end|>": the model learns to stop too.
    return full[len(prompt):].rstrip("\\n")

example = train_rows[0]
print(prompt_text(example)[-600:])
print("---- completion ----")
print(completion_text(example))
lengths = [len(text_tokenizer(prompt_text(r) + completion_text(r))["input_ids"]) for r in train_rows[:200]]
print("tokens: mean", sum(lengths) / len(lengths), "max", max(lengths))"""),
    ("code", """FUNCTION = re.compile(r"<function=([A-Za-z_]+)>|\\"name\\"\\s*:\\s*\\"([A-Za-z_]+)\\"")

def predicted_step(text):
    match = FUNCTION.search(text)
    return (match.group(1) or match.group(2)) if match else None

def evaluate(rows, label):
    FastLanguageModel.for_inference(model)
    results = []
    for i, row in enumerate(rows):
        inputs = text_tokenizer(prompt_text(row), return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        text = text_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        step = predicted_step(text)
        results.append({"mode": row["meta"]["mode"], "label": row["meta"]["label"], "pred": step,
                        "valid": step in row["meta"]["allowed"], "text": text})
        if i % 25 == 0:
            print(i, row["meta"]["label"], "->", step)
    def score(items):
        n = max(1, len(items))
        return {"n": len(items), "valid_tool": round(sum(r["valid"] for r in items) / n, 3),
                "matches_teacher": round(sum(r["pred"] == r["label"] for r in items) / n, 3)}
    report = {"all": score(results)}
    for mode in ("fair", "budget"):
        report[mode] = score([r for r in results if r["mode"] == mode])
    print(label, json.dumps(report, indent=1))
    with open(f"eval-{label}.json", "w") as handle:
        json.dump({"report": report, "results": results}, handle)
    return report

base_report = evaluate(eval_rows, "base")
upload("eval-base.json")"""),
    ("code", """model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

from datasets import Dataset
def pairs(rows):
    return Dataset.from_list([{"prompt": prompt_text(r), "completion": completion_text(r)} for r in rows])
train_dataset, val_dataset = pairs(train_rows), pairs(val_rows)"""),
    ("code", """from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    processing_class=text_tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir="outputs",
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,   # loss only on the tool call, not the prompt
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=SEED,
        report_to="none",
        dataset_num_proc=1,
    ),
)
stats = trainer.train()
print(stats)
# Save and upload the adapter at once: a Colab disconnect after this point loses nothing.
model.save_pretrained("qwen35-4b-dfl-supervisor-lora")
text_tokenizer.save_pretrained("qwen35-4b-dfl-supervisor-lora")
upload("qwen35-4b-dfl-supervisor-lora", "lora")"""),
    ("code", """finetuned_report = evaluate(eval_rows, "finetuned")
print("base     ", base_report)
print("finetuned", finetuned_report)
upload("eval-finetuned.json")"""),
    ("code", """# Merged GGUF (q4_k_m) for Ollama on the master Jetson.
# Unsloth writes the merged 16-bit weights to the folder below and the GGUF to "<folder>_gguf".
import glob
model.save_pretrained_gguf("qwen35-4b-dfl-supervisor-gguf", text_tokenizer, quantization_method="q4_k_m")
GGUF = glob.glob("qwen35-4b-dfl-supervisor-gguf_gguf/*Q4_K_M.gguf")[0]
!ls -la qwen35-4b-dfl-supervisor-gguf_gguf
upload(GGUF, "gguf/qwen35-4b-dfl-supervisor-Q4_K_M.gguf")"""),
]


def main() -> None:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
