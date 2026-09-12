# ---------------------------------------------------------------
# Run this in a Colab cell first (not part of the script below):
#
# !pip install -q transformers datasets scikit-learn accelerate
#
# Runtime > Change runtime type > T4 GPU, before running anything.
# ---------------------------------------------------------------

"""
Finetune ModernBERT-large for 3-way single-label classification
(needed / redundant / irrelevant) on RAG-context relevance data.
Colab / T4 version — same pipeline as the M1 script, different
batch size and precision settings to match the GPU.

Expects a JSONL file where each line has the same shape as your example:
conversation_summary, recent_messages, query, candidates[{retrieved_context, label}]
"""

import json
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

set_seed(42)

assert torch.cuda.is_available(), "No GPU detected — check Runtime > Change runtime type > T4 GPU"

MODEL_ID = "answerdotai/ModernBERT-large"

# Option A: upload the file directly in the Colab file browser and point to it
DATA_PATH = "dataset.jsonl"  # <- update path

# Option B: mount Drive instead, if you want the same file your friend is using
# from google.colab import drive
# drive.mount("/content/drive")
# DATA_PATH = "/content/drive/MyDrive/<path-to-your-file>/dataset.jsonl"

label2id = {"redundant": 0, "needed": 1, "irrelevant": 2}
id2label = {v: k for k, v in label2id.items()}


# ---------------------------------------------------------------
# 1. Load JSONL and flatten into one row per (group, candidate)
# ---------------------------------------------------------------
def load_flat_examples(path):
    flat = []
    with open(path, "r") as f:
        for group_id, line in enumerate(f):
            row = json.loads(line)

            conversation = "summary: " + row["conversation_summary"] + "\n"
            for message in row["recent_messages"]:
                conversation += message + "\n"
            conversation += row["query"] + "\n"

            for candidate in row["candidates"]:
                flat.append({
                    "conversation": conversation,
                    "context": candidate["retrieved_context"],
                    "label": label2id[candidate["label"]],
                    "group_id": group_id,
                })
    return flat


flat_examples = load_flat_examples(DATA_PATH)
print(f"Total flat examples: {len(flat_examples)}")

# ---------------------------------------------------------------
# 2. Group-level train/val split
# ---------------------------------------------------------------
group_ids = [ex["group_id"] for ex in flat_examples]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(splitter.split(flat_examples, groups=group_ids))

train_dataset = Dataset.from_list([flat_examples[i] for i in train_idx])
val_dataset = Dataset.from_list([flat_examples[i] for i in val_idx])

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ---------------------------------------------------------------
# 3. Tokenize
# ---------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize_fn(batch):
    return tokenizer(
        batch["conversation"],
        text_pair=batch["context"],
        truncation=True,
    )


remove_cols = ["conversation", "context", "group_id"]
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=3,
    label2id=label2id,
    id2label=id2label,
)

# ---------------------------------------------------------------
# 5. Metrics — same as the M1 script, macro-F1 as primary
# ---------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )

    metrics = {"f1_macro": macro_f1, "f1_micro": micro_f1}
    for idx, name in id2label.items():
        metrics[f"f1_{name}"] = per_class_f1[idx]
    return metrics


# ---------------------------------------------------------------
# 6. Training arguments — T4 has 16GB VRAM, more headroom than
#    M1 unified memory, so batch size can be higher. fp16=True
#    is a real speedup here (unlike on M1's MPS backend).
# ---------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./modernbert-context-classifier",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # effective batch size 16
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=10,
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------
# 8. Save final model + tokenizer
#    On Colab, the runtime is ephemeral — copy this to Drive
#    before the session disconnects, or it's gone.
# ---------------------------------------------------------------
trainer.save_model("./modernbert-context-classifier/final")
tokenizer.save_pretrained("./modernbert-context-classifier/final")

# Optional: persist to Drive so you don't lose it on disconnect
# import shutil
# shutil.copytree("./modernbert-context-classifier/final",
#                  "/content/drive/MyDrive/modernbert-context-classifier/final")