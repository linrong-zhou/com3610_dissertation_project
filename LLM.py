import random
import numpy as np
import evaluate
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

# =========================================================
# Configuration
# =========================================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256
TRAIN_SAMPLES = 10_000
OOD_MAX_SAMPLES = 1_000
RANDOM_SEEDS = [13, 42, 100]
NUM_LABELS = 3
OUTPUT_CSV = "ood_evaluation_results_deberta.csv"

# =========================================================
# 1. Dataset loading helpers
# =========================================================
def load_hf_dataset(description, split_name):
    """Load and normalize NLI datasets to: premise, hypothesis, label."""
    print(f"Loading {description} ({split_name})...")

    if description == "pietrolesci/nli_fever":
        ds = load_dataset(description, split=split_name)
        ds = ds.rename_column("premise", "new_hypothesis")
        ds = ds.rename_column("hypothesis", "premise")
        ds = ds.rename_column("new_hypothesis", "hypothesis")
        return ds

    if description == "alisawuffles/WANLI":
        ds = load_dataset(description, split=split_name)
        labels = [0 if g == "entailment" else 1 if g == "neutral" else 2 for g in ds["gold"]]
        ds = ds.add_column("label", labels)
        return ds

    if description == "allenai/scitail":
        ds = load_dataset(description, "snli_format", split=split_name)
        labels = [0 if g in ["entails", "entailment"] else 1 for g in ds["gold_label"]]
        ds = ds.add_column("label", labels)
        ds = ds.rename_column("sentence1", "premise")
        ds = ds.rename_column("sentence2", "hypothesis")
        return ds

    return load_dataset(description, split=split_name)


def keep_valid_nli_labels(ds):
    return ds.filter(lambda x: x["label"] in [0, 1, 2])


# =========================================================
# 2. OOD datasets
# =========================================================
ood_configs = [
    ("snli", "test"),
    ("multi_nli", "validation_matched"),
    ("anli", "test_r1"),
    ("allenai/scitail", "test"),
    ("alisawuffles/WANLI", "test"),
]

ood_datasets = {}
for desc, split in ood_configs:
    ds = keep_valid_nli_labels(load_hf_dataset(desc, split))
    if len(ds) > OOD_MAX_SAMPLES:
        ds = ds.select(range(OOD_MAX_SAMPLES))
    ood_datasets[f"{desc}_{split}"] = ds

# =========================================================
# 3. Tokenizer, collator, metrics
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
acc_metric = evaluate.load("accuracy")


def tokenize_fn(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions=preds, references=labels)


print("Tokenizing OOD sets...")
tok_ood = {
    k: v.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in v.column_names if c not in ["label"]],
    )
    for k, v in ood_datasets.items()
}

# =========================================================
# 4. Base SNLI train/validation data
# =========================================================
print("Loading base SNLI train/validation data...")
snli_train_full = keep_valid_nli_labels(load_dataset("snli", split="train"))
snli_val = keep_valid_nli_labels(load_dataset("snli", split="validation"))

if len(snli_val) > 3000:
    snli_val = snli_val.shuffle(seed=13).select(range(3000))

all_results = []

for run_idx, seed in enumerate(RANDOM_SEEDS):
    print(f"\n{'=' * 90}")
    print(f"RUN {run_idx + 1}/{len(RANDOM_SEEDS)} | Seed={seed}")
    print(f"{'=' * 90}")

    set_seed(seed)
    random.seed(seed)

    # Balanced 10k sample from SNLI train
    print("Sampling balanced 10k SNLI training split...")
    c0 = snli_train_full.filter(lambda x: x["label"] == 0).shuffle(seed=seed).select(range(3333))
    c1 = snli_train_full.filter(lambda x: x["label"] == 1).shuffle(seed=seed).select(range(3333))
    c2 = snli_train_full.filter(lambda x: x["label"] == 2).shuffle(seed=seed).select(range(3334))
    train_10k = concatenate_datasets([c0, c1, c2]).shuffle(seed=seed)

    tok_train = train_10k.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in train_10k.column_names if c not in ["label"]],
    )
    tok_val = snli_val.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in snli_val.column_names if c not in ["label"]],
    )

    print("Initializing DeBERTa model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    args = TrainingArguments(
        output_dir=f"./results_deberta_seed_{seed}",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        bf16=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating on OOD datasets...")
    split_res = {"Split_Seed": seed}
    for name, ds in tok_ood.items():
        preds = trainer.predict(ds)
        acc = preds.metrics["test_accuracy"]
        split_res[name] = acc
        print(f"  {name}: {acc:.4f}")

    all_results.append(split_res)

# =========================================================
# 5. Results table and plots
# =========================================================
df_results = pd.DataFrame(all_results).set_index("Split_Seed")
df_mean = df_results.mean().to_frame(name="Average_Acc").T
df_std = df_results.std().to_frame(name="Std_Acc").T
df_final = pd.concat([df_results, df_mean, df_std])

print("\n" + "=" * 80)
print("FINAL OOD EVALUATION TABLE")
print("=" * 80)
print(df_final.to_markdown(floatfmt=".4f"))
print("=" * 80 + "\n")

df_final.to_csv(OUTPUT_CSV)
print(f"Saved results to {OUTPUT_CSV}")

plt.figure(figsize=(10, 5))
df_results.T.plot(kind="bar", ax=plt.gca())
plt.title("Accuracy by Random Split across OOD Datasets")
plt.ylabel("Accuracy")
plt.xlabel("Dataset")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Split Seed")
plt.tight_layout()
plt.savefig("ood_accuracy_by_split.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 4))
df_mean.T["Average_Acc"].sort_values().plot(kind="barh")
plt.title("Average Accuracy Across Splits")
plt.xlabel("Mean Accuracy")
plt.xlim(0, 1.0)
for index, value in enumerate(df_mean.T["Average_Acc"].sort_values()):
    plt.text(value, index, f" {value:.3f}", va="center")
plt.tight_layout()
plt.savefig("ood_average_accuracy.png", dpi=200)
plt.close()

print("Execution complete.")