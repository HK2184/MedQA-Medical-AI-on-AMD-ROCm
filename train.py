import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

MODEL_ID     = "Qwen/Qwen3-1.7B"
OUTPUT_DIR   = "./outputs"
MAX_LENGTH   = 256
NUM_SAMPLES  = 2000
NUM_EPOCHS   = 2
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
LR           = 2e-4

def format_prompt(example):
    options = (
        f"A) {example['opa']}\n"
        f"B) {example['opb']}\n"
        f"C) {example['opc']}\n"
        f"D) {example['opd']}"
    )

    label_map  = {0: "A", 1: "B", 2: "C", 3: "D"}
    answer_letter = label_map.get(example["cop"], "A")
    answer_text   = [
        example["opa"], example["opb"],
        example["opc"], example["opd"]
    ][example["cop"]]

    explanation = example.get("exp") or "No explanation provided."

    prompt = (
        f"### Question:\n{example['question']}\n\n"
        f"### Options:\n{options}\n\n"
        f"### Answer:\n{answer_letter}) {answer_text}\n\n"
        f"### Explanation:\n{explanation}"
    )

    return {"text": prompt}

def tokenize(example, tokenizer):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    print("="*50)
    print(f"PyTorch version : {torch.__version__}")
    print(f"ROCm available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print("="*50)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading MedMCQA dataset...")
    raw = load_dataset("openlifescienceai/medmcqa", split="train")

    raw = raw.filter(
        lambda x: x["cop"] is not None and x["question"] is not None
    )

    raw = raw.select(range(min(NUM_SAMPLES, len(raw))))

    print("Formatting prompts...")
    formatted = raw.map(format_prompt, remove_columns=raw.column_names)

    print("Tokenizing...")
    tokenized = formatted.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=["text"],
        batched=False,
    )

    tokenized.set_format("torch")

    split    = tokenized.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,
        bf16=False,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=2,
        save_total_limit=1,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving LoRA adapter + tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Done! Model saved to {OUTPUT_DIR}")

    print("Files saved:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
