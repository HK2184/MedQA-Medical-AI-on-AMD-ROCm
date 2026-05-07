import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

BASE_MODEL   = "Qwen/Qwen2-1.5B"
ADAPTER_PATH = "./outputs"
NUM_EVAL     = 100
LABEL_MAP    = {0: "A", 1: "B", 2: "C", 3: "D"}

def build_prompt(ex):
    return (
        f"### Question:\n{ex['question']}\n\n"
        f"### Options:\n"
        f"A) {ex['opa']}\nB) {ex['opb']}\n"
        f"C) {ex['opc']}\nD) {ex['opd']}\n\n"
        f"### Answer:\n"
    )

def predict(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new  = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new, skip_special_tokens=True).strip().upper()
    for ch in text:
        if ch in ["A", "B", "C", "D"]:
            return ch
    return "A"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading model + adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()

    print("Loading validation set...")
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.filter(lambda x: x["cop"] is not None and x["question"] is not None)
    ds = ds.select(range(min(NUM_EVAL, len(ds))))

    correct       = 0
    subject_scores = {}

    print(f"\nEvaluating {len(ds)} samples...\n")
    for i, ex in enumerate(ds):
        pred  = predict(build_prompt(ex), model, tokenizer)
        truth = LABEL_MAP.get(ex["cop"], "A")
        hit   = pred == truth
        if hit:
            correct += 1

        subj = ex.get("subject_name", "General")
        if subj not in subject_scores:
            subject_scores[subj] = [0, 0]
        subject_scores[subj][1] += 1
        if hit:
            subject_scores[subj][0] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{len(ds)}]  Accuracy so far: {correct/(i+1)*100:.1f}%")

    acc = correct / len(ds) * 100
    print(f"\n{'='*45}")
    print(f"  FINAL ACCURACY  : {correct}/{len(ds)} = {acc:.1f}%")
    print(f"  RANDOM BASELINE : 25.0%")
    print(f"  GAIN OVER RANDOM: +{acc-25:.1f} pp")
    print(f"{'='*45}")
    print("\nPer-subject breakdown:")
    for subj, (c, t) in sorted(subject_scores.items(), key=lambda x: -x[1][0]/max(x[1][1],1)):
        print(f"  {subj:<35} {c}/{t} ({c/t*100:.0f}%)")

if __name__ == "__main__":
    main()
