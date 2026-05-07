import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

BASE_MODEL     = "Qwen/Qwen2-1.5B"
ADAPTER_PATH   = "./outputs"
MAX_NEW_TOKENS = 200

def build_prompt(question, opa, opb, opc, opd):
    return (
        f"### Question:\n{question}\n\n"
        f"### Options:\n"
        f"A) {opa}\n"
        f"B) {opb}\n"
        f"C) {opc}\n"
        f"D) {opd}\n\n"
        f"### Answer:\n"
    )

def generate(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    print("="*50)
    print(f"PyTorch version : {torch.__version__}")
    print(f"ROCm available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print("="*50)

    if not os.path.exists(ADAPTER_PATH):
        print(f"\n✗ Adapter not found at {ADAPTER_PATH}")
        print("  Run train.py first!")
        return

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (fp16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("✓ Model ready!\n")

    samples = [
        {
            "question": (
                "A 35-year-old male presents with chest pain radiating to "
                "the left arm. ECG shows ST elevation in leads II, III, and "
                "aVF. Which artery is most likely occluded?"
            ),
            "opa": "Left anterior descending artery",
            "opb": "Right coronary artery",
            "opc": "Left circumflex artery",
            "opd": "Left main coronary artery",
        },
        {
            "question": (
                "Which of the following is the first-line treatment "
                "for hypertensive emergency?"
            ),
            "opa": "Oral amlodipine",
            "opb": "IV labetalol or IV nitroprusside",
            "opc": "Sublingual nifedipine",
            "opd": "IM hydralazine",
        },
        {
            "question": (
                "A patient presents with polyuria, polydipsia, and weight "
                "loss. Fasting blood glucose is 280 mg/dL. What is the "
                "most likely diagnosis?"
            ),
            "opa": "Diabetes insipidus",
            "opb": "Type 2 Diabetes Mellitus",
            "opc": "Type 1 Diabetes Mellitus",
            "opd": "Cushing syndrome",
        },
    ]

    for i, s in enumerate(samples, 1):
        prompt = build_prompt(
            s["question"], s["opa"], s["opb"], s["opc"], s["opd"]
        )

        print(f"{'='*60}")
        print(f"  QUESTION {i}")
        print(f"{'='*60}")
        print(f"Q: {s['question']}")
        print(f"   A) {s['opa']}")
        print(f"   B) {s['opb']}")
        print(f"   C) {s['opc']}")
        print(f"   D) {s['opd']}")

        print(f"\n--- Model Output ---")
        answer = generate(prompt, model, tokenizer)
        print(answer)
        print()

if __name__ == "__main__":
    main()
