# MedQA — Medical AI on AMD ROCm

Fine-tuned LLM for clinical question answering.
Built for the AMD Hackathon on lablab.ai.
Runs entirely on AMD hardware. No CUDA required.


## What This Does

MedQA takes a multiple-choice clinical question with 4 options and returns:
- The correct answer letter (A/B/C/D)
- A clinical explanation of the reasoning

Fine-tuned on 193,000 medical MCQs from MedMCQA using LoRA.
Only 0.15% of parameters were trained — fast and memory efficient.


## Stack

Base Model    : Qwen/Qwen2-1.5B
Fine-tuning   : PEFT LoRA (r=4, q/v projection)
Dataset       : openlifescienceai/medmcqa
Hardware      : AMD Instinct MI300X (192GB HBM3)
Framework     : PyTorch + HuggingFace Transformers + TRL
Precision     : bfloat16 (ROCm native)
Web UI        : Gradio


## Project Structure

medqa-finetune/
  train.py        LoRA fine-tuning pipeline
  infer.py        CLI inference on sample questions
  eval.py         Accuracy evaluation with subject breakdown
  app.py          Gradio web app
  requirements.txt
  outputs/        Saved LoRA adapter and tokenizer


## Quickstart

1. Clone and setup

git clone https://github.com/YOUR_USERNAME/medqa-rocm
cd medqa-rocm
python3 -m venv venv && source venv/bin/activate

2. Install dependencies

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
pip install transformers datasets peft accelerate trl gradio

3. Train

python3 train.py
Approx 5 minutes on AMD MI300X

4. Evaluate

python3 eval.py
Runs on 100 MedMCQA validation samples with per-subject breakdown

5. Launch app

python3 app.py
Local  : http://0.0.0.0:7860
Public : shown in terminal as gradio.live link


## AMD ROCm Developer Notes

What worked great:
- AMD MI300X with 192GB HBM3 — zero memory pressure
- bfloat16 is native and stable on ROCm — more stable than fp16
- device_map="auto" works seamlessly
- ROCm 6.1 PyTorch wheel installed without issues

Gotchas and fixes:

Issue                                  Fix
grad_norm nan during training          Switch fp16 to bfloat16
evaluation_strategy keyword error      Renamed to eval_strategy
tokenizer= in Trainer error            Renamed to processing_class=
torch_dtype= deprecated warning        Use dtype= instead
GPU not detected                       Set HSA_OVERRIDE_GFX_VERSION=9.4.2
Exclamation marks in inference output  Load tokenizer from BASE_MODEL not adapter path, use merge_and_unload()
bitsandbytes not supported on ROCm     Use bfloat16 instead of int4 or int8


## Results

Training time     : ~5 mins on MI300X
Trainable params  : ~2.2M of 1.5B (0.15%)
Training samples  : 500
Eval accuracy     : run eval.py to fill this in
Random baseline   : 25% (4-choice MCQ)


## Roadmap

- Scale to 10k+ training samples
- Push adapter to HuggingFace Hub
- Add confidence score per answer
- Subject-specific fine-tuning (cardiology, pharmacology, etc.)
- Deploy on HuggingFace Spaces


## Built in Public

X        : tag @AIatAMD and @lablabai
LinkedIn : tag AMD Developer and lablab.ai


## License

MIT — free to use, modify, and build on.

Built at the AMD Hackathon 2025 on lablab.ai
