![AMD](https://img.shields.io/badge/AMD-MI300X-red)
![ROCm](https://img.shields.io/badge/ROCm-6.1-blue)
![LoRA](https://img.shields.io/badge/LoRA-Finetuned-green)
![Status](https://img.shields.io/badge/status-working-success)
# 🧠 MedQA — Medical AI on AMD ROCm

Clinical question-answering LLM fine-tuned on MedMCQA using LoRA.  
Runs entirely on **AMD Instinct MI300X (ROCm)** — no CUDA required.

---

# 🚀 What it does

MedQA takes a multiple-choice medical question and returns:

- ✅ Correct answer (A/B/C/D)
- 🧾 Clinical reasoning / explanation

---

# 🧠 Example

## Question

> First-line treatment for hypertensive emergency?

## Model Output

```text
Answer: B) IV labetalol or IV nitroprusside

Explanation:
Intravenous beta-blockers rapidly reduce blood pressure quickly and safely in emergency settings.
```

---

# ⚙️ Tech Stack

| Component | Details |
|---|---|
| Base Model | Qwen3-1.7B |
| Fine-tuning | LoRA (PEFT) |
| Dataset | MedMCQA |
| Hardware | AMD MI300X (192GB) |
| Framework | PyTorch + Transformers |
| Precision | bfloat16 (ROCm native) |
| UI | Gradio |

---

# ⚡ AMD Developer Cloud Setup

## 1. Create GPU Droplet

Go to AMD Developer Cloud.

Click **Create Instance / GPU Droplet**

Select:

- GPU: **MI300X**
- Image: **ROCm 6.x (recommended)**
- Region: any available

Click **Launch**

---

## 2. Open Web Console

Click your instance.

Click **Web Console**

You will get a terminal like:

```bash
root@rocm-...:~#
```

---

## 3. Update system (recommended)

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 4. Clone project

```bash
git clone https://github.com/HK2184/MedQA-Medical-AI-on-AMD-ROCm.git

cd MedQA-Medical-AI-on-AMD-ROCm
```

---

## 5. Create environment

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 6. Install dependencies (ROCm)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

pip install transformers datasets peft accelerate trl gradio
```

---

# 🧠 Training

```bash
python train.py
```

- Runs LoRA fine-tuning
- Takes ~5 minutes on MI300X
- Saves adapter in `outputs/`

---

# 🔍 Inference

```bash
python infer.py
```

Runs sample clinical questions.

Outputs answer + explanation.

---

# 🌐 Web App

```bash
python app.py
```

Then open:

```text
http://0.0.0.0:7860
```

Gradio may also provide a public link.

---

# 📁 Project Structure

```text
medqa-finetune/
│
├── train.py        # LoRA fine-tuning
├── infer.py        # CLI inference
├── eval.py         # Evaluation script
├── app.py          # Gradio UI
├── README.md
```

---

# ⚡ Why AMD ROCm?

- MI300X (192GB HBM3) → zero memory issues
- Native bfloat16 → stable training
- No CUDA dependency → fully open stack
- Works seamlessly with HuggingFace

---

# 🛠️ Challenges & Fixes

| Issue | Fix |
|---|---|
| NaN loss | Switched to bfloat16 |
| Trainer errors | Adjusted for transformers version |
| GPU not detected | Set ROCm env variables |
| Inference garbage output | Fixed tokenizer + decoding |
| bitsandbytes unsupported | Used bf16 instead |

---

# 📊 Results

- Trainable params: ~2.2M / 1.5B (0.15%)
- Training time: ~5 minutes
- Dataset: MedMCQA
- Baseline accuracy: 25%

---

## 🖥️ Demo

![App Screenshot]  <img width="1000" height="952" alt="Screenshot From 2026-05-07 14-26-07" src="https://github.com/user-attachments/assets/776df667-472b-445a-b73f-a06ccc47e3c9" />


# 🚀 Roadmap

- Scale to larger dataset
- Push model to HuggingFace Hub
- Add confidence scoring
- Deploy on HuggingFace Spaces

---

# 📜 License

MIT License

---

# 🙌 Acknowledgements

Built for the AMD Hackathon on lablab.ai

Powered by AMD ROCm + HuggingFace ecosystem
