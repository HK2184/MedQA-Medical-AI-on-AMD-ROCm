import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"

BASE_MODEL   = "Qwen/Qwen2-1.5B"
ADAPTER_PATH = "./outputs"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()
print("Ready!")

EXAMPLES = [
    ["Which artery is occluded in inferior MI with ST elevation in II, III, aVF?",
     "Left anterior descending artery", "Right coronary artery",
     "Left circumflex artery", "Left main coronary artery"],
    ["First-line treatment for hypertensive emergency?",
     "Oral amlodipine", "IV labetalol or IV nitroprusside",
     "Sublingual nifedipine", "IM hydralazine"],
    ["Most common cause of community-acquired pneumonia?",
     "Klebsiella pneumoniae", "Streptococcus pneumoniae",
     "Haemophilus influenzae", "Mycoplasma pneumoniae"],
    ["Drug of choice for absence seizures?",
     "Phenytoin", "Carbamazepine",
     "Ethosuximide", "Valproate"],
]

def answer(question, opa, opb, opc, opd):
    if not question.strip():
        return "Please enter a question."
    if not all([opa.strip(), opb.strip(), opc.strip(), opd.strip()]):
        return "Please fill in all four options."
    prompt = (
        f"### Question:\n{question}\n\n"
        f"### Options:\n"
        f"A) {opa}\nB) {opb}\nC) {opc}\nD) {opd}\n\n"
        f"### Answer:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new, skip_special_tokens=True)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #080d1a;
    --surface:  #0f1624;
    --surface2: #162030;
    --border:   #1a3356;
    --accent:   #00c8f0;
    --accent2:  #0055ff;
    --green:    #00f0a0;
    --text:     #deeeff;
    --muted:    #4a6080;
    --danger:   #ff3366;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 20px 60px !important;
}

/* Header */
#header {
    padding: 44px 0 28px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
    position: relative;
}
#header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--accent2), var(--accent), var(--green));
}
.badges { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
.badge {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 9px; border-radius: 4px; border: 1px solid;
}
.b-amd  { color: #ff6030; border-color: #ff603030; background: #ff603010; }
.b-rocm { color: var(--accent); border-color: #00c8f030; background: #00c8f008; }
.b-lora { color: var(--green);  border-color: #00f0a030; background: #00f0a008; }
.b-live { color: #ffcc00; border-color: #ffcc0030; background: #ffcc0008; }

h1#title {
    font-family: 'Syne', sans-serif !important;
    font-size: 42px !important; font-weight: 800 !important;
    letter-spacing: -0.03em !important; line-height: 1 !important;
    color: var(--text) !important; margin-bottom: 10px !important;
}
h1#title em { color: var(--accent); font-style: normal; }
.subtitle { font-size: 14px; color: var(--muted); font-weight: 300; line-height: 1.6; max-width: 520px; }

/* Stats */
#stats {
    display: flex; border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
    background: var(--surface); margin-bottom: 28px;
}
.stat { flex: 1; padding: 14px 16px; text-align: center; border-right: 1px solid var(--border); }
.stat:last-child { border-right: none; }
.sv { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: var(--accent); display: block; }
.sl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--green); margin-right: 4px; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Inputs */
label span, .label-wrap span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    color: var(--muted) !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
textarea, input[type=text] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important; line-height: 1.6 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px #00c8f012 !important;
    outline: none !important;
}

/* Section labels */
.section-label {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px;
    display: flex; align-items: center; gap: 7px;
}
.section-label::before {
    content: ''; width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent); display: inline-block;
}

/* Button */
button.lg.primary {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-family: 'Syne', sans-serif !important;
    font-size: 14px !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; padding: 14px !important;
    width: 100% !important; margin-top: 14px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
button.lg.primary:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* Output */
.out-box textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 14px !important; line-height: 1.8 !important;
    color: var(--text) !important; min-height: 280px !important;
}

/* Examples */
.examples-holder table {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; overflow: hidden !important;
}
.examples-holder td, .examples-holder th {
    background: transparent !important; color: var(--text) !important;
    font-size: 13px !important; border-color: var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.examples-holder tr:hover td { background: var(--surface2) !important; cursor: pointer; }

/* Footer */
#footer {
    margin-top: 44px; padding-top: 22px;
    border-top: 1px solid var(--border);
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 10px;
}
.fl { font-size: 12px; color: var(--muted); }
.fl strong { color: var(--text); }
.fr { display: flex; gap: 14px; }
.flink { font-size: 12px; color: var(--accent); text-decoration: none; }
"""

with gr.Blocks(css=CSS, title="MedQA — AMD ROCm") as demo:

    gr.HTML("""
    <div id="header">
        <div class="badges">
            <span class="badge b-amd">AMD MI300X</span>
            <span class="badge b-rocm">ROCm 6.1</span>
            <span class="badge b-lora">LoRA Fine-tuned</span>
            <span class="badge b-live"><span class="dot"></span>Live Inference</span>
        </div>
        <h1 id="title">Med<em>QA</em> Assistant</h1>
        <p class="subtitle">
            Clinical question-answering AI fine-tuned on MedMCQA.
            Running on AMD Instinct MI300X via ROCm — no CUDA required.
        </p>
    </div>
    <div id="stats">
        <div class="stat"><span class="sv">1.5B</span><span class="sl">Parameters</span></div>
        <div class="stat"><span class="sv">LoRA</span><span class="sl">Fine-tuning</span></div>
        <div class="stat"><span class="sv">193k</span><span class="sl">Training QA</span></div>
        <div class="stat"><span class="sv">MI300X</span><span class="sl">AMD GPU</span></div>
        <div class="stat"><span class="sv">bf16</span><span class="sl">Precision</span></div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Clinical Question</div>')
            question = gr.Textbox(
                label="",
                placeholder="e.g. A 45-year-old presents with sudden onset severe headache...",
                lines=4,
            )
            gr.HTML('<div class="section-label" style="margin-top:14px">Answer Options</div>')
            with gr.Row():
                opa = gr.Textbox(label="Option A", placeholder="First option")
                opb = gr.Textbox(label="Option B", placeholder="Second option")
            with gr.Row():
                opc = gr.Textbox(label="Option C", placeholder="Third option")
                opd = gr.Textbox(label="Option D", placeholder="Fourth option")
            btn = gr.Button("Analyze Question", variant="primary")

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">AI Answer & Reasoning</div>')
            output = gr.Textbox(
                label="",
                placeholder="Answer and clinical explanation will appear here...",
                lines=14,
                elem_classes=["out-box"],
            )

    gr.HTML('<div class="section-label" style="margin-top:24px">Sample Questions — click any to load</div>')
    gr.Examples(
        examples=EXAMPLES,
        inputs=[question, opa, opb, opc, opd],
        label="",
            )

    gr.HTML("""
    <div id="footer">
        <div class="fl">
            Built on <strong>AMD Developer Cloud</strong> &nbsp;·&nbsp;
            Model: <strong>Qwen2-1.5B + LoRA</strong> &nbsp;·&nbsp;
            Dataset: <strong>MedMCQA</strong>
        </div>
        <div class="fr">
            <a class="flink" href="https://github.com" target="_blank">GitHub →</a>
            <a class="flink" href="https://lablab.ai" target="_blank">lablab.ai →</a>
            <a class="flink" href="https://cloud.amd.com" target="_blank">AMD Cloud →</a>
        </div>
    </div>
    """)

    btn.click(fn=answer, inputs=[question, opa, opb, opc, opd], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
