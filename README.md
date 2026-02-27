# Behind the Secrets of Large Language Models — MCQ/SAQ (Llama-3-8B + LoRA / QLoRA)

This repository contains a lightweight, reproducible pipeline to fine-tune **Meta-Llama-3-8B** with **LoRA adapters** (trained on a **4-bit quantized** base model, often called **QLoRA**) for two task types:

- **MCQ**: predict a single option letter **A/B/C/D**
- **SAQ**: generate a short English phrase answer and optionally **self-re-rank** candidates by log-probability

The key idea is to keep the base model frozen (cheap + stable) and train only small adapter weights (LoRA). At inference time, we score answers with model **log-probabilities** (how likely the model thinks a continuation is), which often improves reliability compared to taking the first sampled output.

---

### Training
- `30_train_mcq_lora.py` — LoRA fine-tuning for MCQ.  
  Builds prompts like “Answer with ONLY the letter” and trains the model to output a single letter.
- `31_train_saq_lora.py` — LoRA fine-tuning for SAQ.  
  Uses a canonical target answer (e.g., most frequent English annotation) and trains the model to output a short phrase.

### Inference
- `51_predict_mcq_logits_lora.py` — MCQ prediction by **log-prob scoring** each candidate continuation (`" A"`, `" B"`, `" C"`, `" D"`) and picking the best.
- `50_predict_saq_selfrank.py` — SAQ generation with **self-ranking**:
  1) sample multiple candidate short answers  
  2) score each candidate by completion log-probability  
  3) optionally normalize answers for grader friendliness  
  4) output the best candidate

### Utilities
- `60_convert_mcq_predletter_to_abcd_tsv.py` — convert `(MCQID, pred_letter)` into Codabench-style boolean columns `A,B,C,D`.
- `_utils.py` — parsing + normalization helpers (choices parsing, SAQ annotation parsing, answer cleanup/normalization).
- `13_score_val.py` — local validation scoring for MCQ + SAQ (useful for quick iteration).
- `environment.yml` — conda environment (PyTorch + HF + TRL + PEFT + bitsandbytes).

---

## Setup

### 1) Create environment
```bash
conda env create -f environment.yml
conda activate llmcourse
