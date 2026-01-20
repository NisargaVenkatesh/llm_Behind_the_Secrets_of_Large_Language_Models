# scripts/30_train_mcq_lora.py
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import sys
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

from _utils import parse_choices

OPTION_SPLIT_RE = re.compile(r"\n\s*(A[\.\)\:]|\(?A\)?[\.\)\:])\s+", re.IGNORECASE)

def clean_question_from_prompt(prompt_field: str) -> str:
    """Keep the actual question text; remove duplicated option blocks if present."""
    s = str(prompt_field) if prompt_field is not None else ""
    s = s.strip()

    m = re.search(r"(?is)\bquestion\s*:\s*(.*)", s)
    if m:
        s = m.group(1).strip()

    m2 = OPTION_SPLIT_RE.search(s)
    if m2:
        s = s[:m2.start()].strip()

    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text(question_prompt: str, choices: dict, label: str, country: str | None = None) -> str:
    """One supervised training string. Model learns to output ONE letter."""
    q = clean_question_from_prompt(question_prompt)
    country_line = f"Country context: {country}\n" if country else ""
    text = (
        "Choose the correct option (A, B, C, or D). Answer with ONLY the letter.\n"
        + country_line
        + "\n"
        f"Question: {q}\n"
        "Options:\n"
        f"A) {choices['A']}\n"
        f"B) {choices['B']}\n"
        f"C) {choices['C']}\n"
        f"D) {choices['D']}\n\n"
        "Answer:"
    )
    return text + " " + str(label).strip()

def truncate_to_max_tokens(tok: AutoTokenizer, s: str, max_len: int) -> str:
    """Hard cap length by tokenizing and decoding back."""
    ids = tok(s, add_special_tokens=True, truncation=True, max_length=max_len).input_ids
    return tok.decode(ids, skip_special_tokens=True)

def _precision_flags():
    # On H100 you want bf16. On older GPUs, fp16 is the fallback.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    if torch.cuda.is_available():
        return False, True
    return False, False

def _resolve_csv(data_dir: Path, maybe_name: str | None, default_name: str) -> Path:
    """
    If user passes a CSV, treat it as:
      - absolute path, or
      - path relative to data_dir, or
      - just a filename inside data_dir
    """
    if maybe_name is None:
        return data_dir / default_name
    p = Path(maybe_name)
    if p.is_absolute():
        return p
    # allow user to pass "data/foo.csv" too
    if p.exists():
        return p
    return data_dir / maybe_name

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="lora_runs/mcq")
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)

    # NEW: explicit split files (recommended)
    ap.add_argument("--train_csv", type=str, default=None,
                    help="MCQ train split CSV (relative to --data_dir unless absolute).")
    ap.add_argument("--val_csv", type=str, default=None,
                    help="MCQ val split CSV (relative to --data_dir unless absolute).")

    args = ap.parse_args()

    use_bf16, use_fp16 = _precision_flags()
    print(f"[INFO] cuda={torch.cuda.is_available()} bf16={use_bf16} fp16={use_fp16}")
    if not torch.cuda.is_available():
        print(
            "[WARN] No GPU detected. Training LoRA on CPU will be extremely slow and may fail.\n"
            "Run via sbatch on a GPU node.",
            file=sys.stderr,
        )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_out = out_dir / "mcq_lora"
    ckpt_dir = out_dir / "mcq_lora_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Choose data source:
    # - If train_csv & val_csv given -> use them (no leakage, reproducible)
    # - Else -> fallback to train_dataset_mcq.csv with an internal 90/10 split
    if (args.train_csv is None) ^ (args.val_csv is None):
        raise SystemExit("[ERROR] Provide BOTH --train_csv and --val_csv, or provide neither.")

    if args.train_csv and args.val_csv:
        train_path = _resolve_csv(data_dir, args.train_csv, "train_dataset_mcq.csv")
        val_path = _resolve_csv(data_dir, args.val_csv, "train_dataset_mcq.csv")
        print(f"[INFO] Using explicit splits:\n  train={train_path}\n  val  ={val_path}")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    else:
        df = pd.read_csv(data_dir / "train_dataset_mcq.csv")
        strat = df["country"] if ("country" in df.columns and df["country"].nunique() > 1) else None
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=strat)
        print("[INFO] Using internal random split: 90/10 from train_dataset_mcq.csv")

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    # For training, right padding is okay. (Generation scripts should set padding_side='left'.)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build dataset with ONLY "text" column (TRL stays in "language modeling" mode).
    train_items = []
    for _, r in train_df.iterrows():
        choices = parse_choices(r["choices"])
        country = str(r["country"]).strip() if ("country" in train_df.columns and pd.notna(r.get("country"))) else None
        s = build_text(r["prompt"], choices, r["answer_idx"], country=country)
        s = truncate_to_max_tokens(tok, s, args.max_seq_len)
        train_items.append({"text": s})

    val_items = []
    for _, r in val_df.iterrows():
        choices = parse_choices(r["choices"])
        country = str(r["country"]).strip() if ("country" in val_df.columns and pd.notna(r.get("country"))) else None
        s = build_text(r["prompt"], choices, r["answer_idx"], country=country)
        s = truncate_to_max_tokens(tok, s, args.max_seq_len)
        val_items.append({"text": s})

    print(f"[INFO] train rows={len(train_items)} val rows={len(val_items)}")

    train_ds = Dataset.from_list(train_items)
    val_ds = Dataset.from_list(val_items)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        quantization_config=bnb_cfg,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,   # IMPORTANT on HPC (prevents common hangs)
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        processing_class=tok,  # TRL 0.26.2 uses this instead of tokenizer=
    )

    trainer.train()
    trainer.model.save_pretrained(str(adapter_out))
    tok.save_pretrained(str(adapter_out))
    print(f"[OK] Saved MCQ LoRA adapter to {adapter_out}")

if __name__ == "__main__":
    main()
