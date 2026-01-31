import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

from _utils import parse_saq_annotations, normalize_text

def pick_canonical_answer(ann_str: str) -> str:
    """Pick most frequent English answer if possible, else first available."""
    anns = parse_saq_annotations(ann_str)
    best = None
    best_count = -1
    for a in anns:
        en = a.get("en_answers", [])
        cnt = int(a.get("count", 1))
        if en and cnt > best_count:
            best = en[0]
            best_count = cnt
    if best is None:
        for a in anns:
            en = a.get("en_answers", [])
            if en:
                best = en[0]
                break
    return best or ""

def build_text(en_question: str, answer: str, country: str | None = None) -> str:
    a = normalize_text(answer)
    country_line = f"Country context: {country}\n" if country else ""
    text = (
        "Answer the question with a short phrase in English. Do not explain.\n"
        + country_line
        + "\n"
        f"Question: {str(en_question).strip()}\n\n"
        "Answer:"
    )
    return text + " " + a

def truncate_to_max_tokens(tok: AutoTokenizer, s: str, max_len: int) -> str:
    ids = tok(s, add_special_tokens=True, truncation=True, max_length=max_len).input_ids
    return tok.decode(ids, skip_special_tokens=True)

def _precision_flags():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    if torch.cuda.is_available():
        return False, True
    return False, False

def _resolve_csv(data_dir: Path, maybe_name: str | None, default_name: str) -> Path:
    if maybe_name is None:
        return data_dir / default_name
    p = Path(maybe_name)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return data_dir / maybe_name

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="lora_runs/saq")
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)

    ap.add_argument("--train_csv", type=str, default=None,
                    help="SAQ train split CSV (relative to --data_dir unless absolute).")
    ap.add_argument("--val_csv", type=str, default=None,
                    help="SAQ val split CSV (relative to --data_dir unless absolute).")

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

    adapter_out = out_dir / "saq_lora"
    ckpt_dir = out_dir / "saq_lora_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if (args.train_csv is None) ^ (args.val_csv is None):
        raise SystemExit("[ERROR] Provide BOTH --train_csv and --val_csv, or provide neither.")

    if args.train_csv and args.val_csv:
        train_path = _resolve_csv(data_dir, args.train_csv, "train_dataset_saq.csv")
        val_path = _resolve_csv(data_dir, args.val_csv, "train_dataset_saq.csv")
        print(f"[INFO] Using explicit splits:\n  train={train_path}\n  val  ={val_path}")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    else:
        df = pd.read_csv(data_dir / "train_dataset_saq.csv")
        strat = df["country"] if ("country" in df.columns and df["country"].nunique() > 1) else None
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=strat)
        print("[INFO] Using internal random split: 90/10 from train_dataset_saq.csv")

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_items = []
    skipped = 0
    for _, r in train_df.iterrows():
        q = r.get("en_question") or r.get("question") or ""
        ans = pick_canonical_answer(r["annotations"])
        if not str(ans).strip():
            skipped += 1
            continue
        country = str(r.get("country", "")).strip() if pd.notna(r.get("country")) else None
        s = build_text(q, ans, country=country)
        s = truncate_to_max_tokens(tok, s, args.max_seq_len)
        train_items.append({"text": s})

    val_items = []
    for _, r in val_df.iterrows():
        q = r.get("en_question") or r.get("question") or ""
        ans = pick_canonical_answer(r["annotations"])
        if not str(ans).strip():
            continue
        country = str(r.get("country", "")).strip() if pd.notna(r.get("country")) else None
        s = build_text(q, ans, country=country)
        s = truncate_to_max_tokens(tok, s, args.max_seq_len)
        val_items.append({"text": s})

    print(f"[INFO] SAQ skipped empty answers in train: {skipped}")
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
        dataloader_num_workers=0,   
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        processing_class=tok,
    )

    trainer.train()
    trainer.model.save_pretrained(str(adapter_out))
    tok.save_pretrained(str(adapter_out))
    print(f"[OK] Saved SAQ LoRA adapter to {adapter_out}")

if __name__ == "__main__":
    main()
