import argparse
from pathlib import Path
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    from scripts._utils import parse_choices
except Exception:
    from _utils import parse_choices


def build_prompt(question_prompt: str, choices: dict, country: str | None = None) -> str:
    country_line = f"Country context: {country}\n" if country else ""
    q = str(question_prompt).strip()
    return (
        "Choose the correct option (A, B, C, or D). Answer with ONLY the letter.\n"
        + country_line
        + "\n"
        + f"Question: {q}\n"
        + "Options:\n"
        + f"A) {choices['A']}\n"
        + f"B) {choices['B']}\n"
        + f"C) {choices['C']}\n"
        + f"D) {choices['D']}\n\n"
        + "Answer:"
    )


@torch.no_grad()
def pick_best_letter_by_logprob(model, tok, prompt: str, device: torch.device) -> tuple[str, dict]:
    """
    Score ' A', ' B', ' C', ' D' as single-token (or few-token) continuations and pick best.
    """
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

    scores = {}
    best_letter = "A"
    best_lp = -1e30

    for L in ["A", "B", "C", "D"]:
        comp = " " + L
        comp_ids = tok(comp, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        attn = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits

        start = prompt_ids.shape[1]
        pred_positions = torch.arange(start - 1, start - 1 + comp_ids.shape[1], device=device)
        targets = comp_ids.squeeze(0)

        logprobs = torch.log_softmax(logits[0, pred_positions, :], dim=-1)
        lp = float(logprobs.gather(1, targets.unsqueeze(1)).sum().item())
        scores[L] = lp
        if lp > best_lp:
            best_lp = lp
            best_letter = L

    return best_letter, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, required=True, help="CSV name in data_dir (e.g., test_dataset_mcq.csv or val_mcq_split.csv)")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "mcq_prediction.tsv"

    df = pd.read_csv(data_dir / args.split)

    # expected: MCQID (or ID), prompt, choices, country
    if "MCQID" not in df.columns:
        if "ID" in df.columns:
            df["MCQID"] = df["ID"]
        else:
            raise ValueError("Need MCQID or ID column in the MCQ split file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto" if device.type == "cuda" else None,
        quantization_config=bnb_cfg if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    rows = []
    for _, r in df.iterrows():
        choices = parse_choices(r["choices"])
        country = str(r.get("country", "")).strip() if pd.notna(r.get("country", "")) else ""
        prompt = build_prompt(r["prompt"], choices, country=country if country else None)
        letter, _scores = pick_best_letter_by_logprob(model, tok, prompt, device=device)
        rows.append({"MCQID": r["MCQID"], "pred_letter": letter})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
