import argparse
from pathlib import Path
import math

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# robust import
try:
    from scripts._utils import normalize_text
except Exception:
    from _utils import normalize_text


def build_prompt(en_question: str, country: str | None = None) -> str:
    country_line = f"Country context: {country}\n" if country else ""
    return (
        "Answer the question with a short phrase in English. Do not explain.\n"
        + country_line
        + "\n"
        + f"Question: {str(en_question).strip()}\n\n"
        + "Answer:"
    )


@torch.no_grad()
def score_completion_logprob(model, tok, prompt: str, completion: str, device: torch.device) -> float:
    """
    Compute log P(completion | prompt) by summing token logprobs of completion tokens.
    """
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

    comp_text = " " + completion.strip()
    comp_ids = tok(comp_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  

 
    start = prompt_ids.shape[1]

    pred_positions = torch.arange(start - 1, start - 1 + comp_ids.shape[1], device=device)
    target_tokens = comp_ids.squeeze(0)

    logprobs = torch.log_softmax(logits[0, pred_positions, :], dim=-1)  
    token_logprobs = logprobs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)  
    return float(token_logprobs.sum().item())


@torch.no_grad()
def generate_candidates(model, tok, prompt: str, device: torch.device, n: int, max_new_tokens: int,
                        temperature: float, top_p: float) -> list[str]:
    """
    Sample N short answers from the model.
    We then normalize/deduplicate and rerank.
    """
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    
    tok.padding_side = "left"

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )


    prompt_len = inputs["input_ids"].shape[1]
    cands = []
    for seq in out:
        gen_ids = seq[prompt_len:]
        txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
        txt = txt.split("\n")[0].strip()
        if txt:
            cands.append(txt)
    return cands


def dedup_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, required=True, help="CSV file name inside data_dir (e.g., test_dataset_saq.csv or val_saq_split.csv)")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")

    ap.add_argument("--num_samples", type=int, default=12, help="How many sampled candidates to generate per question")
    ap.add_argument("--max_new_tokens", type=int, default=8, help="Max tokens for the short answer")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--max_candidates", type=int, default=8, help="After dedup, keep top-K candidates for reranking")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "saq_prediction.tsv"

    df = pd.read_csv(data_dir / args.split)
    if "en_question" not in df.columns and "question" in df.columns:
        df["en_question"] = df["question"]

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
        q = str(r.get("en_question", "")).strip()
        cid = str(r.get("country", "")).strip() if pd.notna(r.get("country", "")) else ""
        prompt = build_prompt(q, country=cid if cid else None)

       
        raw_cands = generate_candidates(
            model, tok, prompt, device=device,
            n=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

       
        norm_cands = [normalize_text(x) for x in raw_cands]
        norm_cands = [x for x in norm_cands if x]  
        norm_cands = dedup_keep_order(norm_cands)

       
        if not norm_cands:
            raw = generate_candidates(
                model, tok, prompt, device=device,
                n=1, max_new_tokens=args.max_new_tokens,
                temperature=0.0, top_p=1.0
            )
            fallback = normalize_text(raw[0] if raw else "")
            norm_cands = [fallback] if fallback else [""]

       
        cand_pool = norm_cands[: args.max_candidates]

        
        best = cand_pool[0]
        best_lp = -math.inf
        for c in cand_pool:
            if not c:
                continue
            lp = score_completion_logprob(model, tok, prompt, c, device=device)
            if lp > best_lp:
                best_lp = lp
                best = c

        rows.append({"ID": r["ID"], "answer": best})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
