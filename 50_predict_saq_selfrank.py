import argparse
from pathlib import Path
import math
import re
from collections import Counter

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import joblib
from sklearn.metrics.pairwise import cosine_similarity

# robust import
try:
    from scripts._utils import normalize_answer_for_submission
except Exception:
    from _utils import normalize_answer_for_submission


NUMERIC_ONLY_RE = re.compile(r"(arabic numerals|numbers only|integers?\s*\(|without any decimal)", re.IGNORECASE)
FIRST_INT_RE = re.compile(r"-?\d+")


def load_retriever(path: str):
    return joblib.load(path)


def retrieve_examples(payload, query_text: str, country: str | None, k: int = 4) -> list[dict]:
    query_text = str(query_text).strip()
    if not query_text:
        return []

    if payload.get("use_country_split", False) and country and "per_country" in payload and country in payload["per_country"]:
        block = payload["per_country"][country]
        vec = block["vectorizer"]
        X = block["matrix"]
        meta = block["meta"]
    else:
        vec = payload["vectorizer"]
        X = payload["matrix"]
        meta = payload["meta"]

    qv = vec.transform([query_text])
    sims = cosine_similarity(qv, X).ravel()
    if sims.size == 0:
        return []

    top_idx = sims.argsort()[::-1][:k]
    ex = []
    for i in top_idx:
        ex.append({
            "question": meta["question"][i],
            "answer": meta["answer"][i],
            "ID": meta["ID"][i],
            "country": meta["country"][i],
            "score": float(sims[i]),
        })
    return ex


def format_examples_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["Similar examples (use as hints; keep the final answer short):"]
    for j, e in enumerate(examples, 1):
        lines.append(f"{j}) Q: {e['question']}")
        lines.append(f"   A: {e['answer']}")
    return "\n".join(lines) + "\n\n"


def build_prompt(en_question: str, country: str | None = None, examples_text: str = "") -> str:
    country_line = f"Country context: {country}\n" if country else ""
    return (
        "Answer the question with a short phrase in English. Do not explain.\n"
        + country_line
        + "\n"
        + examples_text
        + f"Question: {str(en_question).strip()}\n\n"
        + "Answer:"
    )


@torch.no_grad()
def score_completion_logprob(model, tok, prompt: str, completion: str, device: torch.device) -> float:
    """
    Compute log P(completion | prompt) by summing token logprobs of completion tokens.
    Higher is better.
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
def generate_candidates(model, tok, prompt: str, device: torch.device,
                        n: int, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    """
    Sample N short answers from the model (or greedy if temperature==0).
    """
    tok.padding_side = "left"
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    do_sample = temperature is not None and temperature > 0.0

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
        ))
    else:
        gen_kwargs.update(dict(num_return_sequences=1))

    out = model.generate(**gen_kwargs)

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


def majority_vote(norm_answers: list[str]) -> tuple[str | None, Counter]:
    """
    Returns (winner_or_none, counts).
    Winner is used only if there's a clear majority signal.
    """
    c = Counter([a for a in norm_answers if a])
    if not c:
        return None, c
    top, top_n = c.most_common(1)[0]
    second_n = c.most_common(2)[1][1] if len(c) >= 2 else 0

    # "clear winner" heuristic: at least 2 votes and strictly beats second place
    if top_n >= 2 and top_n > second_n:
        return top, c
    return None, c


def enforce_numeric_if_needed(question: str, answer: str) -> str:
    """
    If question requests numerals only, extract the first integer from answer.
    """
    if not question:
        return answer
    if not NUMERIC_ONLY_RE.search(question):
        return answer
    m = FIRST_INT_RE.search(answer or "")
    return m.group(0) if m else (answer or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--split", type=str, required=True, help="CSV file name inside data_dir")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")

    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--max_candidates", type=int, default=10)

    # Retrieval options
    ap.add_argument("--retriever_path", type=str, default=None)
    ap.add_argument("--rag_k", type=int, default=4)
    ap.add_argument("--rag_min_sim", type=float, default=0.10,
                    help="Ignore retrieved examples below this cosine similarity.")
    ap.add_argument("--rag_mode", type=str, default="off", choices=["off", "prompt", "candidates"],
                    help="off=baseline, prompt=insert examples into prompt, candidates=use retrieved answers as extra candidates")

    # Decision options
    ap.add_argument("--use_majority_vote", action="store_true",
                    help="Use majority vote among sampled answers when there is a clear winner; otherwise fall back to logprob rerank.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "saq_prediction.tsv"

    df = pd.read_csv(data_dir / args.split)
    if "en_question" not in df.columns and "question" in df.columns:
        df["en_question"] = df["question"]

    has_uid = "uid" in df.columns
    retriever = load_retriever(args.retriever_path) if args.retriever_path else None

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
        country = cid if cid else None

        # Retrieval
        ex = []
        examples_text = ""
        retrieved_answers = []
        if retriever is not None and args.rag_mode != "off":
            ex = retrieve_examples(retriever, q, country=country, k=args.rag_k)
            ex = [e for e in ex if e.get("score", 0.0) >= args.rag_min_sim]
            if args.rag_mode == "prompt":
                examples_text = format_examples_block(ex)
            elif args.rag_mode == "candidates":
                retrieved_answers = [normalize_answer_for_submission(e["answer"]) for e in ex if e.get("answer")]

        prompt = build_prompt(q, country=country, examples_text=examples_text)

        # 1) sample candidates from model
        raw_cands = generate_candidates(
            model, tok, prompt, device=device,
            n=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # normalize generated candidates
        norm_gen = [normalize_answer_for_submission(x) for x in raw_cands]
        norm_gen = [x for x in norm_gen if x]
        norm_gen = dedup_keep_order(norm_gen)

        # greedy fallback (no sampling)
        if not norm_gen:
            raw = generate_candidates(
                model, tok, prompt, device=device,
                n=1, max_new_tokens=args.max_new_tokens,
                temperature=0.0, top_p=1.0
            )
            fallback = normalize_answer_for_submission(raw[0] if raw else "")
            norm_gen = [fallback] if fallback else [""]

        # 2) optional majority vote (vote in normalized space)
        voted = None
        if args.use_majority_vote:
            voted, _counts = majority_vote([normalize_answer_for_submission(x) for x in raw_cands])

        # 3) candidate pool for reranking: generated + retrieved answers 
        pool = norm_gen[:]
        if args.rag_mode == "candidates" and retrieved_answers:
            pool = dedup_keep_order(pool + retrieved_answers)

        pool = pool[: args.max_candidates]

        # Decide final answer:
        if voted is not None:
            best = voted
        else:
            best = pool[0] if pool else ""
            best_lp = -math.inf
            for c in pool:
                if not c:
                    continue
                lp = score_completion_logprob(model, tok, prompt, c, device=device)
                if lp > best_lp:
                    best_lp = lp
                    best = c

        # Numeric enforcement if needed
        best = enforce_numeric_if_needed(q, best)

        out_row = {"ID": r["ID"], "answer": best}
        if has_uid:
            out_row["uid"] = r["uid"]
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    if "uid" in out_df.columns:
        out_df = out_df[["uid", "ID", "answer"]]

    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
