import argparse
from pathlib import Path
import pandas as pd
import re

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def normalize_pred_letter(x: str) -> str:
    """
    Robustly extract A/B/C/D from messy strings like:
      "C", "c", "Answer: C", "The answer is (B).", " option d "
    Returns one of {"A","B","C","D"} or "" if not found.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    m = LETTER_RE.search(s)
    if not m:
        return ""
    return m.group(1).upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", type=str, required=True, help="Input TSV with columns: MCQID, pred_letter")
    ap.add_argument("--out_tsv", type=str, required=True, help="Output TSV with columns: MCQID, A, B, C, D")
    args = ap.parse_args()

    in_path = Path(args.in_tsv)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t")

    if "MCQID" not in df.columns:
        raise ValueError(f"Input missing MCQID column. Found: {list(df.columns)}")
    if "pred_letter" not in df.columns:
        raise ValueError(f"Input missing pred_letter column. Found: {list(df.columns)}")

    df["pred_letter"] = df["pred_letter"].apply(normalize_pred_letter)

    invalid = (df["pred_letter"] == "").sum()
    if invalid > 0:
       
        print(f"[WARN] {invalid} rows had invalid pred_letter; setting them to 'A' as fallback.")
        df.loc[df["pred_letter"] == "", "pred_letter"] = "A"

    out = pd.DataFrame({"MCQID": df["MCQID"].astype(str)})
    for L in ["A", "B", "C", "D"]:
        out[L] = (df["pred_letter"] == L)

    bad = (out[["A","B","C","D"]].sum(axis=1) != 1).sum()
    if bad != 0:
        raise ValueError(f"Output has {bad} rows without exactly one True (this should never happen).")

    out.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote {out_path} rows={len(out)} cols={list(out.columns)}")
    print("[INFO] Pred letter counts:\n", df["pred_letter"].value_counts())

if __name__ == "__main__":
    main()
