import argparse
from pathlib import Path
import pandas as pd

from _utils import parse_saq_annotations, normalize_text

LETTERS = ["A", "B", "C", "D"]


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def score_mcq(val_df: pd.DataFrame, pred_path: Path):
    preds = pd.read_csv(pred_path, sep="\t")


    if "pred_letter" in preds.columns:
        preds["pred_letter"] = preds["pred_letter"].astype(str).str.strip()
        preds = preds[["MCQID", "pred_letter"]]
    else:
        missing = [c for c in (["MCQID"] + LETTERS) if c not in preds.columns]
        if missing:
            raise ValueError(
                f"[MCQ] {pred_path} missing columns: {missing}. "
                f"Expected either (MCQID,pred_letter) or (MCQID,A,B,C,D)."
            )

        def row_to_letter(r):
            trues = [L for L in LETTERS if _to_bool(r[L])]
            if len(trues) == 1:
                return trues[0]
            return None

        preds["pred_letter"] = preds.apply(row_to_letter, axis=1)
        preds = preds[["MCQID", "pred_letter"]]

    merged = val_df.merge(preds, on="MCQID", how="left")
    merged["correct"] = (merged["pred_letter"] == merged["answer_idx"])

    overall = float(merged["correct"].mean())
    by_country = merged.groupby("country")["correct"].mean().sort_index()

    n = len(merged)
    missing_pred = int(merged["pred_letter"].isna().sum())
    invalid_pred = int(((merged["pred_letter"].notna()) & (~merged["pred_letter"].isin(LETTERS))).sum())

    return overall, by_country, {"rows": n, "missing_pred": missing_pred, "invalid_pred": invalid_pred}


def _build_acceptables_from_annotations(ann_str: str) -> set:
    ann = parse_saq_annotations(ann_str)
    acceptable = set()
    for a in ann:
        for ans in a.get("en_answers", []):
            acceptable.add(normalize_text(ans))
    return acceptable


def score_saq(val_df: pd.DataFrame, pred_path: Path):
    preds = pd.read_csv(pred_path, sep="\t", dtype=str)

    # normalize columns
    for c in ["uid", "ID", "answer"]:
        if c in preds.columns:
            preds[c] = preds[c].astype(str).str.strip()

    if "answer" in preds.columns:
        preds["answer"] = preds["answer"].fillna("").astype(str)

    # Decide merge key:
    # - Use uid if BOTH exist (correct for your validation split).
    # - Otherwise fall back to ID (Codabench test format).
    use_uid = ("uid" in val_df.columns) and ("uid" in preds.columns)

    val_df = val_df.copy()
    if "uid" in val_df.columns:
        val_df["uid"] = val_df["uid"].astype(str).str.strip()
    val_df["ID"] = val_df["ID"].astype(str).str.strip()

    if use_uid:
        key = "uid"
        # enforce uniqueness on uid
        if val_df[key].duplicated().any():
            raise ValueError("[SAQ] val has duplicated uid, expected unique uid per row.")
        if preds[key].duplicated().any():
            # If duplicates exist, keep first non-empty answer
            preds = preds.sort_values(by="answer", ascending=False).drop_duplicates(subset=[key], keep="first")

        merged = val_df.merge(preds[[key, "answer"]], on=key, how="left")
        merged["pred_norm"] = merged["answer"].fillna("").apply(normalize_text)
        merged["acceptable"] = merged["annotations"].apply(_build_acceptables_from_annotations)
        merged["correct"] = merged.apply(lambda r: r["pred_norm"] in r["acceptable"], axis=1)

        overall = float(merged["correct"].mean())
        by_country = merged.groupby("country")["correct"].mean().sort_index()

        empty_pred = int((merged["answer"].fillna("").str.strip() == "").sum())
        missing_pred = int(merged["answer"].isna().sum())

        return overall, by_country, {
            "rows": len(merged),
            "merge_key": "uid",
            "empty_pred": empty_pred,
            "missing_pred": missing_pred,
        }

    else:
        # Fallback: merge on ID (not ideal for val, but required for Codabench test format)
        key = "ID"
        if preds[key].duplicated().any():
            preds = preds.sort_values(by="answer", ascending=False).drop_duplicates(subset=[key], keep="first")

        # If val has duplicate IDs, evaluation is ambiguous.
        # We score row-by-row (each row keeps its own acceptable set),
        # but the same prediction will be reused for all rows with same ID.
        merged = val_df.merge(preds[[key, "answer"]], on=key, how="left")
        merged["pred_norm"] = merged["answer"].fillna("").apply(normalize_text)
        merged["acceptable"] = merged["annotations"].apply(_build_acceptables_from_annotations)
        merged["correct"] = merged.apply(lambda r: r["pred_norm"] in r["acceptable"], axis=1)

        overall = float(merged["correct"].mean())
        by_country = merged.groupby("country")["correct"].mean().sort_index()

        empty_pred = int((merged["answer"].fillna("").str.strip() == "").sum())
        missing_pred = int(merged["answer"].isna().sum())

        return overall, by_country, {
            "rows": len(merged),
            "merge_key": "ID",
            "empty_pred": empty_pred,
            "missing_pred": missing_pred,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="both", choices=["both", "mcq", "saq"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)

    val_mcq = pd.read_csv(data_dir / "val_mcq_split.csv")
    val_saq = pd.read_csv(data_dir / "val_saq_split.csv")

    mcq_path = pred_dir / "mcq_prediction.tsv"
    saq_path = pred_dir / "saq_prediction.tsv"

    if args.task in {"both", "mcq"}:
        if mcq_path.exists():
            mcq_overall, mcq_by, mcq_diag = score_mcq(val_mcq, mcq_path)
            print("MCQ overall:", round(mcq_overall, 4))
            print("MCQ by country:")
            print(mcq_by)
            print(f"[MCQ diag] rows={mcq_diag['rows']} missing_pred={mcq_diag['missing_pred']} invalid_pred={mcq_diag['invalid_pred']}")
        else:
            print(f"[MCQ] Skipping: not found -> {mcq_path}")

    if args.task in {"both", "saq"}:
        if saq_path.exists():
            saq_overall, saq_by, saq_diag = score_saq(val_saq, saq_path)
            print(f"\nSAQ overall: {round(saq_overall, 4)}")
            print("SAQ by country:")
            print(saq_by)
            print(f"[SAQ diag] rows={saq_diag['rows']} merge_key={saq_diag['merge_key']} missing_pred={saq_diag['missing_pred']} empty_pred={saq_diag['empty_pred']}")
        else:
            print(f"[SAQ] Skipping: not found -> {saq_path}")


if __name__ == "__main__":
    main()
