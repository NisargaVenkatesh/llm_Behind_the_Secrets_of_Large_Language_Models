import re
import json
import ast
import pandas as pd

STRIP_CHARS = " \t\n\r\f\v.,;:!?'\"\\()[]{}"

_LEADING_FILLER_RE = re.compile(
    r"^(?:the|a|an)\s+|^(?:it\s+is|it's|its|this\s+is|answer\s*[:\-])\s+",
    re.IGNORECASE,
)

_TRAILING_FILLER_RE = re.compile(r"\s+(?:please|thanks|thank you)$", re.IGNORECASE)


def normalize_text(s: str) -> str:
    # Simple normalization for validation: lowercase + strip + remove extra spaces/punct
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove surrounding punctuation
    s = s.strip(STRIP_CHARS)
    return s


def parse_choices(choices_str):
    # choices is stored as JSON-like string with keys A/B/C/D
    if choices_str is None or (isinstance(choices_str, float) and pd.isna(choices_str)):
        return {"A": "", "B": "", "C": "", "D": ""}
    if isinstance(choices_str, dict):
        return choices_str
    return json.loads(str(choices_str))


def parse_saq_annotations(ann_str: str):
    # annotations is a Python-literal string like: "[{'answers': ..., 'en_answers': ...}]"
    if ann_str is None or (isinstance(ann_str, float) and pd.isna(ann_str)):
        return []
    s = str(ann_str).strip()
    if not s:
        return []
    return ast.literal_eval(s)


def first_line(s: str) -> str:
    s = (s or "").strip()
    return s.splitlines()[0].strip() if s else ""


def clean_saq_answer(s: str) -> str:
    # Keep it short + grader-friendly
    s = first_line(s)
    s = s.strip()
    # remove a leading "answer:" if the model includes it
    if s.lower().startswith("answer:"):
        s = s.split(":", 1)[1].strip()
    # remove trailing punctuation
    s = s.rstrip(" .,!?:;")
    return s.strip()


def normalize_answer_for_submission(s: str) -> str:
    """
    Stronger normalization than normalize_text(), aimed at exact-match graders.
    - lowercases, strips punctuation/whitespace (via normalize_text)
    - removes leading articles and common filler prefixes
    - trims trailing polite filler
    """
    s = normalize_text(s)
    s = _LEADING_FILLER_RE.sub("", s).strip()
    s = _TRAILING_FILLER_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s
