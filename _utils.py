import re
import json
import ast
STRIP_CHARS = " \t\n\r\f\v.,;:!?'\"\\()[]{}"

def normalize_text(s: str) -> str:
    # Simple normalization for validation: lowercase + strip + remove extra spaces/punct
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove surrounding punctuation
    s = s.strip(STRIP_CHARS)

    return s

def parse_choices(choices_str: str) -> dict:
    # choices is stored as JSON-like string with keys A/B/C/D
    return json.loads(choices_str)

def parse_saq_annotations(ann_str: str):
    # annotations is a Python-literal string like: "[{'answers': ..., 'en_answers': ...}]"
    return ast.literal_eval(ann_str)

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