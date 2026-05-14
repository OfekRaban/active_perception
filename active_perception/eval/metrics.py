"""Simple answer-accuracy metrics for VQA-style evaluation."""
from __future__ import annotations
import re
import string
from typing import Optional


def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = re.sub(r'\s+', ' ', ans)
    ans = ans.translate(str.maketrans('', '', string.punctuation))
    return ans.strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_match(pred: str, gold: str) -> bool:
    return normalize_answer(gold) in normalize_answer(pred)


def extract_final_answer(text: str) -> str:
    """Extract answer from text that may contain <think>...</think> blocks."""
    think_end = text.rfind("</think>")
    if think_end != -1:
        return text[think_end + len("</think>"):].strip()
    return text.strip()


def compute_accuracy(predictions: list, gold_answers: list, mode: str = "exact") -> float:
    assert len(predictions) == len(gold_answers)
    if not predictions:
        return 0.0
    fn = exact_match if mode == "exact" else contains_match
    correct = sum(fn(p, g) for p, g in zip(predictions, gold_answers))
    return correct / len(predictions)
