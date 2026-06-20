"""Corpus loading + the two extra scorers (choice, rouge) on top of lib.score()."""
import json
import sys

from . import lib
from .config import DATA_DIR, SUMMARIZATION


def load_corpus(name, limit):
    """Read bench/data/<name>.jsonl into (case_name, messages, meta). Handles the friendly
    {context?, question, system?} shape (request form with tools is out of scope)."""
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"WARNING: {path} missing - run download.py", file=sys.stderr)
        return []
    cases, kept = [], 0
    for ln in path.read_text().splitlines():
        if not ln.strip() or kept >= limit:
            continue
        v = json.loads(ln)
        scorer = v.get("scorer", "contains")
        if v.get("gold") is None or "request" in v:
            # request-form rows carry tools we don't yet plumb through both libs (TODO).
            continue
        msgs = []
        if v.get("system"):
            msgs.append({"role": "system", "content": v["system"]})
        ctx = next((v[k] for k in ("context", "input", "passage", "document") if v.get(k)), None)
        if ctx:
            msgs.append({"role": "user", "content": ctx})
        q = next((v[k] for k in ("question", "query", "prompt") if v.get(k)), None)
        if q:
            msgs.append({"role": "user", "content": q})
        if not msgs:
            continue
        if name in SUMMARIZATION:
            scorer = "rouge"
        meta = {"corpus": name, "question": q, "gold": v["gold"], "scorer": scorer}
        cases.append((v["name"], msgs, meta))
        kept += 1
    return cases


_ROUGE = None


def _rouge_l(answer, gold):
    """ROUGE-L F-measure (stemmed) - the standard summarization metric, not token-F1."""
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE.score(str(gold), answer or "")["rougeL"].fmeasure


def score_v2(scorer, answer, gold):
    """lib.score() plus 'choice' (truthfulqa MC1) and 'rouge' (summarization). Returns a
    float in [0,1]; callers decide a threshold for binary 'correct'."""
    if scorer == "choice":
        a = (answer or "").strip()
        # first standalone A-D letter the model emits
        for ch in a:
            if ch.upper() in "ABCDEFGH":
                return 1.0 if ch.upper() == str(gold).strip().upper() else 0.0
        return 0.0
    if scorer == "rouge":
        return _rouge_l(answer, gold)
    return lib.score(scorer, answer, gold)
