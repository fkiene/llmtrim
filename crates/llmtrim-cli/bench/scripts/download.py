#!/usr/bin/env python3
"""Download + normalize the benchmark corpora into bench/data/<name>.jsonl.

Each output line is a llmtrim bench case (see bench::load_bench_corpus):
friendly `{context?, question, gold, scorer, system?}` or explicit `{request, gold, scorer}`.
Real public datasets via the HF datasets-server (no auth). Pins dataset id/config/split
+ a sha256 of every output file in bench/data/manifest.json, so a run is reproducible
and a silent upstream change is detectable.

Usage:  python3 bench/download.py [N_per_corpus]   (default 40)
"""
import datetime
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
# Optional comma-list of corpus names to (re)fetch; the rest are left untouched and their
# manifest entries are preserved. Omit to rebuild every corpus.
ONLY = set(filter(None, (sys.argv[2].split(",") if len(sys.argv) > 2 else []))) or None
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bench/ root (this script lives in bench/scripts/)
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
BASE = "https://datasets-server.huggingface.co/rows"


def fetch(dataset, config, split, n):
    """Page through datasets-server (<=100/call) and return up to n raw rows."""
    rows, offset = [], 0
    while len(rows) < n:
        length = min(100, n - len(rows))
        url = f"{BASE}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    batch = json.load(r).get("rows", [])
                break
            except Exception as e:  # transient datasets-server hiccups
                if attempt == 3:
                    raise
                time.sleep(2 + attempt)
        if not batch:
            break
        rows.extend(x["row"] for x in batch)
        offset += length
    return rows[:n]


def write(name, cases, source):
    path = os.path.join(DATA, f"{name}.jsonl")
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"  {name:12} {len(cases):4} cases  sha256={sha[:16]}  <- {source}")
    return {"file": f"{name}.jsonl", "cases": len(cases), "sha256": sha, "source": source}


# ---- per-corpus normalizers -------------------------------------------------

def norm_gsm8k(rows):
    out = []
    for i, r in enumerate(rows):
        m = re.search(r"####\s*([-\d,.]+)", r["answer"])
        if not m:
            continue
        out.append({
            "name": f"gsm8k-{i}",
            "question": r["question"] + "\nThink step by step, then give the final number.",
            "gold": m.group(1).replace(",", ""),
            "scorer": "numeric",
        })
    return out


def norm_humaneval(rows):
    out = []
    for r in rows:
        out.append({
            "name": r["task_id"].replace("/", "-"),
            "system": "Complete the Python function. Respond with only the full function definition in a single code block, no prose.",
            "question": r["prompt"],
            "gold": {"test": r["test"], "entry_point": r["entry_point"]},
            "scorer": "pass@1",
        })
    return out


SQUAD_INSTR = (
    "\nAnswer with the shortest exact span from the context. "
    "If the context does not contain the answer, reply with exactly: unanswerable."
)


def norm_squad(rows):
    """SQuAD v2 extractive QA, keeping BOTH answerable and unanswerable rows. Answerable:
    token-F1 against the gold span. Unanswerable (no gold text): gold sentinel
    "unanswerable", scored by containment, so a correct refusal is a hit (the SQuAD v2
    no-answer contract). Balanced ~50/50 so the no-answer skill is actually measured."""
    answerable, unanswerable = [], []
    for i, r in enumerate(rows):
        texts = (r.get("answers") or {}).get("text") or []
        if texts:
            answerable.append({
                "name": f"squad-{i}",
                "context": r["context"],
                "question": r["question"] + SQUAD_INSTR,
                "gold": texts[0],
                "scorer": "f1",
            })
        else:
            unanswerable.append({
                "name": f"squad-{i}-noans",
                "context": r["context"],
                "question": r["question"] + SQUAD_INSTR,
                "gold": "unanswerable",
                "scorer": "contains",
            })
    half = N // 2
    out = answerable[:half] + unanswerable[:N - half]
    if len(out) < N:
        print(f"  WARNING: squad2 has only {len(out)} cases (wanted {N}); "
              f"answerable={len(answerable)} unanswerable={len(unanswerable)} "
              f"(raise the fetch multiplier)")
    return out


def norm_truthfulqa(rows):
    """TruthfulQA MC1: one true answer among distractors. Each case is a lettered
    constrained choice; gold = the letter of the true option. Choices are shuffled with a
    per-row deterministic seed so the correct answer isn't always position A. Scored by
    `choice` (the selected letter), the standard MC1 metric — deterministic, no judge."""
    out = []
    for i, r in enumerate(rows):
        mc1 = r.get("mc1_targets") or {}
        choices = list(mc1.get("choices") or [])
        labels = list(mc1.get("labels") or [])
        if len(choices) < 2 or 1 not in labels:
            continue
        correct = choices[labels.index(1)]
        order = list(range(len(choices)))
        random.Random(i).shuffle(order)
        letters = [chr(ord("A") + k) for k in range(len(order))]
        lines, gold = [], None
        for letter, idx in zip(letters, order):
            lines.append(f"{letter}) {choices[idx]}")
            if choices[idx] == correct:
                gold = letter
        question = (
            r["question"]
            + "\n\n" + "\n".join(lines)
            + "\n\nAnswer with the single letter of the correct option."
        )
        out.append({
            "name": f"truthfulqa-{i}",
            "question": question,
            "gold": gold,
            "scorer": "choice",
        })
    return out


def norm_hotpot(rows):
    out = []
    for i, r in enumerate(rows):
        ctx = r["context"]
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        paras = [f"{t}: {''.join(s)}" for t, s in zip(titles, sents)]
        out.append({
            "name": f"hotpot-{i}",
            "context": "\n\n".join(paras),
            "question": r["question"] + "\nAnswer concisely.",
            "gold": r["answer"],
            "scorer": "f1",
        })
    return out


def json_objects(s):
    """Balanced-brace scan → every top-level {...} substring (handles deep nesting
    that a regex can't, e.g. glaive tool defs with nested parameters/properties)."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                objs.append(s[start:i + 1])
    return objs


def norm_glaive(rows):
    """Extract (tools, conversation up to first call, gold call) from glaive."""
    out = []
    for i, r in enumerate(rows):
        sysmsg, chat = r.get("system", ""), r.get("chat", "")
        tools = []
        for raw in json_objects(sysmsg):
            try:
                td = json.loads(raw)
                if "name" in td:
                    tools.append({"type": "function", "function": td})
            except Exception:
                pass
        fc = chat.find("<functioncall>")
        if not tools or fc < 0:
            continue
        # gold = name of that first call.
        name_m = re.search(r'"name"\s*:\s*"([^"]+)"', chat[fc:])
        if not name_m:
            continue
        # messages = USER/ASSISTANT turns before the call.
        messages, role = [], None
        for tok in re.split(r"\n*(USER:|ASSISTANT:)\s*", chat[:fc]):
            if tok == "USER:":
                role = "user"
            elif tok == "ASSISTANT:":
                role = "assistant"
            elif role and tok.strip():
                messages.append({"role": role, "content": tok.replace("<|endoftext|>", "").strip()})
                role = None
        if not any(m["role"] == "user" for m in messages):
            continue
        request = {"model": "x", "messages": messages, "tools": tools}
        out.append({
            "name": f"glaive-{i}",
            "request": json.dumps(request, ensure_ascii=False),
            "gold": json.dumps({"name": name_m.group(1)}),
            "scorer": "tool",
        })
    return out


BFCL_REPO = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
# BFCL Python type names → JSON-schema types, so the tools payload is a valid request.
_JSON_TYPE = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "bool": "boolean"}


def _schemaize(node):
    """Recursively rewrite BFCL's Python type names into JSON-schema types in place."""
    if isinstance(node, dict):
        if isinstance(node.get("type"), str):
            node["type"] = _JSON_TYPE.get(node["type"], node["type"])
        for v in node.values():
            _schemaize(v)
    elif isinstance(node, list):
        for v in node:
            _schemaize(v)
    return node


# `live_multiple` is the multi-tool BFCL slice (2-37 candidate functions per query),
# where tool selection has irrelevant schemas to drop. The single-tool `simple` slice
# leaves nothing to compress, so it's not a useful row for a tool-compression claim.
BFCL_CATEGORY = "BFCL_v3_live_multiple"


def fetch_bfcl(n):
    """Join BFCL's prompt file with its possible_answer file by id (raw HF files; the
    datasets-server can't view this repo). Returns up to n single-turn cases from the
    multi-tool category, gold = the function the call must invoke."""
    def load(path):
        url = f"{BFCL_REPO}/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    prompts = load(f"{BFCL_CATEGORY}.json")
    gold_by_id = {g["id"]: g for g in load(f"possible_answer/{BFCL_CATEGORY}.json")}
    out = []
    for p in prompts:
        if len(out) >= n:
            break
        gt = gold_by_id.get(p["id"], {}).get("ground_truth") or []
        if not gt or not isinstance(gt[0], dict):
            continue
        fn_name = next(iter(gt[0]))  # the function the call must invoke
        messages = p["question"][0]  # single-turn: one message list
        tools = [
            {"type": "function", "function": _schemaize(dict(f))}
            for f in p.get("function", [])
            if f.get("name")
        ]
        if not tools:
            continue
        request = {"model": "x", "messages": messages, "tools": tools}
        out.append({
            "name": f"bfcl-{p['id']}",
            "request": json.dumps(request, ensure_ascii=False),
            "gold": json.dumps({"name": fn_name}),
            "scorer": "tool",
        })
    return out


def norm_adult(rows):
    """Bundle real census rows into uniform JSON record arrays + a deterministic
    aggregate question — exercises Stage D serialization (TOON/CSV) losslessly."""
    cols = ["age", "education", "occupation", "hours_worked_per_week", "marital_status"]
    out, K = [], 12
    for b in range(0, len(rows) - K, K):
        bundle = rows[b:b + K]
        records = [{c: r.get(c) for c in cols} for r in bundle]
        # most common occupation in the bundle → count is the gold.
        occs = [r["occupation"] for r in bundle]
        target = max(set(occs), key=occs.count)
        gold = str(occs.count(target))
        content = (
            json.dumps(records, ensure_ascii=False)
            + f"\n\nIn the JSON array above, how many records have occupation equal to \"{target}\"? Answer with just the number."
        )
        out.append({
            "name": f"adult-{b}",
            "request": json.dumps({"model": "x", "messages": [{"role": "user", "content": content}]}),
            "gold": gold,
            "scorer": "numeric",
        })
    return out


def norm_cnn(rows):
    out = []
    for i, r in enumerate(rows):
        out.append({
            "name": f"cnn-{i}",
            "context": r["article"],
            "question": "Summarize the article above in 2-3 sentences.",
            "gold": r["highlights"],
            "scorer": "f1",
        })
    return out


def norm_dolly(rows):
    """Instruction-following / long-form generation (output-heavy real usage). Keep
    only responses ≥200 chars so output_control has something to cut; judge-scored."""
    out = []
    for i, r in enumerate(rows):
        resp = (r.get("response") or "").strip()
        if len(resp) < 200:
            continue
        case = {
            "name": f"dolly-{i}",
            "question": (r.get("instruction") or "").strip(),
            "gold": resp,
            "scorer": "judge",
        }
        ctx = (r.get("context") or "").strip()
        if ctx:
            case["context"] = ctx
        out.append(case)
    return out


def norm_cache(rows):
    """Synthetic cache corpus: conversations with a long shared system prompt.

    Re-uses ultrachat conversations but wraps each in a long static system prompt
    so Stage A (cache-zone marking) has a shared prefix to annotate. Tests that
    the cache preset leaves quality intact while allowing providers to bill at
    the cached-input rate.
    """
    # A ~1280-token (≈5120-char) shared system prompt — representative of real
    # agent/RAG deployments where a long context is amortised across many turns.
    SYSTEM = (
        "You are a knowledgeable AI assistant. You provide accurate, helpful, "
        "and concise answers. When answering questions: be direct and clear; "
        "cite sources when relevant; admit uncertainty rather than guessing; "
        "use examples to clarify abstract concepts; and keep responses focused "
        "on what was asked. " * 40  # ~1280 tokens
    ).strip()
    out = []
    for i, row in enumerate(rows):
        if len(out) >= 40:
            break
        msgs = row.get("messages") or row.get("conversation") or []
        if len(msgs) < 2:
            continue
        # ultrachat turns alternate and end on assistant: the final assistant turn is
        # the gold, the user turn before it is the question.
        if msgs[-1].get("role") != "assistant" or msgs[-2].get("role") != "user":
            continue
        gold = (msgs[-1].get("content") or "").strip()
        question = (msgs[-2].get("content") or "").strip()
        if not gold or not question or len(question) > 300:
            continue
        request = {"model": "x", "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]}
        out.append({
            "name": f"cache-{i}",
            "request": json.dumps(request, ensure_ascii=False),
            "gold": gold,
            "scorer": "judge",
        })
    return out


def norm_chat(rows):
    """Multi-turn chat (the most common real LLM workload): the conversation history is
    the request, the final assistant turn is the gold. Output-heavy + long shared
    context. Judge-scored."""
    out = []
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            continue
        gold = (msgs[-1].get("content") or "").strip()
        if len(gold) < 150:
            continue
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs[:-1]
            if m.get("content")
        ]
        if not history:
            continue
        request = {"model": "x", "messages": history}
        out.append({
            "name": f"chat-{i}",
            "request": json.dumps(request, ensure_ascii=False),
            "gold": gold,
            "scorer": "judge",
        })
    return out


CORPORA = [
    ("gsm8k", "openai/gsm8k", "main", "test", norm_gsm8k),
    ("humaneval", "openai/openai_humaneval", "openai_humaneval", "test", norm_humaneval),
    ("dolly", "databricks/databricks-dolly-15k", "default", "train", norm_dolly),
    ("hotpotqa", "hotpotqa/hotpot_qa", "distractor", "validation", norm_hotpot),
    ("glaive", "glaiveai/glaive-function-calling-v2", "default", "train", norm_glaive),
    ("chat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", norm_chat),
    ("cnn", "abisee/cnn_dailymail", "3.0.0", "validation", norm_cnn),
    ("cache", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", norm_cache),
    ("truthfulqa", "truthfulqa/truthful_qa", "multiple_choice", "validation", norm_truthfulqa),
    ("squad2", "rajpurkar/squad_v2", "squad_v2", "validation", norm_squad),
]


def main():
    manifest_path = os.path.join(DATA, "manifest.json")
    # Merge into the existing manifest when fetching only a subset, so the untouched
    # corpora keep their pinned sha256 and the diff stays surgical.
    if ONLY and os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        manifest["fetched"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        manifest.setdefault("corpora", {})
    else:
        manifest = {
            "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "n_requested": N,
            "corpora": {},
        }
    todo = [c for c in CORPORA if ONLY is None or c[0] in ONLY]
    print(f"downloading {len(todo)} corpora, up to {N} cases each:")
    for name, ds, cfg, split, fn in todo:
        try:
            # over-fetch where normalizers drop/bundle rows (unanswerable, parse fails,
            # or 12 rows → 1 record-array case).
            mult = {"glaive": 3, "dolly": 4, "chat": 3, "squad2": 6}.get(name, 1)
            raw = fetch(ds, cfg, split, N * mult)
            cases = fn(raw)[:N]
            entry = write(name, cases, f"{ds}:{cfg}:{split}")
            entry.update({"dataset": ds, "config": cfg, "split": split})
            manifest["corpora"][name] = entry
        except Exception as e:
            print(f"  {name:12} FAILED: {e}")

    # BFCL ships as raw repo files, not a datasets-server view, so it has its own fetch.
    if ONLY is None or "bfcl" in ONLY:
        try:
            cases = fetch_bfcl(N)
            entry = write("bfcl", cases, f"{BFCL_REPO} ({BFCL_CATEGORY})")
            entry.update({
                "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                "config": BFCL_CATEGORY,
                "split": "test",
            })
            manifest["corpora"]["bfcl"] = entry
        except Exception as e:
            print(f"  {'bfcl':12} FAILED: {e}")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
