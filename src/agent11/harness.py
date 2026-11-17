"""The harness: score any retriever's rankings and any answer set against the labelled data."""
import json, re
from pathlib import Path

from .bm25 import bm25_index
from .metrics import auc, citation_overlap, mean, precision_at_k, r3, recall_at_k, reciprocal_rank

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA = json.loads((DATA_DIR / "data.json").read_text())
OFFLINE = json.loads((DATA_DIR / "offline.json").read_text())
JUDGE_PROMPT = "You are a strict fact checker. Reply with exactly SUPPORTED if every claim in the ANSWER is stated in the SOURCES, otherwise reply exactly UNSUPPORTED."
MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def score_retrieval(rankings, queries=None, ks=(1, 3, 5)):
    queries = DATA["queries"] if queries is None else queries
    out = {"n": len(queries)}
    for k in ks:
        out[f"precision@{k}"] = r3(mean([precision_at_k(rankings[q["id"]], q["relevant"], k) for q in queries]))
        out[f"recall@{k}"] = r3(mean([recall_at_k(rankings[q["id"]], q["relevant"], k) for q in queries]))
    out["mrr"] = r3(mean([reciprocal_rank(rankings[q["id"]], q["relevant"]) for q in queries]))
    return out


def bm25_rankings(data=None):
    data = DATA if data is None else data
    search = bm25_index(data["chunks"])
    return {q["id"]: [h["id"] for h in search(q["q"])] for q in data["queries"]}


def score_faithfulness(answers=None, threshold=0.5, chunks=None):
    answers = DATA["answers"] if answers is None else answers
    chunks = DATA["chunks"] if chunks is None else chunks
    text = {c["id"]: c["text"] for c in chunks}
    rows = [{**a, "overlap": r3(citation_overlap(a["answer"], [text.get(i, "") for i in a["cites"]]))} for a in answers]
    pred = [1 if r["overlap"] >= threshold else 0 for r in rows]
    lab = [r["label"] for r in rows]
    tp = sum(1 for p, l in zip(pred, lab) if p and l)
    fp = sum(1 for p, l in zip(pred, lab) if p and not l)
    fn = sum(1 for p, l in zip(pred, lab) if not p and l)
    return {"n": len(rows), "threshold": threshold, "auc": r3(auc([r["overlap"] for r in rows], lab)), "accuracy": r3(sum(1 for p, l in zip(pred, lab) if p == l) / len(rows)),
            "precision": r3(tp / ((tp + fp) or 1)), "recall": r3(tp / ((tp + fn) or 1)),
            "mean_overlap_faithful": r3(mean([r["overlap"] for r in rows if r["label"]])), "mean_overlap_unfaithful": r3(mean([r["overlap"] for r in rows if not r["label"]])), "rows": rows}


def judge(ai, answer, sources):
    """LLM-as-judge faithfulness. ai(model, input) -> dict. Returns 1 (supported) / 0 (not supported)."""
    r = ai(MODEL, {"messages": [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": "SOURCES:\n" + "\n".join(sources) + f"\n\nANSWER: {answer}"}], "max_tokens": 5})
    v = r.get("response") if isinstance(r, dict) else None
    t = v if isinstance(v, str) else json.dumps(v if v is not None else r, separators=(",", ":"), ensure_ascii=False)
    return 1 if re.match(r"[^A-Za-z0-9_]*supported", t, re.I) else 0


def cap(s, n=500):
    return isinstance(s, str) and bool(s.strip()) and len(s) <= n


def handle(ai, method, path, body=None, judge_flag=False):
    """Pure router: (status, json); status 0 = serve the page. (Rate limiting is applied by the Worker.)"""
    if path == "/api/data":
        return 200, {"chunks": DATA["chunks"], "queries": [{"id": q["id"], "q": q["q"], "relevant": q["relevant"]} for q in DATA["queries"]]}
    if path == "/api/eval":
        f = {k: v for k, v in score_faithfulness().items() if k != "rows"}
        return 200, {"live_bm25": score_retrieval(bm25_rankings()), "citation_overlap": f, "offline": OFFLINE}
    if method == "POST" and path == "/api/score":
        b = body if isinstance(body, dict) else {}
        answer, cites = b.get("answer"), b.get("cites")
        if not cap(answer) or not isinstance(cites, list) or len(cites) > 5:
            return 400, {"error": "answer (<=500 chars) and cites (chunk ids) required"}
        by_id = {c["id"]: c["text"] for c in DATA["chunks"]}
        src = [by_id[i] for i in cites if isinstance(i, str) and i in by_id and by_id[i]]
        if not src:
            return 400, {"error": "unknown chunk ids"}
        out = {"overlap": r3(citation_overlap(answer, src))}
        if judge_flag:
            try:
                out["judge"] = judge(ai, answer, src)
            except Exception as e:
                out["judge_error"] = str(e)
        return 200, out
    return 0, None
