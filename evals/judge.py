"""Live LLM-judge faithfulness eval (Llama 3.3 70B on Workers AI) over the 40 labelled answers,
throttled to the demo's 5 req/min. Usage: BASE=https://... python3 evals/judge.py"""
import json, os, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent.parent
BASE = os.environ.get("BASE", "https://agent-11-rag-eval-harness.braylo777.workers.dev")
GAP = float(os.environ.get("EVAL_GAP_MS", "12500")) / 1000
D = json.loads((root / "data" / "data.json").read_text())


def call(url, body):
    for _ in range(6):
        time.sleep(GAP)
        t = time.time()
        req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=60); status = r.status; raw = r.read()
        except urllib.error.HTTPError as e:
            status = e.code; raw = e.read()
        if status == 429:
            time.sleep(30); continue
        try:
            b = json.loads(raw)
        except ValueError:
            b = None
        return {"status": status, "ms": (time.time() - t) * 1000, "body": b}
    raise RuntimeError("rate limited repeatedly")


ratio = lambda k, n: {"correct": k, "n": n, "rate": round(k / n, 4) if n else None}
rows = []
for a in D["answers"]:
    r = call(BASE + "/api/score?judge=1", {"answer": a["answer"], "cites": a["cites"]})
    b = r["body"] or {}
    rows.append({"qid": a["qid"], "label": a["label"], "judge": b.get("judge"), "error": b.get("judge_error") or (r["status"] if r["status"] != 200 else None), "ms": r["ms"]})
    print(len(rows), a["label"], rows[-1]["judge"], flush=True)
ok = [r for r in rows if r["judge"] is not None]
k = lambda f: sum(1 for r in ok if f(r))
path = root / "evals" / "results.json"
res = json.loads(path.read_text())
res["faithfulness"]["llm_judge"] = {"model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "run_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"), "n_scored": len(ok), "n": len(rows),
    "accuracy": ratio(k(lambda r: r["judge"] == r["label"]), len(ok)), "faithful_recall": ratio(k(lambda r: r["label"] and r["judge"]), k(lambda r: r["label"])),
    "unfaithful_caught": ratio(k(lambda r: not r["label"] and not r["judge"]), k(lambda r: not r["label"])), "rows": rows}
path.write_text(json.dumps(res, indent=2))
print(res["faithfulness"]["llm_judge"])
