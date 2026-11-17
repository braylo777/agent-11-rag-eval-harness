"""Offline harness run (no Workers AI): BM25 retrieval + citation-overlap faithfulness.
Merges data/offline.json. Writes evals/results.json."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from agent11.harness import bm25_rankings, score_faithfulness, score_retrieval  # noqa: E402

res_path = root / "evals" / "results.json"
off = json.loads((root / "data" / "offline.json").read_text())
old = json.loads(res_path.read_text()) if res_path.exists() else {}
f = score_faithfulness()
out = {"project": "agent-11-rag-eval-harness", "run_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
       "dataset": {"chunks": 22, "queries": 20, "answers": 40},
       "retrieval": {"bm25": score_retrieval(bm25_rankings()), **off["models"]},
       "faithfulness": {"citation_overlap": f, "llm_judge": (old.get("faithfulness") or {}).get("llm_judge") or {"status": "not yet run", "reason": "Workers AI free daily quota exhausted on Oct 27, 2026", "rerun": "BASE=https://agent-11-rag-eval-harness.braylo777.workers.dev python3 evals/judge.py"}}}
res_path.write_text(json.dumps(out, indent=2))
print(json.dumps({"retrieval": out["retrieval"], "faithfulness": {**f, "rows": None}}, indent=1))
