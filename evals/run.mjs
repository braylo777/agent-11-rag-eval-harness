// Offline harness run (no Workers AI): BM25 retrieval + citation-overlap faithfulness. Merges evals/offline.json. Writes evals/results.json.
import fs from "node:fs";
import { scoreRetrieval, bm25Rankings, scoreFaithfulness } from "../src/harness.js";
const off = JSON.parse(fs.readFileSync(new URL("./offline.json", import.meta.url)));
const old = fs.existsSync(new URL("./results.json", import.meta.url)) ? JSON.parse(fs.readFileSync(new URL("./results.json", import.meta.url))) : {};
const f = scoreFaithfulness();
const out = { project: "agent-11-rag-eval-harness", run_at: new Date().toISOString(), dataset: { chunks: 22, queries: 20, answers: 40 },
  retrieval: { bm25: scoreRetrieval(bm25Rankings()), ...Object.fromEntries(Object.entries(off.models).map(([k, v]) => [k, v])) },
  faithfulness: { citation_overlap: f, llm_judge: old.faithfulness?.llm_judge ?? { status: "not yet run", reason: "Workers AI free daily quota exhausted on Oct 27, 2026", rerun: "BASE=https://agent-11-rag-eval-harness.braylo777.workers.dev node evals/judge.mjs" } } };
fs.writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(out, null, 2));
console.log(JSON.stringify({ retrieval: out.retrieval, faithfulness: { ...f, rows: undefined } }, null, 1));
