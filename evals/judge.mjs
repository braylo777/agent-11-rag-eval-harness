// Live LLM-judge faithfulness eval (Llama 3.3 70B on Workers AI) over the 40 labelled answers, throttled to the demo's 5 req/min.
import fs from "node:fs";
import { call, ratio } from "./lib.mjs";
const BASE = process.env.BASE || "https://agent-11-rag-eval-harness.braylo777.workers.dev";
const D = JSON.parse(fs.readFileSync(new URL("./data.json", import.meta.url)));
const rows = [];
for (const a of D.answers) {
  const r = await call(BASE + "/api/score?judge=1", { method: "POST", body: JSON.stringify({ answer: a.answer, cites: a.cites }) });
  rows.push({ qid: a.qid, label: a.label, judge: r.body?.judge ?? null, error: r.body?.judge_error || (r.status !== 200 ? r.status : undefined), ms: r.ms });
  console.log(rows.length, a.label, rows.at(-1).judge);
}
const ok = rows.filter((r) => r.judge !== null), k = (f) => ok.filter(f).length;
const res = JSON.parse(fs.readFileSync(new URL("./results.json", import.meta.url)));
res.faithfulness.llm_judge = { model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast", run_at: new Date().toISOString(), n_scored: ok.length, n: rows.length,
  accuracy: ratio(k((r) => r.judge === r.label), ok.length), faithful_recall: ratio(k((r) => r.label && r.judge), k((r) => r.label)), unfaithful_caught: ratio(k((r) => !r.label && !r.judge), k((r) => !r.label)), rows };
fs.writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(res, null, 2));
console.log(res.faithfulness.llm_judge);
