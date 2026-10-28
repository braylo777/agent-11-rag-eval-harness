import { test } from "node:test";
import assert from "node:assert/strict";
import { precisionAtK, recallAtK, reciprocalRank, citationOverlap, auc, tokens } from "../src/metrics.js";
import { bm25Index } from "../src/bm25.js";
import { scoreRetrieval, scoreFaithfulness, bm25Rankings, judge } from "../src/harness.js";
import { DATA } from "../src/data.js";
import worker from "../src/index.js";

test("precision@k and recall@k", () => { assert.equal(precisionAtK(["a", "b", "c"], ["b", "z"], 2), 0.5); assert.equal(recallAtK(["a", "b", "c"], ["b", "z"], 3), 0.5); assert.equal(recallAtK(["a"], [], 1), 0); });
test("reciprocal rank: first hit position, 0 on miss", () => { assert.equal(reciprocalRank(["x", "y", "a"], ["a"]), 1 / 3); assert.equal(reciprocalRank(["x"], ["a"]), 0); });
test("AUC: perfect, inverted and tied scores", () => { assert.equal(auc([0.9, 0.8, 0.1], [1, 1, 0]), 1); assert.equal(auc([0.1, 0.9], [1, 0]), 0); assert.equal(auc([0.5, 0.5], [1, 0]), 0.5); });
test("tokens drop stopwords and fold plurals", () => { assert.deepEqual(tokens("The agents are routing tools"), ["agent", "rout", "tool"]); });
test("citation overlap: copied text = 1, unrelated = 0", () => { assert.equal(citationOverlap("vectors rank passages", ["Vectors rank the passages."]), 1); assert.equal(citationOverlap("bananas fly", ["Vectors rank passages."]), 0); assert.equal(citationOverlap("", ["x"]), 0); });
test("BM25 ranks the matching chunk first", () => { const s = bm25Index([{ id: "a", text: "cats purr softly" }, { id: "b", text: "JSON schema describes tools" }]); assert.equal(s("which schema describes tools")[0].id, "b"); });
test("labelled set is consistent", () => {
  const ids = new Set(DATA.chunks.map((c) => c.id)); assert.equal(DATA.chunks.length, 22); assert.equal(DATA.queries.length, 20); assert.equal(DATA.answers.length, 40);
  assert.ok(DATA.queries.every((q) => q.relevant.length && q.relevant.every((r) => ids.has(r)))); assert.equal(DATA.answers.filter((a) => a.label === 1).length, 20);
});
test("an oracle retriever scores perfect recall and MRR", () => { const m = scoreRetrieval(Object.fromEntries(DATA.queries.map((q) => [q.id, [...q.relevant, "zz"]]))); assert.equal(m.mrr, 1); assert.equal(m["recall@3"], 1); assert.equal(m["precision@1"], 1); });
test("BM25 and faithfulness results are deterministic", () => { assert.deepEqual(scoreRetrieval(bm25Rankings()), scoreRetrieval(bm25Rankings())); const f = scoreFaithfulness(); assert.ok(f.auc > 0.5 && f.n === 40); });
test("judge parses SUPPORTED / UNSUPPORTED", async () => { assert.equal(await judge({}, "a", ["s"], async () => "SUPPORTED"), 1); assert.equal(await judge({}, "a", ["s"], async () => "UNSUPPORTED."), 0); });
test("HTTP: /api/eval and /api/score validation", async () => {
  const e = await (await worker.fetch(new Request("https://x/api/eval"), {})).json(); assert.equal(e.live_bm25.n, 20); assert.ok("bge-base-en-v1.5" in e.offline.models);
  const bad = await worker.fetch(new Request("https://x/api/score", { method: "POST", body: "{}" }), {}); assert.equal(bad.status, 400);
  const unk = await worker.fetch(new Request("https://x/api/score", { method: "POST", body: JSON.stringify({ answer: "hi", cites: ["nope"] }) }), {}); assert.equal(unk.status, 400);
  const ok = await (await worker.fetch(new Request("https://x/api/score", { method: "POST", body: JSON.stringify({ answer: "Approvals are logged for auditing.", cites: ["d9c2"] }) }), {})).json(); assert.equal(ok.overlap, 1);
});
test("HTTP: judge failure is reported, not a crash", async () => {
  const env = { AI: { run: async () => { throw new Error("quota"); } } };
  const r = await (await worker.fetch(new Request("https://x/api/score?judge=1", { method: "POST", body: JSON.stringify({ answer: "x y z", cites: ["d1c1"] }) }), env)).json(); assert.equal(r.judge_error, "quota");
});
