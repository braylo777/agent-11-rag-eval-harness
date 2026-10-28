import { test } from "node:test";
import assert from "node:assert/strict";
import { precisionAtK, recallAtK, reciprocalRank, citationOverlap, auc } from "../src/metrics.js";
import { bm25Index } from "../src/bm25.js";
import worker from "../src/index.js";
test("edge: k larger than ranking, empty ranking", () => { assert.equal(recallAtK([], ["a"], 5), 0); assert.equal(reciprocalRank([], ["a"]), 0); assert.ok(precisionAtK(["a"], ["a"], 1) === 1); });
test("edge: metrics stay in [0,1]", () => { for (const v of [precisionAtK(["a","b"],["a","b"],2), recallAtK(["a","b"],["a"],2), citationOverlap("x y", ["x"]), auc([0.3,0.7,0.2],[0,1,1])]) assert.ok(v >= 0 && v <= 1); });
test("edge: BM25 on query with no matching terms returns no positive score", () => { const s = bm25Index([{ id: "a", text: "cats purr" }]); const r = s("zebra quantum"); assert.ok(r.length === 0 || r[0].score <= 0); });
test("edge: unknown route is not 200 OK HTML crash", async () => { const r = await worker.fetch(new Request("https://x/does-not-exist"), {}); assert.ok(r.status < 500); });
