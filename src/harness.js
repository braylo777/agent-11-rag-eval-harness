// The harness: score any retriever's rankings and any answer set against the labelled data.
import { precisionAtK, recallAtK, reciprocalRank, mean, r3, citationOverlap, auc } from "./metrics.js";
import { bm25Index } from "./bm25.js";
import { DATA } from "./data.js";

export function scoreRetrieval(rankings, queries = DATA.queries, ks = [1, 3, 5]) {
  const out = { n: queries.length };
  for (const k of ks) {
    out[`precision@${k}`] = r3(mean(queries.map((q) => precisionAtK(rankings[q.id], q.relevant, k))));
    out[`recall@${k}`] = r3(mean(queries.map((q) => recallAtK(rankings[q.id], q.relevant, k))));
  }
  out.mrr = r3(mean(queries.map((q) => reciprocalRank(rankings[q.id], q.relevant))));
  return out;
}
export function bm25Rankings(data = DATA) { const search = bm25Index(data.chunks); return Object.fromEntries(data.queries.map((q) => [q.id, search(q.q).map((h) => h.id)])); }
export function scoreFaithfulness(answers = DATA.answers, threshold = 0.5, chunks = DATA.chunks) {
  const text = Object.fromEntries(chunks.map((c) => [c.id, c.text]));
  const rows = answers.map((a) => ({ ...a, overlap: r3(citationOverlap(a.answer, a.cites.map((id) => text[id] ?? ""))) }));
  const pred = rows.map((r) => (r.overlap >= threshold ? 1 : 0)), lab = rows.map((r) => r.label);
  const tp = pred.filter((p, i) => p && lab[i]).length, fp = pred.filter((p, i) => p && !lab[i]).length, fn = pred.filter((p, i) => !p && lab[i]).length;
  return { n: rows.length, threshold, auc: r3(auc(rows.map((r) => r.overlap), lab)), accuracy: r3(pred.filter((p, i) => p === lab[i]).length / rows.length),
    precision: r3(tp / (tp + fp || 1)), recall: r3(tp / (tp + fn || 1)), mean_overlap_faithful: r3(mean(rows.filter((r) => r.label).map((r) => r.overlap))), mean_overlap_unfaithful: r3(mean(rows.filter((r) => !r.label).map((r) => r.overlap))), rows };
}
// LLM-as-judge faithfulness on Workers AI. Returns 1 (supported) / 0 (not supported).
export const JUDGE_PROMPT = "You are a strict fact checker. Reply with exactly SUPPORTED if every claim in the ANSWER is stated in the SOURCES, otherwise reply exactly UNSUPPORTED.";
export async function judge(env, answer, sources, ask) {
  const r = await ask(env, JUDGE_PROMPT, `SOURCES:\n${sources.join("\n")}\n\nANSWER: ${answer}`, { max_tokens: 5 });
  return /^\W*supported/i.test(r) ? 1 : 0;
}
