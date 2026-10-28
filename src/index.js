import { json, limited, body, cap, ask } from "./core.js";
import { scoreRetrieval, bm25Rankings, scoreFaithfulness, judge } from "./harness.js";
import { citationOverlap, r3 } from "./metrics.js";
import { DATA } from "./data.js";
import { OFFLINE } from "./offline.js";
import page from "./page.html";

export default {
  async fetch(req, env) {
    const url = new URL(req.url), p = url.pathname;
    if (p === "/api/data") return json({ chunks: DATA.chunks, queries: DATA.queries.map(({ id, q, relevant }) => ({ id, q, relevant })) });
    if (p === "/api/eval") { const { rows, ...f } = scoreFaithfulness(); return json({ live_bm25: scoreRetrieval(bm25Rankings()), citation_overlap: f, offline: OFFLINE }); }
    if (req.method === "POST" && p === "/api/score") {
      const { answer, cites } = await body(req);
      if (!cap(answer) || !Array.isArray(cites) || cites.length > 5) return json({ error: "answer (<=500 chars) and cites (chunk ids) required" }, 400);
      const src = cites.map((id) => DATA.chunks.find((c) => c.id === id)?.text).filter(Boolean);
      if (!src.length) return json({ error: "unknown chunk ids" }, 400);
      const out = { overlap: r3(citationOverlap(answer, src)) };
      if (url.searchParams.get("judge") === "1") {
        if (await limited(req, env)) return json({ error: "Rate limit: try again in a minute." }, 429);
        try { out.judge = await judge(env, answer, src, ask); } catch (e) { out.judge_error = e.message; }
      }
      return json(out);
    }
    return new Response(page, { headers: { "content-type": "text/html;charset=utf-8" } });
  },
};
