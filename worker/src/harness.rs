//! The harness: score any retriever's rankings and any answer set against the labelled data.
use crate::bm25::bm25_search;
use crate::metrics::*;
use regex::Regex;
use serde_json::{json, Map, Value};

pub const MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
pub const JUDGE_PROMPT: &str = "You are a strict fact checker. Reply with exactly SUPPORTED if every claim in the ANSWER is stated in the SOURCES, otherwise reply exactly UNSUPPORTED.";
const DATA_JSON: &str = include_str!("../data/data.json");
const OFFLINE_JSON: &str = include_str!("../data/offline.json");

#[allow(async_fn_in_trait)]
pub trait Ai {
    async fn run(&self, model: &str, input: Value) -> Result<Value, String>;
}

pub fn data() -> Value {
    serde_json::from_str(DATA_JSON).expect("data.json")
}
pub fn offline() -> Value {
    serde_json::from_str(OFFLINE_JSON).expect("offline.json")
}
fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}
fn chunks(d: &Value) -> Vec<(String, String)> {
    d["chunks"].as_array().unwrap().iter().map(|c| (c["id"].as_str().unwrap().to_string(), c["text"].as_str().unwrap().to_string())).collect()
}

pub fn score_retrieval(rankings: &Map<String, Value>, queries: &[Value]) -> Value {
    let mut out = Map::new();
    out.insert("n".into(), json!(queries.len()));
    for k in [1usize, 3, 5] {
        let p: Vec<f64> = queries.iter().map(|q| precision_at_k(&strs(&rankings[q["id"].as_str().unwrap()]), &strs(&q["relevant"]), k)).collect();
        let r: Vec<f64> = queries.iter().map(|q| recall_at_k(&strs(&rankings[q["id"].as_str().unwrap()]), &strs(&q["relevant"]), k)).collect();
        out.insert(format!("precision@{k}"), json!(r3(mean(&p))));
        out.insert(format!("recall@{k}"), json!(r3(mean(&r))));
    }
    let rr: Vec<f64> = queries.iter().map(|q| reciprocal_rank(&strs(&rankings[q["id"].as_str().unwrap()]), &strs(&q["relevant"]))).collect();
    out.insert("mrr".into(), json!(r3(mean(&rr))));
    Value::Object(out)
}

pub fn bm25_rankings(d: &Value) -> Map<String, Value> {
    let ch = chunks(d);
    d["queries"].as_array().unwrap().iter().map(|q| (q["id"].as_str().unwrap().to_string(), json!(bm25_search(&ch, q["q"].as_str().unwrap()).into_iter().map(|h| h.id).collect::<Vec<_>>()))).collect()
}

/// Returns the summary plus the per-row details (callers drop `rows` for the API).
pub fn score_faithfulness(d: &Value, threshold: f64) -> Value {
    let text: std::collections::HashMap<String, String> = chunks(d).into_iter().collect();
    let mut rows: Vec<Value> = vec![];
    for a in d["answers"].as_array().unwrap() {
        let cited: Vec<String> = strs(&a["cites"]).iter().map(|i| text.get(i).cloned().unwrap_or_default()).collect();
        let mut r = a.clone();
        r["overlap"] = json!(r3(citation_overlap(a["answer"].as_str().unwrap(), &cited)));
        rows.push(r);
    }
    let pred: Vec<i64> = rows.iter().map(|r| if r["overlap"].as_f64().unwrap() >= threshold { 1 } else { 0 }).collect();
    let lab: Vec<i64> = rows.iter().map(|r| r["label"].as_i64().unwrap()).collect();
    let cnt = |f: &dyn Fn(i64, i64) -> bool| pred.iter().zip(&lab).filter(|(p, l)| f(**p, **l)).count() as f64;
    let (tp, fp, fn_) = (cnt(&|p, l| p == 1 && l == 1), cnt(&|p, l| p == 1 && l == 0), cnt(&|p, l| p == 0 && l == 1));
    let ov = |want: i64| -> Vec<f64> { rows.iter().filter(|r| r["label"].as_i64().unwrap() == want).map(|r| r["overlap"].as_f64().unwrap()).collect() };
    let nz = |x: f64| if x == 0.0 { 1.0 } else { x };
    json!({"n": rows.len(), "threshold": threshold, "auc": r3(auc(&rows.iter().map(|r| r["overlap"].as_f64().unwrap()).collect::<Vec<_>>(), &lab)),
        "accuracy": r3(cnt(&|p, l| p == l) / rows.len() as f64), "precision": r3(tp / nz(tp + fp)), "recall": r3(tp / nz(tp + fn_)),
        "mean_overlap_faithful": r3(mean(&ov(1))), "mean_overlap_unfaithful": r3(mean(&ov(0))), "rows": rows})
}

/// LLM-as-judge faithfulness. Returns 1 (supported) / 0 (not supported).
pub async fn judge<A: Ai>(ai: &A, answer: &str, sources: &[String]) -> Result<i64, String> {
    let r = ai.run(MODEL, json!({"messages":[{"role":"system","content":JUDGE_PROMPT},{"role":"user","content":format!("SOURCES:\n{}\n\nANSWER: {}", sources.join("\n"), answer)}],"max_tokens":5})).await?;
    let t = match r.get("response") { Some(Value::String(s)) => s.clone(), Some(Value::Null) | None => r.to_string(), Some(v) => v.to_string() };
    Ok(if Regex::new(r"(?i)^[^A-Za-z0-9_]*supported").unwrap().is_match(&t) { 1 } else { 0 })
}

pub fn cap(s: Option<&str>, n: usize) -> bool {
    matches!(s, Some(s) if !s.trim().is_empty() && s.chars().count() <= n)
}

/// Pure router: (status, json); status 0 = serve the page. Rate limiting for `judge_flag` is applied by the Worker.
pub async fn handle<A: Ai>(ai: &A, method: &str, path: &str, body: &Value, judge_flag: bool) -> (u16, Value) {
    let d = data();
    if path == "/api/data" {
        let qs: Vec<Value> = d["queries"].as_array().unwrap().iter().map(|q| json!({"id": q["id"], "q": q["q"], "relevant": q["relevant"]})).collect();
        return (200, json!({"chunks": d["chunks"], "queries": qs}));
    }
    if path == "/api/eval" {
        let mut f = score_faithfulness(&d, 0.5);
        f.as_object_mut().unwrap().remove("rows");
        let queries = d["queries"].as_array().unwrap().clone();
        return (200, json!({"live_bm25": score_retrieval(&bm25_rankings(&d), &queries), "citation_overlap": f, "offline": offline()}));
    }
    if method == "POST" && path == "/api/score" {
        let (answer, cites) = (body.get("answer").and_then(|v| v.as_str()), body.get("cites").and_then(|v| v.as_array()));
        if !cap(answer, 500) || cites.map_or(true, |c| c.len() > 5) {
            return (400, json!({"error": "answer (<=500 chars) and cites (chunk ids) required"}));
        }
        let by_id: std::collections::HashMap<String, String> = chunks(&d).into_iter().collect();
        let src: Vec<String> = cites.unwrap().iter().filter_map(|i| i.as_str()).filter_map(|i| by_id.get(i).cloned()).filter(|t| !t.is_empty()).collect();
        if src.is_empty() {
            return (400, json!({"error": "unknown chunk ids"}));
        }
        let mut out = json!({"overlap": r3(citation_overlap(answer.unwrap(), &src))});
        if judge_flag {
            match judge(ai, answer.unwrap(), &src).await {
                Ok(j) => out["judge"] = json!(j),
                Err(e) => out["judge_error"] = json!(e),
            }
        }
        return (200, out);
    }
    (0, Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    struct Fixed(&'static str);
    impl Ai for Fixed {
        async fn run(&self, _m: &str, _i: Value) -> Result<Value, String> { Ok(json!({"response": self.0})) }
    }
    struct Boom;
    impl Ai for Boom {
        async fn run(&self, _m: &str, _i: Value) -> Result<Value, String> { Err("quota".into()) }
    }
    fn v(a: &[&str]) -> Vec<String> { a.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn metric_basics() {
        assert_eq!(precision_at_k(&v(&["a", "b", "c"]), &v(&["b", "z"]), 2), 0.5);
        assert_eq!(recall_at_k(&v(&["a", "b", "c"]), &v(&["b", "z"]), 3), 0.5);
        assert_eq!(recall_at_k(&v(&["a"]), &v(&[]), 1), 0.0);
        assert_eq!(reciprocal_rank(&v(&["x", "y", "a"]), &v(&["a"])), 1.0 / 3.0);
        assert_eq!(reciprocal_rank(&v(&["x"]), &v(&["a"])), 0.0);
        assert_eq!(auc(&[0.9, 0.8, 0.1], &[1, 1, 0]), 1.0);
        assert_eq!(auc(&[0.1, 0.9], &[1, 0]), 0.0);
        assert_eq!(auc(&[0.5, 0.5], &[1, 0]), 0.5);
        assert_eq!(tokens("The agents are routing tools"), ["agent", "rout", "tool"]);
        assert_eq!(citation_overlap("vectors rank passages", &v(&["Vectors rank the passages."])), 1.0);
        assert_eq!(citation_overlap("bananas fly", &v(&["Vectors rank passages."])), 0.0);
        assert_eq!(citation_overlap("", &v(&["x"])), 0.0);
    }
    #[test]
    fn bm25_labelled_set_and_oracle() {
        let c = vec![("a".to_string(), "cats purr softly".to_string()), ("b".to_string(), "JSON schema describes tools".to_string())];
        assert_eq!(bm25_search(&c, "which schema describes tools")[0].id, "b");
        let d = data();
        assert_eq!((d["chunks"].as_array().unwrap().len(), d["queries"].as_array().unwrap().len(), d["answers"].as_array().unwrap().len()), (22, 20, 40));
        let qs = d["queries"].as_array().unwrap().clone();
        let oracle: Map<String, Value> = qs.iter().map(|q| { let mut r = strs(&q["relevant"]); r.push("zz".into()); (q["id"].as_str().unwrap().to_string(), json!(r)) }).collect();
        let m = score_retrieval(&oracle, &qs);
        assert_eq!((m["mrr"].as_f64(), m["recall@3"].as_f64(), m["precision@1"].as_f64()), (Some(1.0), Some(1.0), Some(1.0)));
        assert_eq!(score_retrieval(&bm25_rankings(&d), &qs), score_retrieval(&bm25_rankings(&d), &qs));
        let f = score_faithfulness(&d, 0.5);
        assert!(f["auc"].as_f64().unwrap() > 0.5 && f["n"] == 40);
    }
    #[test]
    fn judge_and_http() {
        assert_eq!(block_on(judge(&Fixed("SUPPORTED"), "a", &v(&["s"]))).unwrap(), 1);
        assert_eq!(block_on(judge(&Fixed("UNSUPPORTED."), "a", &v(&["s"]))).unwrap(), 0);
        let go = |m: &str, p: &str, b: Value| block_on(handle(&Fixed("x"), m, p, &b, false));
        let (s, e) = go("GET", "/api/eval", json!({}));
        assert_eq!((s, e["live_bm25"]["n"].clone()), (200, json!(20)));
        assert!(e["offline"]["models"].get("bge-base-en-v1.5").is_some());
        assert_eq!(go("POST", "/api/score", json!({})).0, 400);
        assert_eq!(go("POST", "/api/score", json!({"answer":"hi","cites":["nope"]})).0, 400);
        assert_eq!(go("POST", "/api/score", json!({"answer":"Approvals are logged for auditing.","cites":["d9c2"]})).1["overlap"], 1.0);
        let (_, r) = block_on(handle(&Boom, "POST", "/api/score", &json!({"answer":"x y z","cites":["d1c1"]}), true));
        assert_eq!(r["judge_error"], "quota");
        assert_eq!(go("GET", "/does-not-exist", json!({})).0, 0);
    }
}
