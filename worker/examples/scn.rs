// Parity helper: same checks as the legacy JS harness; prints JSON.
use agent_11_rag_eval_harness::{bm25::bm25_search, harness::*, metrics::*};
use serde_json::{json, Value};
fn main() {
    let d = data();
    let qs = d["queries"].as_array().unwrap().clone();
    let ch: Vec<(String, String)> = d["chunks"].as_array().unwrap().iter().map(|c| (c["id"].as_str().unwrap().into(), c["text"].as_str().unwrap().into())).collect();
    let tk = ["The agents are routing tools", "passes boxes ies classes", "Hello, WORLD! 123 abc-def_ghi", "é à ñ uber 2026s", "ing es s", "running", "thinkings", ""];
    let rank = bm25_rankings(&d);
    let scores: Vec<Value> = qs.iter().map(|q| Value::Array(bm25_search(&ch, q["q"].as_str().unwrap()).into_iter().take(5).map(|h| json!([h.id, h.score])).collect())).collect();
    let out = json!({"tok": tk.iter().map(|t| tokens(t)).collect::<Vec<_>>(), "rank": rank, "ret": score_retrieval(&rank, &qs), "faith": score_faithfulness(&d, 0.5), "faith2": score_faithfulness(&d, 0.3), "scores": scores,
        "ov": [citation_overlap("vectors rank passages", &["Vectors rank the passages.".to_string()]), citation_overlap("", &["x".to_string()])], "auc": auc(&[0.9, 0.8, 0.1], &[1, 1, 0]), "r3": [r3(0.0005), r3(0.1235), r3(2.0 / 3.0), r3(1.0005)]});
    println!("{}", serde_json::to_string(&out).unwrap());
}
