//! BM25 lexical retriever (k1=1.5, b=0.75). No model calls.
use crate::metrics::tokens;
use std::collections::{HashMap, HashSet};

pub struct Hit {
    pub id: String,
    pub score: f64,
}

pub fn bm25_search(chunks: &[(String, String)], query: &str) -> Vec<Hit> {
    let (k1, b) = (1.5f64, 0.75f64);
    let docs: Vec<Vec<String>> = chunks.iter().map(|c| tokens(&c.1)).collect();
    let n_docs = docs.len() as f64;
    let avg = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / n_docs;
    let mut df: HashMap<&str, f64> = HashMap::new();
    for d in &docs {
        for t in d.iter().map(|s| s.as_str()).collect::<HashSet<_>>() {
            *df.entry(t).or_insert(0.0) += 1.0;
        }
    }
    let q = tokens(query);
    let mut out: Vec<Hit> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut s = 0.0;
            for t in &q {
                let f = d.iter().filter(|x| *x == t).count() as f64;
                if f == 0.0 {
                    continue;
                }
                let n = df[t.as_str()];
                s += (1.0 + (n_docs - n + 0.5) / (n + 0.5)).ln() * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * d.len() as f64 / avg));
            }
            Hit { id: chunks[i].0.clone(), score: s }
        })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    out
}
