//! Cloudflare Worker entry (wasm32 only). Same routes and JSON as the old src/index.js.
use crate::harness::{handle, Ai};
use crate::page::PAGE;
use serde_json::{json, Value};
use wasm_bindgen::JsCast;
use worker::*;

struct CfAi(worker::Ai);
impl Ai for CfAi {
    async fn run(&self, model: &str, input: Value) -> std::result::Result<Value, String> {
        self.0.run::<Value, Value>(model, input).await.map_err(|e| e.to_string())
    }
}

async fn rate_ok(env: &Env, ip: &str) -> bool {
    let rl = match js_sys::Reflect::get(env.as_ref(), &"RATE_LIMITER".into()) {
        Ok(v) if !v.is_undefined() && !v.is_null() => v,
        _ => return true,
    };
    let opts = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&opts, &"key".into(), &ip.into());
    let Ok(f) = js_sys::Reflect::get(&rl, &"limit".into()).and_then(|f| f.dyn_into::<js_sys::Function>().map_err(|e| e)) else { return true };
    let Ok(p) = f.call1(&rl, &opts) else { return true };
    match wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(p)).await {
        Ok(r) => js_sys::Reflect::get(&r, &"success".into()).ok().and_then(|v| v.as_bool()).unwrap_or(true),
        Err(_) => true,
    }
}

#[event(fetch)]
async fn fetch(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let url = req.url()?;
    let path = url.path().to_string();
    let method = req.method();
    let judge_flag = url.query_pairs().any(|(k, v)| k == "judge" && v == "1");
    let body: Value = if method == Method::Post { req.json().await.unwrap_or(json!({})) } else { json!({}) };
    // The old Worker rate-limited only valid judge requests, after validation and before the model call.
    let mut judge = false;
    if method == Method::Post && path == "/api/score" && judge_flag {
        let (st, _) = handle(&CfAi(env.ai("AI")?), "POST", "/api/score", &body, false).await;
        if st == 200 {
            let ip = req.headers().get("cf-connecting-ip")?.unwrap_or_else(|| "anon".into());
            if !rate_ok(&env, &ip).await {
                return Ok(Response::from_json(&json!({"error":"Rate limit: try again in a minute."}))?.with_status(429));
            }
            judge = true;
        }
    }
    let (status, v) = handle(&CfAi(env.ai("AI")?), method.as_ref(), &path, &body, judge).await;
    if status == 0 {
        let h = Headers::new();
        h.set("content-type", "text/html;charset=utf-8")?;
        return Ok(Response::ok(PAGE)?.with_headers(h));
    }
    Ok(Response::from_json(&v)?.with_status(status))
}
