import pytest
from agent11.metrics import precision_at_k, recall_at_k, reciprocal_rank, citation_overlap, auc, tokens
from agent11.bm25 import bm25_index
from agent11.harness import score_retrieval, score_faithfulness, bm25_rankings, judge, DATA, handle


def test_precision_recall_rr_and_edges():
    assert precision_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5 and recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5 and recall_at_k(["a"], [], 1) == 0
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3 and reciprocal_rank(["x"], ["a"]) == 0
    assert recall_at_k([], ["a"], 5) == 0 and reciprocal_rank([], ["a"]) == 0 and precision_at_k(["a"], ["a"], 1) == 1
    for v in [precision_at_k(["a", "b"], ["a", "b"], 2), recall_at_k(["a", "b"], ["a"], 2), citation_overlap("x y", ["x"]), auc([0.3, 0.7, 0.2], [0, 1, 1])]:
        assert 0 <= v <= 1


def test_auc_tokens_overlap():
    assert auc([0.9, 0.8, 0.1], [1, 1, 0]) == 1 and auc([0.1, 0.9], [1, 0]) == 0 and auc([0.5, 0.5], [1, 0]) == 0.5
    assert tokens("The agents are routing tools") == ["agent", "rout", "tool"]
    assert citation_overlap("vectors rank passages", ["Vectors rank the passages."]) == 1 and citation_overlap("bananas fly", ["Vectors rank passages."]) == 0 and citation_overlap("", ["x"]) == 0


def test_bm25():
    s = bm25_index([{"id": "a", "text": "cats purr softly"}, {"id": "b", "text": "JSON schema describes tools"}])
    assert s("which schema describes tools")[0]["id"] == "b"
    r = bm25_index([{"id": "a", "text": "cats purr"}])("zebra quantum")
    assert not r or r[0]["score"] <= 0


def test_labelled_set_and_oracle():
    ids = {c["id"] for c in DATA["chunks"]}
    assert len(DATA["chunks"]) == 22 and len(DATA["queries"]) == 20 and len(DATA["answers"]) == 40
    assert all(q["relevant"] and all(r in ids for r in q["relevant"]) for q in DATA["queries"]) and sum(1 for a in DATA["answers"] if a["label"] == 1) == 20
    m = score_retrieval({q["id"]: [*q["relevant"], "zz"] for q in DATA["queries"]})
    assert m["mrr"] == 1 and m["recall@3"] == 1 and m["precision@1"] == 1


def test_deterministic_and_faithfulness():
    assert score_retrieval(bm25_rankings()) == score_retrieval(bm25_rankings())
    f = score_faithfulness()
    assert f["auc"] > 0.5 and f["n"] == 40


def test_judge_parse():
    assert judge(lambda m, i: {"response": "SUPPORTED"}, "a", ["s"]) == 1 and judge(lambda m, i: {"response": "UNSUPPORTED."}, "a", ["s"]) == 0


def test_http_router():
    s, e = handle(None, "GET", "/api/eval")
    assert s == 200 and e["live_bm25"]["n"] == 20 and "bge-base-en-v1.5" in e["offline"]["models"]
    assert handle(None, "POST", "/api/score", {})[0] == 400
    assert handle(None, "POST", "/api/score", {"answer": "hi", "cites": ["nope"]})[0] == 400
    assert handle(None, "POST", "/api/score", {"answer": "Approvals are logged for auditing.", "cites": ["d9c2"]})[1]["overlap"] == 1
    def boom(m, i): raise RuntimeError("quota")
    assert handle(boom, "POST", "/api/score", {"answer": "x y z", "cites": ["d1c1"]}, judge_flag=True)[1]["judge_error"] == "quota"
    assert handle(None, "GET", "/does-not-exist")[0] == 0
