"""Evaluates Med-RAG on 15 hand-written questions (answer in the guidelines) + 3 out-of-scope questions.

Retrieval: does the passage holding the evidence appear in the top 6, with dense-only vs hybrid search?
Answers:   hybrid retrieval + local LLM; checked for the reference figure and for a cited page that holds it.
Writes results/eval.json.   python eval/run_eval.py
`--only 9,10,11` recomputes retrieval for every question but re-answers only those ids, keeping the other stored
answers (useful after replacing a few questions; same corpus and settings).
"""
import json
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import EMBED_MODEL, LLM_MODEL, setup_models  # noqa: E402
from src.engine import NOT_FOUND, MedRAG  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9≥/]+", "", s.lower())


def load(name):
    return [json.loads(l) for l in (ROOT / "eval" / name).open(encoding="utf-8")]


def node_info(n):
    return {"file": n.metadata.get("file_name"), "page": n.metadata.get("page_label"),
            "score": round(float(n.score or 0), 3), "text": n.get_content()[:1500]}


if __name__ == "__main__":
    setup_models()
    rag = MedRAG()
    qs, oos = load("questions.jsonl"), load("out_of_scope.jsonl")
    only = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")} if "--only" in sys.argv else None
    prev = {}
    if only:
        old = json.loads((ROOT / "results" / "eval.json").read_text(encoding="utf-8"))
        prev = {a["id"]: a for a in old["answers"]}
    out = {"llm": LLM_MODEL, "embeddings": EMBED_MODEL, "machine": platform.processor() or platform.machine(),
           "retrieval": {}, "answers": []}

    for mode in ("default", "hybrid"):          # default = dense vectors only
        r, rows, t0 = rag.retriever(mode, k=6), [], time.time()
        for q in qs:
            nodes = r.retrieve(q["q"])
            hit = [i for i, n in enumerate(nodes) if n.metadata.get("file_name") == q["doc"]
                   and norm(q["evidence"]) in norm(n.get_content())]
            rows.append({"id": q["id"], "rank": hit[0] + 1 if hit else None,
                         "doc_in_top6": any(n.metadata.get("file_name") == q["doc"] for n in nodes)})
        n = len(rows)
        out["retrieval"][mode] = {"hit@1": round(sum(x["rank"] == 1 for x in rows) / n, 3),
                                  "hit@6": round(sum(x["rank"] is not None for x in rows) / n, 3),
                                  "mrr": round(sum(1 / x["rank"] for x in rows if x["rank"]) / n, 3),
                                  "doc@6": round(sum(x["doc_in_top6"] for x in rows) / n, 3),
                                  "ms_per_query": round(1000 * (time.time() - t0) / n), "per_question": rows}
        print(mode, {k: v for k, v in out["retrieval"][mode].items() if k != "per_question"}, flush=True)

    engine = rag.get_query_engine("hybrid", k=6, streaming=False)
    for q in qs + oos:
        if only and q["id"] not in only and q["id"] in prev:
            out["answers"].append(prev[q["id"]])
            continue
        t0 = time.time()
        resp = engine.query(q["q"])
        text = str(resp).strip()
        srcs = [node_info(n) for n in resp.source_nodes]
        row = {"id": q["id"], "q": q["q"], "answer": text, "sources": srcs, "s": round(time.time() - t0, 1),
               "abstained": norm(NOT_FOUND) in norm(text) or "idontknow" in norm(text)}
        if "answer_keys" in q:
            fig = any(norm(k) in norm(text) for k in q["answer_keys"])
            grounded = any(n.metadata.get("file_name") == q["doc"] and norm(q["evidence"]) in norm(n.get_content())
                           for n in resp.source_nodes)   # full chunk text, not the 1500-char excerpt
            row.update(gold={"doc": q["doc"], "keys": q["answer_keys"], "evidence": q["evidence"]},
                       correct=fig and not row["abstained"], grounded=grounded)
        else:
            row["out_of_scope"] = True
        out["answers"].append(row)
        print(f"{q['id']:>3} {row['s']:>6}s  {text[:110]!r}", flush=True)

    ins = [a for a in out["answers"] if "gold" in a]
    out["summary"] = {"n": len(ins), "correct": sum(a["correct"] for a in ins),
                      "correct_and_grounded": sum(a["correct"] and a["grounded"] for a in ins),
                      "abstained_in_scope": sum(a["abstained"] for a in ins),
                      "oos_abstained": sum(a["abstained"] for a in out["answers"] if a.get("out_of_scope")),
                      "n_oos": len(oos), "s_median": sorted(a["s"] for a in out["answers"])[len(out["answers"]) // 2]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
