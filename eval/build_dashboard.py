"""Turns results/eval.json into results/results.json and renders docs/ (ES) and docs/en/ (EN)."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHY = json.loads((ROOT / "eval" / "review.json").read_text(encoding="utf-8")) if (ROOT / "eval" / "review.json").exists() else {}


BOILER = re.compile(r"(PLEASE NOTE.*?All Rights Reserved\.|Printed by .*?(AM|PM)\.|Downloaded from \S+.*?\d{4}|©[^.]*\.)", re.S | re.I)


def snippet(text, keys, width=220):
    """Short excerpt around the reference term (the guidelines are copyrighted: never publish whole chunks)."""
    t = re.sub(r"\s+", " ", BOILER.sub(" ", text.replace("<br>", " "))).strip()
    low = t.lower()
    for k in keys:
        i = low.find(k.lower())
        if i >= 0:
            a, b = max(0, i - width // 2), min(len(t), i + len(k) + width // 2)
            return ("…" if a else "") + t[a:b] + ("…" if b < len(t) else "")
    return t[:width] + ("…" if len(t) > width else "")


def outcome(a):
    if a.get("out_of_scope"):
        return "abstained_ok" if a["abstained"] else "hallucinated"
    if a["abstained"]:
        return "abstained"
    if a["correct"]:
        return "correct" if a["grounded"] else "unsupported"
    return "wrong"


if __name__ == "__main__":
    ev = json.loads((ROOT / "results" / "eval.json").read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "eval" / "corpus.json").read_text(encoding="utf-8"))
    short = {c["file"]: c["short"] for c in corpus}
    answers = []
    for a in ev["answers"]:
        o = outcome(a)
        keys = (a["gold"]["keys"] + [a["gold"]["evidence"]]) if a.get("gold") else []
        cites = [{"n": i + 1, "cite": f"{short.get(s['file'], s['file'])} · p. {s['page']}", "text": snippet(s["text"], keys)}
                 for i, s in enumerate(a["sources"])]
        answers.append({"id": a["id"], "q": a["q"], "answer": a["answer"], "outcome": o, "s": a["s"],
                        "gold": {"doc": short[a["gold"]["doc"]], "answer_key": a["gold"]["keys"][0],
                                 "evidence": a["gold"]["evidence"]} if a.get("gold") else None,
                        "why": WHY.get(str(a["id"])), "citations": cites})
    cnt = lambda k: sum(x["outcome"] == k for x in answers if x["gold"])
    s = ev["summary"]
    data = {"corpus": corpus, "retrieval": {k: {m: v for m, v in r.items() if m != "per_question"} for k, r in ev["retrieval"].items()},
            "summary": {"n": s["n"], "correct": cnt("correct"), "unsupported": cnt("unsupported"), "wrong": cnt("wrong"),
                        "abstained": cnt("abstained"), "oos_abstained": s["oos_abstained"], "n_oos": s["n_oos"],
                        "s_median": s["s_median"], "llm": ev["llm"], "embeddings": ev["embeddings"]},
            "answers": answers}
    notes_f = ROOT / "eval" / "notes.json"
    notes = json.loads(notes_f.read_text(encoding="utf-8")) if notes_f.exists() else {}
    for lang, name in (("es", "results.json"), ("en", "results.en.json")):
        data["notes"] = notes.get(lang)
        (ROOT / "results" / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "shared" / "build.py"), str(ROOT)], check=True)
