"""Quick retrieval check: does the search find the right source? (not the official evaluation, only to compare changes)

    python bench_retrieval.py --config configs/final.yaml
    python bench_retrieval.py --config configs/bge_small.yaml

44 questions asked WITHOUT a course filter (the hardest case), in 3 groups:
  main (28): admin + content over the 7 courses, the same set used for all earlier comparisons
  images (8): the answer is only inside a slide's image (diagram, chart, table), not in the slide text
  spanish (8): main questions asked in Spanish
The right sources: syllabus sections by their heading, slides by a keyword, or fixed slide ids ("ids:").
Shows hit@1, hit@5 and MRR after fusion and after the reranker. Does not change anything.
"""
import argparse
import json
import re
import time

from src.config import PATHS, load_config
from src.generation.answer import _build_hits
from src.generation.rewrite import translate_to_english_query
from src.retrieval import rerank
from src.utils.io import read_jsonl

# (question, course, "section:<SECTION_NAME>" or "slide:<regex in title/text>")
CASES = [
    ("What is the final exam worth in Management Control?", "mgmt_control", "section:EVALUATION"),
    ("What is the final exam worth in Operating Systems?", "os", "section:EVALUATION"),
    ("How is Organizational Behaviour evaluated?", "ob", "section:EVALUATION"),
    ("What is the re-sit policy in Time Series?", "time_series", "section:RESIT"),
    ("Who teaches Strategies?", "strategies", "section:PROFESSOR"),
    ("What happens if I miss more than 20% of the Macroeconomics classes?", "macro", "section:RESIT"),
    ("Can I use ChatGPT in Machine Learning Foundations?", "ai_ml", "section:AI_POLICY"),
    ("What book do we use in Management Control?", "mgmt_control", "section:BIBLIOGRAPHY"),
    ("What is overfitting?", "ai_ml", r"slide:overfitting"),
    ("How does a support vector classifier work?", "ai_ml", r"slide:support vector"),
    ("Explain the k-means algorithm", "ai_ml", r"slide:k-means"),
    ("What is backpropagation?", "ai_ml", r"slide:backpropagation"),
    ("Explain the IS-LM model", "macro", r"slide:IS-LM"),
    ("What does the Phillips curve show?", "macro", r"slide:Phillips"),
    ("How is the unemployment rate calculated?", "macro", r"slide:unemployment rate"),
    ("What is the aggregate demand curve?", "macro", r"slide:aggregate demand"),
    ("What is transfer pricing?", "mgmt_control", r"slide:transfer pric"),
    ("What is a flexible budget?", "mgmt_control", r"slide:flexible budget"),
    ("What types of responsibility centers exist?", "mgmt_control", r"slide:responsibility center"),
    ("How do you calculate ROI?", "mgmt_control", r"slide:\bROI\b"),
    ("What is a page fault?", "os", r"slide:page fault"),
    ("How does virtual memory work?", "os", r"slide:virtual memory"),
    ("What are the Big Five personality traits?", "ob", r"slide:big five"),
    ("What are the early theories of motivation?", "ob", r"slide:motivation"),
    ("What are Porter's five forces?", "strategies", r"slide:five forces"),
    ("What is a blue ocean strategy?", "strategies", r"slide:blue ocean"),
    ("What is a Nash equilibrium?", "strategies", r"slide:nash equilibrium"),
    ("What is vertical integration?", "strategies", r"slide:vertical integration"),
]
IMAGE_CASES = [
    ("What is the Dupont formula for ROE?", "strategies", "ids:strategies_s02_p040"),
    ("What are the steps of the expectancy theory of motivation?", "ob", "ids:ob_s05_p036"),
    ("Which parts are connected to the CPU in a modern desktop PC architecture?", "os", "ids:os_s02_p003,os_s02_p004"),
    ("What is inside the Intel Arrow Lake processor package, with P-cores and E-cores?", "os", "ids:os_s02_p005"),
    ("Which functions report to the CFO, like the controller and treasury?", "mgmt_control", "ids:mgmt_control_s03_p024"),
    ("Which country grew the most between Spain, France, Italy and Germany from 2021 to 2025?", "macro", "ids:macro_s02-03-04_p033"),
    ("How does vision drive strategy and tactics according to Jim Collins?", "strategies", "ids:strategies_s01_p005,strategies_s01_p006"),
    ("What are the four VRIO questions in the Arrogant Brewery case?", "strategies", "ids:strategies_s11_p006,strategies_s11_p008"),
]
SPANISH_CASES = [
    ("¿Cuánto vale el examen final de Management Control?", "mgmt_control", "section:EVALUATION"),
    ("¿Quién da la asignatura de Strategies?", "strategies", "section:PROFESSOR"),
    ("¿Cuál es la asistencia mínima en Macroeconomics?", "macro", "section:RESIT"),
    ("¿Qué es el sobreajuste en machine learning?", "ai_ml", r"slide:overfitting"),
    ("Explica el modelo IS-LM", "macro", r"slide:IS-LM"),
    ("¿Qué son los precios de transferencia?", "mgmt_control", r"slide:transfer pric"),
    ("¿Qué es un fallo de página en un sistema operativo?", "os", r"slide:page fault"),
    ("¿Qué es el equilibrio de Nash?", "strategies", r"slide:nash equilibrium"),
]
GROUPS = [("main", CASES), ("images", IMAGE_CASES), ("spanish", SPANISH_CASES)]


def gold_ids(course: str, target: str, sections: list, slides: list) -> set[str]:
    kind, value = target.split(":", 1)
    if kind == "ids":
        return set(value.split(","))
    if kind == "section":
        return {s["section_id"] for s in sections if s["course_code"] == course and s["section_name"] == value}
    pattern = re.compile(value, re.IGNORECASE)
    return {s["slide_id"] for s in slides
            if s["course_code"] == course and pattern.search(f"{s.get('title', '')} {s.get('text', '')}")}


def rank_of_first_hit(hits, gold: set[str]) -> int | None:
    for i, hit in enumerate(hits, 1):
        if gold & set(hit.chunk.covers):
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--group", choices=["all", *(name for name, _ in GROUPS)], default="all")
    parser.add_argument("--k-dense", type=int)
    parser.add_argument("--k-bm25", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.k_dense is not None:
        cfg.retrieval["k_dense"] = args.k_dense
    if args.k_bm25 is not None:
        cfg.retrieval["k_bm25"] = args.k_bm25
    for key in ("k_dense", "k_bm25"):
        value = cfg.retrieval.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            parser.error(f"{key.replace('_', '-')} must be a positive integer")
    sections = read_jsonl(PATHS["syllabus_sections"])
    slides = read_jsonl(PATHS["slides"])
    model = rerank._model(cfg.retrieval["reranker_model"])
    print(
        f"config={cfg.name}  kb_sig={cfg.kb_sig}  embed_model={cfg.kb['embed_model']}  "
        f"k_dense={cfg.retrieval['k_dense']}  k_bm25={cfg.retrieval['k_bm25']}  "
        f"translate_query={cfg.generation.get('translate_query', False)}\n"
    )

    results = {}
    started = time.perf_counter()
    for group, cases in GROUPS:
        if args.group != "all" and group != args.group:
            continue
        fused_ranks, reranked_ranks = [], []
        print(f"--- {group}")
        for question, course, target in cases:
            gold = gold_ids(course, target, sections, slides)
            query = question
            if cfg.generation.get("translate_query", False):
                query = translate_to_english_query(question, cfg)
            fused = _build_hits(query, cfg, None)  # no course filter on purpose
            scores = model.predict(
                [(query, hit.chunk.embed_text) for hit in fused],
                show_progress_bar=False,
            )
            reranked = [hit for _, hit in sorted(zip(scores, fused), key=lambda item: -item[0])]
            fused_rank = rank_of_first_hit(fused, gold)
            reranked_rank = rank_of_first_hit(reranked, gold)
            fused_ranks.append(fused_rank)
            reranked_ranks.append(reranked_rank)
            print(
                f"fusion #{fused_rank if fused_rank else '-':>2}  "
                f"rerank #{reranked_rank if reranked_rank else '-':>2}  "
                f"({len(gold):2} gold)  {question}"
            )
            if query != question:
                print(f"  search query: {query}")
        results[group] = (fused_ranks, reranked_ranks)

    def summary(name, ranks):
        n = len(ranks)
        hit1 = sum(1 for r in ranks if r == 1) / n
        hit5 = sum(1 for r in ranks if r and r <= 5) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{name:15} hit@1 {hit1:.2f}   hit@5 {hit5:.2f}   MRR {mrr:.2f}")

    count = sum(len(cases) for group, cases in GROUPS if args.group in {"all", group})
    print(f"\n{count} questions, {time.perf_counter() - started:.0f} s")
    for group, (fused_ranks, reranked_ranks) in results.items():
        print(f"[{group}]")
        summary("after fusion", fused_ranks)
        summary("after reranker", reranked_ranks)
    print(json.dumps({"config": cfg.name, **{g: r for g, (_, r) in results.items()}}))


if __name__ == "__main__":
    main()
