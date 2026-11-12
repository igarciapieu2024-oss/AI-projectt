"""Diagnostic only: runs the real engine on a few questions and shows which ones get a cited answer. Does not change anything."""
import time

from src.config import load_config
from src.generation.answer import answer

cfg = load_config("configs/final.yaml")
QUESTIONS = [
    "What is the final exam worth in Management Control?",
    "How much is class participation in Operating Systems?",
    "Is there a final exam in Macroeconomics?",
    "Who is the professor of Time Series?",
    "What is the minimum attendance in Strategies?",
    "Which courses have the highest final exam weight?",
    "Explain the IS-LM model",
    "What is backpropagation?",
]

ok = 0
for q in QUESTIONS:
    t = time.perf_counter()
    r = answer(q, cfg=cfg)
    seconds = time.perf_counter() - t
    ok += not r.refused
    status = "REFUSED" if r.refused else "OK     "
    print(f"\n{status} {seconds:5.1f}s  route={r.route.label}  {q}")
    print("   ", r.answer_text.replace("\n", " ")[:300])
print(f"\nAnswered with valid citations: {ok}/{len(QUESTIONS)}")
