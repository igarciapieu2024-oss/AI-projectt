"""Question router: 5 classifiers (GridSearchCV, StratifiedKFold 5, macro-F1) vs the LLM -> models/router.joblib + router_meta.json. Run: python -m src.train.train_router

Owner: Francesca.

What it does
1. Reads data/labels/router_questions.csv (question, label in 4 classes).
2. Turns every question into a vector with the same embedding model as the knowledge base
   (cfg.kb["embed_model"], all-MiniLM-L6-v2).
3. Compares 5 classifiers - logistic regression, KNN, SVM, random forest, MLP. Each one is
   tuned with GridSearchCV inside (3 folds), and scored with an outer StratifiedKFold(5) on macro-F1
   (nested cross-validation, so the score is never measured on data used for tuning).
4. Compares the best one with the LLM used as a router (zero-shot, no training).
5. Refits the best classifier on all 300 questions and saves models/router.joblib
   + models/router_meta.json, and writes the result tables/figure to reports/.

Options
    python -m src.train.train_router                 # full run (LLM baseline included)
    python -m src.train.train_router --no-llm        # skip the LLM baseline (fast)
    python -m src.train.train_router --features tfidf  # TF-IDF instead of embeddings (comparison)

Use from the engine (Sergio):
    from src.train.train_router import load_router
    router = load_router()
    label, prob = router.route("What is the final exam worth in Macro?")
    # label in {course_admin, course_content, cross_course, out_of_scope}
    # if prob < cfg.routing["min_prob"] (0.6) the router is not sure -> search everywhere
"""
from __future__ import annotations

import os

# macOS: PyTorch (embeddings) + parallel scikit-learn workers can crash Python (SIGABRT).
# Everything runs in one process; 300 questions are small enough for that.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import PATHS, load_config

LABELS = ["course_admin", "course_content", "cross_course", "out_of_scope"]
SEED = 42

# The 5 classifiers and the settings GridSearchCV tries for each one.
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"C": [0.1, 1, 10, 100]},
    ),
    "knn": (
        KNeighborsClassifier(metric="cosine"),
        {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
    ),
    "svm": (
        SVC(random_state=SEED),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [200, 400], "max_depth": [None, 20]},
    ),
    "mlp": (
        MLPClassifier(max_iter=1000, random_state=SEED),
        {"hidden_layer_sizes": [(128,), (256, 64)], "alpha": [1e-4, 1e-2]},
    ),
}

LLM_PROMPT = """You are the router of a university course assistant. The student's courses are:
AI Machine Learning Foundations, Macroeconomics, Management Control, Operating Systems,
Organizational Behaviour, Strategies, Time Series.

Classify the student's question into exactly one category:
- course_admin: rules and logistics of ONE course (grading, exam weights, attendance, retakes,
  professor, contact, office hours, bibliography, policies) - answered by that course's syllabus.
- course_content: the subject matter taught in the courses (concepts, models, methods, cases)
  - answered by the session slides.
- cross_course: compares or lists information ACROSS several of the student's courses
  (e.g. "which of my courses have no final exam?").
- out_of_scope: anything the course materials cannot answer (other courses or degrees,
  university services, personal grades, topics not taught, general questions).

Answer with the category name only."""


# ---------------------------------------------------------------- data and features

def load_questions() -> pd.DataFrame:
    df = pd.read_csv(PATHS["router_questions"], dtype=str, keep_default_na=False)
    missing = {"question", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"router_questions.csv is missing columns: {sorted(missing)}")
    df["question"] = df["question"].str.strip()
    df["label"] = df["label"].str.strip()
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels in router_questions.csv: {bad}")
    dupes = df["question"].str.lower().duplicated()
    if dupes.any():
        raise ValueError(f"Duplicate questions: {df.loc[dupes, 'question'].tolist()[:5]}")
    return df


class Embedder:
    """Question -> vector. Same model as the knowledge base, so it is already on every laptop."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


class TfidfFeatures:
    """Alternative representation, only used to compare against embeddings (--features tfidf)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = "tfidf"
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._fitted = False

    def fit(self, texts):
        self._vec.fit(texts)
        self._fitted = True
        return self

    def __call__(self, texts):
        return self._vec.transform(list(texts)).toarray()


# ---------------------------------------------------------------- comparison

def compare_classifiers(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Nested CV: GridSearchCV (inner 3 folds) tunes, outer StratifiedKFold(5) scores macro-F1."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    rows = []
    for name, (estimator, grid) in MODELS.items():
        search = GridSearchCV(estimator, grid, cv=inner, scoring="f1_macro", n_jobs=1)
        t0 = time.perf_counter()
        res = cross_validate(search, X, y, cv=outer, scoring=["f1_macro", "accuracy"], n_jobs=1)
        rows.append({
            "model": name,
            "macro_f1_mean": res["test_f1_macro"].mean(),
            "macro_f1_std": res["test_f1_macro"].std(),
            "accuracy_mean": res["test_accuracy"].mean(),
            "train_seconds": time.perf_counter() - t0,
        })
        print(f"  {name:<20} macro-F1 {rows[-1]['macro_f1_mean']:.3f} ± {rows[-1]['macro_f1_std']:.3f}")
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def parse_llm_label(text: str) -> str:
    text = text.strip().lower()
    for label in LABELS:
        if label in text:
            return label
    return "invalid"


def llm_baseline(questions: list[str], y: np.ndarray, model: str, limit: int | None = None) -> dict:
    """The LLM as router, zero-shot. No training, so every question is a fair test question."""
    import ollama

    if limit:
        questions, y = questions[:limit], y[:limit]
    preds, seconds = [], []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": q}],
            options={"temperature": 0},
        )
        seconds.append(time.perf_counter() - t0)
        preds.append(parse_llm_label(resp["message"]["content"]))
        if i % 25 == 0:
            print(f"  LLM: {i}/{len(questions)} questions")
    return {
        "model": f"llm:{model}",
        "macro_f1": f1_score(y, preds, labels=LABELS, average="macro"),
        "accuracy": accuracy_score(y, preds),
        "invalid_answers": int(sum(p == "invalid" for p in preds)),
        "seconds_per_question": float(np.median(seconds)),
        "n_questions": len(questions),
    }


def save_confusion_figure(cm: np.ndarray, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    short = ["admin", "content", "cross", "out of scope"]
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main entry

def train(cfg, use_llm: bool = True, features: str = "embeddings", llm_limit: int | None = None) -> dict:
    """Train, compare and save the best router; return the scores."""
    df = load_questions()
    texts, y = df["question"].tolist(), df["label"].to_numpy()
    print(f"Loaded {len(df)} router questions: {df['label'].value_counts().to_dict()}")

    if features == "tfidf":
        featurizer = TfidfFeatures().fit(texts)
    else:
        featurizer = Embedder(cfg.kb["embed_model"])
    X = featurizer(texts)
    print(f"Features: {featurizer.model_name}  shape={X.shape}")

    print("Comparing 5 classifiers (nested 5-fold CV, macro-F1)...")
    table = compare_classifiers(X, y)
    best_name = table.loc[0, "model"]
    estimator, grid = MODELS[best_name]

    # Out-of-fold predictions of the winner -> honest confusion matrix and per-class report.
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    search = GridSearchCV(estimator, grid, cv=inner, scoring="f1_macro", n_jobs=1)
    oof = cross_val_predict(search, X, y, cv=outer)
    cm = confusion_matrix(y, oof, labels=LABELS)
    per_class = classification_report(y, oof, labels=LABELS, output_dict=True, zero_division=0)

    # Final model: tune and fit on all the questions.
    search.fit(X, y)
    final = search.best_estimator_
    if not hasattr(final, "predict_proba"):
        # SVM gives scores, not probabilities: calibrate them so the engine can use min_prob.
        final = CalibratedClassifierCV(final, cv=5).fit(X, y)

    # Speed of the trained router (embedding + prediction), for the "LLM vs classifier" slide.
    t0 = time.perf_counter()
    for q in texts[:50]:
        final.predict_proba(featurizer([q]))
    clf_seconds = (time.perf_counter() - t0) / 50

    llm = None
    if use_llm:
        print(f"LLM baseline with {cfg.generation['llm_model']} (zero-shot)...")
        llm = llm_baseline(texts, y, cfg.generation["llm_model"], limit=llm_limit)
        print(f"  LLM macro-F1 {llm['macro_f1']:.3f}  ({llm['seconds_per_question']:.2f} s/question)")

    # ---- save the model and its metadata
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    if features == "embeddings":
        joblib.dump({"classifier": final, "embed_model": featurizer.model_name, "labels": LABELS},
                    PATHS["router_model"])
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_questions": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "features": featurizer.model_name,
        "best_model": best_name,
        "best_params": search.best_params_,
        "cv": "nested: GridSearchCV inner StratifiedKFold(3), outer StratifiedKFold(5), seed 42",
        "comparison": table.to_dict(orient="records"),
        "best_per_class_f1": {lab: per_class[lab]["f1-score"] for lab in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "classifier_seconds_per_question": clf_seconds,
        "llm_baseline": llm,
        "min_prob": cfg.routing.get("min_prob"),
    }
    if features == "embeddings":
        PATHS["router_meta"].write_text(json.dumps(meta, indent=2, default=str))

    # ---- result files for the docs and the slides
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    suffix = "" if features == "embeddings" else f"_{features}"
    out = table.copy()
    if llm:
        out = pd.concat([out, pd.DataFrame([{
            "model": llm["model"], "macro_f1_mean": llm["macro_f1"], "macro_f1_std": np.nan,
            "accuracy_mean": llm["accuracy"], "train_seconds": 0.0}])], ignore_index=True)
    out.round(4).to_csv(PATHS["reports"] / f"router_comparison{suffix}.csv", index=False)
    save_confusion_figure(cm, f"Router ({best_name}), out-of-fold",
                          PATHS["figures"] / f"router_confusion{suffix}.png")

    print("\nResult")
    print(out.round(3).to_string(index=False))
    print(f"\nBest: {best_name} {search.best_params_}")
    if features == "embeddings":
        print(f"Saved {PATHS['router_model'].relative_to(PATHS['root'])} and "
              f"{PATHS['router_meta'].relative_to(PATHS['root'])}")
    return meta


# ---------------------------------------------------------------- used by the engine

class Router:
    def __init__(self, bundle: dict):
        from sentence_transformers import SentenceTransformer

        self.classifier = bundle["classifier"]
        self.labels = list(self.classifier.classes_)
        self._embed = SentenceTransformer(bundle["embed_model"], device="cpu")

    def route(self, question: str) -> tuple[str, float]:
        """Return (label, probability of that label)."""
        x = self._embed.encode([question], normalize_embeddings=True, show_progress_bar=False)
        probs = self.classifier.predict_proba(x)[0]
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i])


def load_router(path=None) -> Router:
    return Router(joblib.load(path or PATHS["router_model"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM baseline")
    ap.add_argument("--llm-limit", type=int, default=None, help="only N questions for the LLM baseline")
    ap.add_argument("--features", choices=["embeddings", "tfidf"], default="embeddings")
    args = ap.parse_args()
    train(load_config(args.config), use_llm=not args.no_llm, features=args.features,
          llm_limit=args.llm_limit)
