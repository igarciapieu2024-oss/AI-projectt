"""Question router: 5 classifiers (GridSearchCV, StratifiedKFold 5, macro-F1) vs the LLM -> models/router.joblib + router_meta.json. Run: python -m src.train.train_router"""
# The line above stays as a docstring because the command line uses it as its description
# (__doc__.splitlines()[0] at the end of this file); without it the script would stop with an error.
#
# Train the question router (step 2 of the answer phase). It:
#   - reads data/labels/router_questions.csv: 300 questions labelled by hand, 75 per label;
#   - turns every question into a vector with the same embedding model as the knowledge base
#     (cfg.kb["embed_model"], all-MiniLM-L6-v2);
#   - compares 5 classifiers (logistic regression, KNN, SVM, random forest, MLP) with nested
#     cross-validation on macro-F1, so the score is never measured on data used for tuning;
#   - compares the best one with the LLM used as a router without any training;
#   - fits the best classifier on all 300 questions and saves models/router.joblib and
#     models/router_meta.json, plus reports/router_comparison.csv and a confusion-matrix figure.
# src/retrieval/router.py loads the saved model through load_router() at the end of this file.
#
# How to run it from the repo folder:
#   python -m src.train.train_router                    full run, LLM comparison included (needs Ollama)
#   python -m src.train.train_router --no-llm           skip the LLM comparison (fast)
#   python -m src.train.train_router --features tfidf   TF-IDF instead of embeddings, only to compare
from __future__ import annotations

# os lets us set the environment variables below.
import os

# On macOS, PyTorch (used for the embeddings) together with scikit-learn running in parallel can crash Python. These
# settings keep everything in one process; 300 questions are small enough for that. They have to be set before those
# libraries are imported, which is why they come before the other imports.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# argparse reads the options of the command; json and joblib save the results; time measures the speed.
import argparse
import json
import time
from datetime import datetime, timezone

# joblib saves the trained model, numpy holds the vectors and pandas the tables.
import joblib
import numpy as np
import pandas as pd
# From scikit-learn: the 5 classifiers, the calibration of the SVM, the metrics and the cross-validation tools.
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# The folders of the project and the config reader.
from src.config import PATHS, load_config

# The four labels, in a fixed order (used for the confusion matrix and the per-label scores).
LABELS = ["course_admin", "course_content", "cross_course", "out_of_scope"]
# A fixed seed for the folds and the models, so training again gives the same result.
SEED = 42

# The 5 classifiers and, for each one, the settings GridSearchCV tries.
MODELS = {
    # C controls how strongly the model is regularised (smaller C = simpler model).
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"C": [0.1, 1, 10, 100]},
    ),
    # Labels a question with the labels of its closest questions; cosine distance, like the search.
    "knn": (
        KNeighborsClassifier(metric="cosine"),
        {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
    ),
    # Tries a linear and a non-linear (rbf) boundary between the labels.
    "svm": (
        SVC(random_state=SEED),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    ),
    # Number of trees and how deep they can grow (None = no limit).
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [200, 400], "max_depth": [None, 20]},
    ),
    # A small neural network: one layer of 128 or two layers of 256 and 64; alpha is its regularisation.
    "mlp": (
        MLPClassifier(max_iter=1000, random_state=SEED),
        {"hidden_layer_sizes": [(128,), (256, 64)], "alpha": [1e-4, 1e-2]},
    ),
}

# The instruction we give to the LLM when it works as a router. It lists the student's 7 courses, describes each label
# and asks for the label name only, so we can read the answer automatically. router.py uses the same instruction when
# the classifier isn't sure, so the LLM works there as it did here.
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


# ----------------------------------------------------------------------
# Data and features

# Read the labelled questions and check them: both columns have to exist, every label has to be one of the four, and
# no question may appear twice (a repeated question could end up in both the training fold and the test fold and make
# the score look better than it is).
def load_questions() -> pd.DataFrame:
    # Read the CSV with every value as text (an empty cell stays an empty text).
    df = pd.read_csv(PATHS["router_questions"], dtype=str, keep_default_na=False)
    # The two columns we need.
    missing = {"question", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"router_questions.csv is missing columns: {sorted(missing)}")
    # Remove the spaces around the questions and the labels.
    df["question"] = df["question"].str.strip()
    df["label"] = df["label"].str.strip()
    # Every label has to be one of the four.
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels in router_questions.csv: {bad}")
    # We compare the questions in lower case, so the same question written with other capitals also counts.
    dupes = df["question"].str.lower().duplicated()
    if dupes.any():
        raise ValueError(f"Duplicate questions: {df.loc[dupes, 'question'].tolist()[:5]}")
    return df


# Turn questions into vectors with the same embedding model as the knowledge base, which is already downloaded on
# every laptop. The vectors are normalised to length 1, as in the search.
class Embedder:

    def __init__(self, model_name: str):
        # Imported here, because the library is heavy.
        from sentence_transformers import SentenceTransformer

        # Load the model on the CPU, so it works the same on every laptop.
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    # Called like a function: one vector per text.
    def __call__(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


# Another way of turning questions into numbers: TF-IDF counts words and pairs of words, and gives more weight to the
# rare ones. We only use it to compare against the embeddings (--features tfidf); it's never saved as the router.
class TfidfFeatures:

    def __init__(self):
        # Imported here, because only this comparison needs it.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = "tfidf"
        # Single words and pairs of words; sublinear_tf dampens words repeated many times.
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._fitted = False

    # Learn the vocabulary from the questions.
    def fit(self, texts):
        self._vec.fit(texts)
        self._fitted = True
        return self

    # Called like a function: one row of numbers per text.
    def __call__(self, texts):
        return self._vec.transform(list(texts)).toarray()


# ----------------------------------------------------------------------
# Comparison

# Score the 5 classifiers with nested cross-validation and return a table sorted from best to worst. Outer loop: the
# questions are split into 5 folds, and each fold is the test set once. Inner loop: inside the other 4 folds,
# GridSearchCV tries every setting with 3 folds and keeps the best. So the outer score is always measured on questions
# that weren't used to choose the settings. StratifiedKFold keeps the same share of each label in every fold. The
# metric is macro-F1, the average F1 of the four labels, so every label counts the same.
def compare_classifiers(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    # The 5 outer folds.
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # A different seed for the inner folds, so they aren't cut in the same way as the outer ones.
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    # One row of results per classifier.
    rows = []
    for name, (estimator, grid) in MODELS.items():
        # The inner search: it tries every setting of this classifier and keeps the one with the best macro-F1.
        search = GridSearchCV(estimator, grid, cv=inner, scoring="f1_macro", n_jobs=1)
        # Start the clock for this classifier.
        t0 = time.perf_counter()
        # n_jobs=1: everything in one process (see the macOS note at the top).
        res = cross_validate(search, X, y, cv=outer, scoring=["f1_macro", "accuracy"], n_jobs=1)
        # The mean and the spread (standard deviation) of the 5 outer scores, and the time it took.
        rows.append({
            "model": name,
            "macro_f1_mean": res["test_f1_macro"].mean(),
            "macro_f1_std": res["test_f1_macro"].std(),
            "accuracy_mean": res["test_accuracy"].mean(),
            "train_seconds": time.perf_counter() - t0,
        })
        print(f"  {name:<20} macro-F1 {rows[-1]['macro_f1_mean']:.3f} ± {rows[-1]['macro_f1_std']:.3f}")
    # The table, sorted from the best macro-F1 to the worst.
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


# Read the label from the LLM's answer: the first of the four labels found in the text, or "invalid". router.py uses
# it too.
def parse_llm_label(text: str) -> str:
    # Compare in lower case.
    text = text.strip().lower()
    for label in LABELS:
        if label in text:
            return label
    return "invalid"


# Score the LLM used as a router, without any training (zero-shot). It never saw the labelled questions, so all 300
# are fair test questions. limit uses only the first questions, to try it quickly.
def llm_baseline(questions: list[str], y: np.ndarray, model: str, limit: int | None = None) -> dict:
    # Imported here, so the rest of the file works without Ollama.
    import ollama

    if limit:
        questions, y = questions[:limit], y[:limit]
    # The label the LLM gives to each question, and how long each one took.
    preds, seconds = [], []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        # temperature 0, so the same question always gets the same label.
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": q}],
            options={"temperature": 0},
        )
        # Keep the time and the label read from the answer.
        seconds.append(time.perf_counter() - t0)
        preds.append(parse_llm_label(resp["message"]["content"]))
        # Print the progress every 25 questions.
        if i % 25 == 0:
            print(f"  LLM: {i}/{len(questions)} questions")
    # The same metrics as the classifiers, plus how many answers weren't a valid label and the median time.
    return {
        "model": f"llm:{model}",
        "macro_f1": f1_score(y, preds, labels=LABELS, average="macro"),
        "accuracy": accuracy_score(y, preds),
        "invalid_answers": int(sum(p == "invalid" for p in preds)),
        "seconds_per_question": float(np.median(seconds)),
        "n_questions": len(questions),
    }


# Draw the confusion matrix (true label against predicted label) and save it as an image.
def save_confusion_figure(cm: np.ndarray, title: str, path) -> None:
    # Imported here, because only this function draws.
    import matplotlib

    # "Agg" draws straight to a file, without opening a window.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Make sure the folder exists, then draw the matrix in blue (darker = more questions).
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    # Short names for the four labels on both axes.
    short = ["admin", "content", "cross", "out of scope"]
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Write the count in each cell, in white on dark cells so it stays readable.
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    # Tidy the margins, save the image and close the figure.
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Training

# Train, compare and save the best router; return its scores (the content of router_meta.json).
def train(cfg, use_llm: bool = True, features: str = "embeddings", llm_limit: int | None = None) -> dict:
    # The questions and their labels.
    df = load_questions()
    texts, y = df["question"].tolist(), df["label"].to_numpy()
    print(f"Loaded {len(df)} router questions: {df['label'].value_counts().to_dict()}")

    # Turn the questions into numbers: embeddings (the real router) or TF-IDF (only to compare).
    if features == "tfidf":
        featurizer = TfidfFeatures().fit(texts)
    else:
        featurizer = Embedder(cfg.kb["embed_model"])
    # One row of numbers per question.
    X = featurizer(texts)
    print(f"Features: {featurizer.model_name}  shape={X.shape}")

    # Compare the 5 classifiers; the first row of the table is the best.
    print("Comparing 5 classifiers (nested 5-fold CV, macro-F1)...")
    table = compare_classifiers(X, y)
    # The name of the winner, and its estimator with the settings to try.
    best_name = table.loc[0, "model"]
    estimator, grid = MODELS[best_name]

    # The predictions of the winner for every question while it was in the test fold (out-of-fold), so the confusion
    # matrix and the scores of each label are measured on questions the model didn't train on.
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    search = GridSearchCV(estimator, grid, cv=inner, scoring="f1_macro", n_jobs=1)
    oof = cross_val_predict(search, X, y, cv=outer)
    # The confusion matrix and the scores of each label, from those predictions.
    cm = confusion_matrix(y, oof, labels=LABELS)
    per_class = classification_report(y, oof, labels=LABELS, output_dict=True, zero_division=0)

    # The final model: we choose the settings and fit it on all 300 questions.
    search.fit(X, y)
    final = search.best_estimator_
    # The SVM gives scores, not probabilities. CalibratedClassifierCV turns them into probabilities (with 5 folds),
    # which router.py needs to compare with min_prob.
    if not hasattr(final, "predict_proba"):
        final = CalibratedClassifierCV(final, cv=5).fit(X, y)

    # The average time to embed and classify one question, so we can compare the speed of the classifier with the LLM.
    t0 = time.perf_counter()
    for q in texts[:50]:
        final.predict_proba(featurizer([q]))
    clf_seconds = (time.perf_counter() - t0) / 50

    # The LLM comparison, with the answer model of the config (needs Ollama).
    llm = None
    if use_llm:
        print(f"LLM baseline with {cfg.generation['llm_model']} (zero-shot)...")
        llm = llm_baseline(texts, y, cfg.generation["llm_model"], limit=llm_limit)
        print(f"  LLM macro-F1 {llm['macro_f1']:.3f}  ({llm['seconds_per_question']:.2f} s/question)")

    # ---- Save the model and its metadata
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    # We only save the embedding version as the router: the classifier, the name of its embedding model (so router.py
    # embeds the questions in the same way) and the labels.
    if features == "embeddings":
        joblib.dump({"classifier": final, "embed_model": featurizer.model_name, "labels": LABELS},
                    PATHS["router_model"])
    # Everything about this training, which we save in models/router_meta.json for the documentation.
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

    # ---- Result files: the comparison table in reports/ and the confusion matrix in docs/figures/
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    # The TF-IDF run writes its own files ("_tfidf"), so it never overwrites the embedding results.
    # A copy of the table, so the original isn't changed.
    suffix = "" if features == "embeddings" else f"_{features}"
    out = table.copy()
    # We add the LLM as one more row of the table (it has no spread and no training time).
    if llm:
        out = pd.concat([out, pd.DataFrame([{
            "model": llm["model"], "macro_f1_mean": llm["macro_f1"], "macro_f1_std": np.nan,
            "accuracy_mean": llm["accuracy"], "train_seconds": 0.0}])], ignore_index=True)
    # Write the table with 4 decimals, and the figure.
    out.round(4).to_csv(PATHS["reports"] / f"router_comparison{suffix}.csv", index=False)
    save_confusion_figure(cm, f"Router ({best_name}), out-of-fold",
                          PATHS["figures"] / f"router_confusion{suffix}.png")

    # Print the final table and what was saved.
    print("\nResult")
    print(out.round(3).to_string(index=False))
    print(f"\nBest: {best_name} {search.best_params_}")
    if features == "embeddings":
        print(f"Saved {PATHS['router_model'].relative_to(PATHS['root'])} and "
              f"{PATHS['router_meta'].relative_to(PATHS['root'])}")
    return meta


# ----------------------------------------------------------------------
# Used by the answer phase (src/retrieval/router.py)

# The saved router: it embeds a question and gives the most likely label and its probability.
class Router:
    # bundle is what train() saved: the classifier, the name of its embedding model and the labels.
    def __init__(self, bundle: dict):
        from sentence_transformers import SentenceTransformer

        self.classifier = bundle["classifier"]
        # The labels in the order of the classifier's probabilities.
        self.labels = list(self.classifier.classes_)
        # The same embedding model we used for training, saved in the bundle.
        self._embed = SentenceTransformer(bundle["embed_model"], device="cpu")

    # Return (label, probability of that label) for one question.
    def route(self, question: str) -> tuple[str, float]:
        # The vector of the question, normalised like the training questions.
        x = self._embed.encode([question], normalize_embeddings=True, show_progress_bar=False)
        # One probability per label; the highest one wins.
        probs = self.classifier.predict_proba(x)[0]
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i])


# Load the router saved by train() (models/router.joblib by default). Called by router.py.
def load_router(path=None) -> Router:
    return Router(joblib.load(path or PATHS["router_model"]))


# Command line: runs only when the file is executed directly (see the commands at the top).
if __name__ == "__main__":
    # The options of the command.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM baseline")
    ap.add_argument("--llm-limit", type=int, default=None, help="only N questions for the LLM baseline")
    ap.add_argument("--features", choices=["embeddings", "tfidf"], default="embeddings")
    # Read them and train.
    args = ap.parse_args()
    train(load_config(args.config), use_llm=not args.no_llm, features=args.features,
          llm_limit=args.llm_limit)
