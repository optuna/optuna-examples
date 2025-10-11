from __future__ import annotations
import argparse, random
import numpy as np
import optuna
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)

def load_tweeteval_sentiment(max_train: int, seed: int):
    ds = load_dataset("tweet_eval", "sentiment")
    def to_xy(split):
        X = [r["text"] for r in ds[split]]
        y = [int(r["label"]) for r in ds[split]]
        return X, y
    X_train_all, y_train_all = to_xy("train")
    X_valid, y_valid = to_xy("validation")
    X_test, y_test = to_xy("test")
    if max_train and 0 < max_train < len(X_train_all):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train_all), size=max_train, replace=False)
        X_train = [X_train_all[i] for i in idx]; y_train = [y_train_all[i] for i in idx]
    else:
        X_train, y_train = X_train_all, y_train_all
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def build_pipeline(trial: optuna.Trial):
    max_features = trial.suggest_int("tfidf_max_features", 10000, 60000, step=5000)
    min_df = trial.suggest_float("tfidf_min_df", 1e-5, 5e-3, log=True)
    ngram_high = trial.suggest_int("tfidf_ngram_high", 1, 2)
    sublinear_tf = trial.suggest_categorical("tfidf_sublinear_tf", [True, False])
    C = trial.suggest_float("svm_C", 1e-2, 10.0, log=True)
    loss = trial.suggest_categorical("svm_loss", ["hinge", "squared_hinge"])
    class_weight = trial.suggest_categorical("svm_class_weight", [None, "balanced"])
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df,
                          ngram_range=(1, ngram_high), sublinear_tf=sublinear_tf, norm="l2")
    svm = LinearSVC(C=C, loss=loss, class_weight=class_weight, random_state=0)
    return make_pipeline(vec, svm)

def objective(trial: optuna.Trial, X_train, y_train, X_valid, y_valid):
    pipe = build_pipeline(trial)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_valid)
    return 1.0 - f1_score(y_valid, y_pred, average="macro")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_tweeteval_sentiment(args.max_train, args.seed)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(lambda t: objective(t, Xtr, ytr, Xva, yva),
                   n_trials=args.n_trials, show_progress_bar=True)
    best = study.best_trial
    print("\nBest validation (1 - macroF1):", best.value)
    print("Best params:", best.params)
    # retrain on train+valid, evaluate on test
    pipe = build_pipeline(best)
    pipe.fit(Xtr + Xva, ytr + yva)
    y_pred = pipe.predict(Xte)
    print("\n=== Test set ===")
    print("macro-F1:", f1_score(yte, y_pred, average="macro"))
    print(classification_report(yte, y_pred, digits=4))

if __name__ == "__main__":
    main()
