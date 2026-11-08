"""Train the ShuttleDestiny next-shot model and export it for the website.

Usage:  python train.py

What it does
1. Builds one example per shot from ShuttleSet (see features.py).
2. Scores the model honestly with 5-fold cross-validation BY MATCH: the model
   is always tested on matches it never saw during training.
3. Compares it with a majority-class baseline and with the original
   3-feature Random Forest that ShuttleDestiny v1 used.
4. Trains the final model on all matches and writes:
     shuttledestiny_model.joblib        -> used by predict.py
     site/data/model.json        -> the same model, run in the browser
     site/data/metrics.json      -> numbers shown on the website
     site/data/analytics.json    -> shot statistics for the analytics page
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             top_k_accuracy_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import (CATEGORICAL, CATEGORY_VOCAB, NUMERIC, ORIGINAL_FEATURES, ROOT, SEED, SHOTS,
                      ShuttleDestinyModel, load_dataset)

SITE_DATA = ROOT / "site" / "data"
MODEL_FILE = ROOT / "shuttledestiny_model.joblib"
def original_model():
    """ShuttleDestiny v1: Random Forest on shot type, landing area, shot number."""
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["shot", "landing_area"]),
            ("numeric", "passthrough", ["ball_round"]),
        ])),
        ("classifier", RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                              random_state=SEED, n_jobs=-1)),
    ])


def scores(y, proba, classes):
    pred = classes[proba.argmax(1)]
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "top3": top_k_accuracy_score(y, proba, k=3, labels=classes),
        "log_loss": log_loss(y, proba, labels=classes),
    }


def mean_scores(rows):
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]} | {
        "accuracy_std": round(float(np.std([r["accuracy"] for r in rows])), 4)}


def cross_validate(data, players):
    """5-fold cross-validation by match. Returns metrics and pooled predictions."""
    folds = GroupKFold(n_splits=5)
    new, new_any, old, base = [], [], [], []
    y_all, pred_all = [], []
    for k, (train_idx, test_idx) in enumerate(folds.split(data, groups=data["match"]), 1):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        y_test = test["next_shot"]

        model = ShuttleDestinyModel(players).fit(train, train["next_shot"])
        proba = model.predict_proba(test)
        new.append(scores(y_test, proba, model.classes_))
        new_any.append(scores(y_test, model.predict_proba(test.drop(columns="receiver_name")), model.classes_))
        y_all.append(y_test.to_numpy())
        pred_all.append(model.classes_[proba.argmax(1)])

        v1 = original_model().fit(train[ORIGINAL_FEATURES], train["next_shot"])
        old.append(scores(y_test, v1.predict_proba(test[ORIGINAL_FEATURES]), v1.classes_))

        majority = train["next_shot"].mode()[0]
        base.append({"accuracy": float((y_test == majority).mean())})
        print(f"  fold {k}: {test['match'].nunique()} test matches | "
              f"new {new[-1]['accuracy']:.1%} | v1 {old[-1]['accuracy']:.1%} | baseline {base[-1]['accuracy']:.1%}")

    y_all, pred_all = np.concatenate(y_all), np.concatenate(pred_all)
    replies = [s for s in SHOTS if s in set(y_all)]  # serves can never be a reply
    per_shot = f1_score(y_all, pred_all, labels=replies, average=None, zero_division=0)
    conf = confusion_matrix(y_all, pred_all, labels=replies)
    return {
        "new_known_player": mean_scores(new),
        "new_any_player": mean_scores(new_any),
        "v1_random_forest": mean_scores(old),
        "baseline_accuracy": round(float(np.mean([b["accuracy"] for b in base])), 4),
        "per_shot_f1": {s: round(float(f), 3) for s, f in zip(replies, per_shot)},
        "confusion": {"labels": replies, "matrix": conf.tolist()},
    }


def feature_importance(data, players):
    """How much accuracy drops when each input is scrambled (on held-out matches)."""
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(data, groups=data["match"]))
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    model = ShuttleDestinyModel(players).fit(train, train["next_shot"])
    y = test["next_shot"]
    base = accuracy_score(y, model.classes_[model.predict_proba(test).argmax(1)])
    groups = {
        "Current shot type": ["shot"],
        "Earlier shots (2 back)": ["prev_shot", "prev2_shot"],
        "Landing spot": ["land_x", "land_y", "land_depth"],
        "Receiver's position": ["receiver_x", "receiver_y", "receiver_to_land"],
        "Hitter's position": ["hitter_x", "hitter_y"],
        "Receiving player": ["receiver_name"],
        "Contact (height, backhand)": ["hit_high", "backhand", "around_head"],
        "Shot number & score": ["ball_round", "score_diff", "points_total"],
    }
    rng = np.random.default_rng(SEED)
    result = {}
    for label, cols in groups.items():
        drops = []
        for _ in range(3):
            shuffled = test.copy()
            order = rng.permutation(len(test))
            for col in cols:
                shuffled[col] = test[col].to_numpy()[order]
            acc = accuracy_score(y, model.classes_[model.predict_proba(shuffled).argmax(1)])
            drops.append(base - acc)
        result[label] = round(float(np.mean(drops)), 4)
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def export_for_browser(model):
    """Write the boosted trees as compact JSON for site/js/model.js."""
    clf = model.clf
    trees = []
    for iteration in clf._predictors:
        for predictor in iteration:  # one tree per class
            n = predictor.nodes
            trees.append({
                "f": n["feature_idx"].astype(int).tolist(),
                "t": [float(v) for v in n["num_threshold"]],
                "l": n["left"].astype(int).tolist(),
                "r": n["right"].astype(int).tolist(),
                "m": n["missing_go_to_left"].astype(int).tolist(),
                "leaf": n["is_leaf"].astype(int).tolist(),
                "v": [round(float(v), 6) for v in n["value"]],
                "c": n["is_categorical"].astype(int).tolist(),
                "b": n["bitset_idx"].astype(int).tolist(),
                "bits": predictor.raw_left_cat_bitsets.astype(int).tolist(),
            })
    # scikit-learn re-encodes each categorical column to 0..n-1 using the
    # values seen in training (unseen values become missing); the trees use
    # those re-encoded codes, so the browser needs the same mapping.
    encoder = clf._preprocessor.named_transformers_["encoder"]
    category_maps = [[int(c) for c in cats if not np.isnan(c)] for cats in encoder.categories_]
    return {
        "version": 2,
        "classes": clf.classes_.tolist(),
        "columns": model.columns,
        "categorical_count": len(CATEGORICAL) + 1,
        "vocab": CATEGORY_VOCAB,
        "players": model.players,
        "category_maps": category_maps,
        "baseline": [round(float(v), 6) for v in np.ravel(clf._baseline_prediction)],
        "n_classes": len(clf.classes_),
        "trees": trees,
    }


def analytics(data):
    shot_counts = data["shot"].value_counts()
    follow = pd.crosstab(data["shot"], data["next_shot"]).reindex(index=SHOTS, columns=SHOTS, fill_value=0)
    by_player = {}
    for name, rows in data.groupby("receiver_name"):
        if len(rows) < 300:
            continue
        share = rows["next_shot"].value_counts(normalize=True)
        by_player[name] = {"replies": int(len(rows)),
                           "mix": {s: round(float(share.get(s, 0)), 4) for s in SHOTS}}
    rally = data.groupby(data["match"].astype(str))["ball_round"].max()
    return {
        "examples": int(len(data)),
        "matches": int(data["match"].nunique()),
        "players": int(pd.concat([data["receiver_name"], data["hitter_name"]]).nunique()),
        "shot_counts": {s: int(shot_counts.get(s, 0)) for s in SHOTS},
        "follow_matrix": {"labels": SHOTS, "matrix": follow.to_numpy().tolist()},
        "by_player": by_player,
        "longest_rally": int(data["ball_round"].max()),
        "median_match_longest_rally": float(rally.median()),
    }


def main():
    start = time.time()
    data = load_dataset()
    players = sorted(set(data["receiver_name"]) | set(data["hitter_name"]))
    print(f"Examples: {len(data):,} shots from {data['match'].nunique()} matches, {len(players)} players")

    print("\nCross-validating (5 folds, split by match)...")
    cv = cross_validate(data, players)
    k, a, o = cv["new_known_player"], cv["new_any_player"], cv["v1_random_forest"]
    print(f"\nMajority baseline     accuracy {cv['baseline_accuracy']:.1%}")
    print(f"v1 Random Forest      accuracy {o['accuracy']:.1%}  macro-F1 {o['macro_f1']:.3f}  top-3 {o['top3']:.1%}")
    print(f"v2 (any player)       accuracy {a['accuracy']:.1%}  macro-F1 {a['macro_f1']:.3f}  top-3 {a['top3']:.1%}")
    print(f"v2 (player selected)  accuracy {k['accuracy']:.1%}  macro-F1 {k['macro_f1']:.3f}  top-3 {k['top3']:.1%}")

    print("\nMeasuring which inputs matter...")
    importance = feature_importance(data, players)
    for label, drop in importance.items():
        print(f"  {label:28s} {-drop:+.1%}")

    print("\nTraining final model on all matches...")
    final = ShuttleDestinyModel(players).fit(data, data["next_shot"])
    joblib.dump(final, MODEL_FILE, compress=3)

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "model.json").write_text(json.dumps(export_for_browser(final), separators=(",", ":")))
    metrics = cv | {"importance": importance, "trees": len(final.clf._predictors),
                    "trained_on": {"shots": int(len(data)), "matches": int(data["match"].nunique())}}
    (SITE_DATA / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (SITE_DATA / "analytics.json").write_text(json.dumps(analytics(data), separators=(",", ":")))
    print(f"Saved {MODEL_FILE.name} and site/data/*.json in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
