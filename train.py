"""Train a badminton next-shot predictor using ShuttleSet CSV annotations."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_FILE = ROOT / "shuttle_model.joblib"

# ShuttleSet annotations sometimes use Chinese shot names.
SHOT_NAMES = {
    "放小球": "net shot", "擋小球": "return net", "殺球": "smash",
    "點扣": "wrist smash", "挑球": "lob", "防守回挑": "defensive return lob",
    "長球": "clear", "平球": "drive", "小平球": "driven flight",
    "後場抽平球": "back-court drive", "切球": "drop",
    "過渡切球": "passive drop", "推球": "push", "撲球": "rush",
    "防守回抽": "defensive return drive", "勾球": "cross-court net shot",
    "發短球": "short service", "發長球": "long service",
}


def shot_name(value):
    text = str(value).strip().lower()
    return SHOT_NAMES.get(text, text)


def area_name(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_rallies():
    examples = []
    required = {"rally", "ball_round", "type", "landing_area"}
    for csv_path in DATA_DIR.rglob("*.csv"):
        try:
            shots = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        except (UnicodeError, pd.errors.ParserError) as error:
            print(f"Skipping {csv_path.name}: {error}")
            continue
        if not required.issubset(shots.columns):
            continue  # E.g., match.csv and homography.csv aren't shot tables.

        shots = shots.dropna(subset=["rally", "ball_round", "type", "landing_area"]).copy()
        shots["ball_round"] = pd.to_numeric(shots["ball_round"], errors="coerce")
        shots = shots.dropna(subset=["ball_round"])
        shots = shots.sort_values(["rally", "ball_round"])

        # Each row describes a known shot; the label is the NEXT shot in its rally.
        shots["next_shot"] = shots.groupby("rally")["type"].shift(-1)
        shots = shots.dropna(subset=["next_shot"]).copy()
        if shots.empty:
            continue
        shots["previous_shot"] = shots["type"].map(shot_name)
        shots["landing_area"] = shots["landing_area"].map(area_name)
        shots["next_shot"] = shots["next_shot"].map(shot_name)
        # All sets within one source match belong to the same evaluation group.
        shots["match_group"] = str(csv_path.parent.relative_to(DATA_DIR))
        examples.append(shots[["previous_shot", "landing_area", "ball_round", "next_shot", "match_group"]])

    if not examples:
        raise SystemExit(
            "No ShuttleSet stroke CSVs found. Download ShuttleSet and copy its set/ "
            "directory into this project's data/ directory. See README.md."
        )
    dataset = pd.concat(examples, ignore_index=True)
    if dataset["match_group"].nunique() < 2:
        raise SystemExit("Need data from at least TWO match folders for a match-level train/test split.")
    return dataset


def main():
    data = load_rallies()
    feature_columns = ["previous_shot", "landing_area", "ball_round"]
    features = data[feature_columns]
    labels = data["next_shot"]

    # Split by MATCH, not by individual stroke: avoids testing on the same match.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(features, labels, groups=data["match_group"]))
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["previous_shot", "landing_area"]),
        ("numeric", "passthrough", ["ball_round"]),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    most_common = y_train.mode().iloc[0]
    baseline = (y_test == most_common).mean()
    print(f"Training examples: {len(x_train):,} | Test examples: {len(x_test):,}")
    print(f"Training matches: {data.iloc[train_idx]['match_group'].nunique()}")
    print(f"Test matches: {data.iloc[test_idx]['match_group'].nunique()}")
    print(f"Majority-class baseline accuracy: {baseline:.1%} ({most_common})")
    print(f"Model test accuracy: {accuracy_score(y_test, predictions):.1%}")
    print("\nPer-shot performance:\n")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved trained model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
