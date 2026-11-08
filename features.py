"""Shared data loading and feature engineering for ShuttleDestiny.

Each training example is one shot in a rally. The label is the type of the
NEXT shot (what the opponent plays in reply). Every feature describes things
already known when that shot lands: the shot itself, the shots before it,
where both players were standing, where the shuttle is landing, and the score.

Positions come from ShuttleSet's per-match homography, which maps broadcast
video pixels onto one shared court frame. We convert that to metres and turn
the court so the HITTER is always on the near side (negative y):
  x = sideways, metres from the centre line (+ = hitter's right)
  y = metres from the net (negative = hitter's half, positive = receiver's half)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SET_DIR = DATA_DIR / "ShuttleSet" / "set"

# ShuttleSet annotations use Chinese shot names; map them to English.
SHOT_NAMES = {
    "放小球": "net shot", "擋小球": "return net", "殺球": "smash",
    "點扣": "wrist smash", "挑球": "lob", "防守回挑": "defensive return lob",
    "長球": "clear", "平球": "drive", "小平球": "driven flight",
    "後場抽平球": "back-court drive", "切球": "drop",
    "過渡切球": "passive drop", "推球": "push", "撲球": "rush",
    "防守回抽": "defensive return drive", "勾球": "cross-court net shot",
    "發短球": "short service", "發長球": "long service",
}
SHOTS = list(SHOT_NAMES.values())
NONE = "none"  # placeholder for "no earlier shot in this rally"

# Court frame produced by the homography: x 25..325, y 150..810, net at y=480.
COURT_CX, COURT_NET_Y = 175.0, 480.0
METRES_PER_UNIT = 6.10 / 300.0  # doubles width 6.10 m spans 300 units

CATEGORICAL = ["shot", "prev_shot", "prev2_shot"]
POSITION = [
    "land_x", "land_y", "hitter_x", "hitter_y", "receiver_x", "receiver_y",
    "receiver_to_land", "land_depth",
]
NUMERIC = [
    "ball_round", "hit_high", "backhand", "around_head",
    "score_diff", "points_total",
] + POSITION
FEATURES = CATEGORICAL + NUMERIC
ORIGINAL_FEATURES = ["shot", "landing_area", "ball_round"]


def shot_name(value):
    text = str(value).strip().lower()
    return SHOT_NAMES.get(text, text)


def load_homographies():
    table = pd.read_csv(SET_DIR / "homography.csv")
    return {row.video: np.array(json.loads(row.homography_matrix)) for row in table.itertuples()}


def load_players():
    """ShuttleSet labels the match winner 'A' and the loser 'B'."""
    table = pd.read_csv(SET_DIR / "match.csv")
    return {row.video: {"A": row.winner, "B": row.loser} for row in table.itertuples()}


def to_court(H, px, py):
    """Pixel coordinates -> court metres (unrotated)."""
    pts = np.vstack([px, py, np.ones_like(px)])
    q = H @ pts
    x = (q[0] / q[2] - COURT_CX) * METRES_PER_UNIT
    y = (q[1] / q[2] - COURT_NET_Y) * METRES_PER_UNIT
    return x, y


def build_examples(shots: pd.DataFrame, H, names) -> pd.DataFrame:
    """Turn one set's stroke table into (features, next shot) rows."""
    shots = shots.dropna(subset=["rally", "ball_round", "type"]).copy()
    shots["ball_round"] = pd.to_numeric(shots["ball_round"], errors="coerce")
    shots = shots.dropna(subset=["ball_round"]).sort_values(["rally", "ball_round"])
    shots["shot"] = shots["type"].map(shot_name)

    by_rally = shots.groupby("rally")
    shots["next_shot"] = by_rally["shot"].shift(-1)
    shots["prev_shot"] = by_rally["shot"].shift(1).fillna(NONE)
    shots["prev2_shot"] = by_rally["shot"].shift(2).fillna(NONE)

    hx, hy = to_court(H, shots["player_location_x"].to_numpy(float), shots["player_location_y"].to_numpy(float))
    rx, ry = to_court(H, shots["opponent_location_x"].to_numpy(float), shots["opponent_location_y"].to_numpy(float))
    lx, ly = to_court(H, shots["landing_x"].to_numpy(float), shots["landing_y"].to_numpy(float))
    # Rotate the court 180 degrees when the hitter is on the far side, so the
    # hitter is always at negative y. The homography's x axis runs toward the
    # hitter's LEFT in this frame (checked against the singles serving rule:
    # 95% of serves at an even score come from the side this makes the
    # server's right), so x is negated to make + the hitter's right.
    flip = np.where(hy > 0, -1.0, 1.0)
    shots["hitter_x"], shots["hitter_y"] = -hx * flip, hy * flip
    shots["receiver_x"], shots["receiver_y"] = -rx * flip, ry * flip
    shots["land_x"], shots["land_y"] = -lx * flip, ly * flip
    shots["receiver_to_land"] = np.hypot(shots["land_x"] - shots["receiver_x"], shots["land_y"] - shots["receiver_y"])
    shots["land_depth"] = shots["land_y"]  # distance from net into receiver's half

    # hit_height: 1 = shuttle contacted above the net, 2 = below.
    shots["hit_high"] = (shots["hit_height"] == 1).astype(float)
    shots["backhand"] = shots["backhand"].fillna(0).astype(float)
    shots["around_head"] = shots["aroundhead"].fillna(0).astype(float)

    # Score from the hitter's point of view at the START of the rally.
    # ShuttleSet records the score after the rally, so the rally winner's
    # point is taken back off (otherwise the score would leak the outcome).
    winner = shots["rally"].map(shots.dropna(subset=["getpoint_player"]).groupby("rally")["getpoint_player"].last())
    score_a = shots["roundscore_A"] - (winner == "A").astype(int)
    score_b = shots["roundscore_B"] - (winner == "B").astype(int)
    is_a = shots["player"] == "A"
    own = np.where(is_a, score_a, score_b)
    opp = np.where(is_a, score_b, score_a)
    shots["score_diff"] = (own - opp).astype(float)
    shots["points_total"] = (own + opp).astype(float)

    shots["hitter_name"] = shots["player"].map(names)
    shots["receiver_name"] = shots["player"].map({"A": names["B"], "B": names["A"]})

    shots = shots.dropna(subset=["next_shot", "landing_area", "land_x", "receiver_x", "hitter_x"])
    shots = shots[shots["next_shot"].isin(SHOTS) & shots["shot"].isin(SHOTS)]
    # Drop the rare annotation glitches that map far outside the court.
    ok = (shots[["land_x", "receiver_x", "hitter_x"]].abs() < 5).all(axis=1) & \
         (shots[["land_y", "receiver_y", "hitter_y"]].abs() < 9).all(axis=1)
    return shots[ok]


def load_dataset() -> pd.DataFrame:
    homographies, players = load_homographies(), load_players()
    frames = []
    required = {"rally", "ball_round", "type", "landing_area", "player"}
    for csv_path in sorted(SET_DIR.glob("*/*.csv")):
        match = csv_path.parent.name
        table = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        if not required.issubset(table.columns) or match not in homographies:
            continue
        examples = build_examples(table, homographies[match], players[match])
        # All sets of one match share a group, so a match is never split
        # between training and testing.
        examples["match"] = match
        frames.append(examples)
    if not frames:
        raise SystemExit("No ShuttleSet stroke CSVs found under data/. See README.md.")
    keep = FEATURES + ["landing_area", "next_shot", "hitter_name", "receiver_name", "match"]
    data = pd.concat(frames, ignore_index=True)[keep]
    data["landing_area"] = data["landing_area"].astype(int)
    return data


SEED = 42
# Share of training rows whose receiving player is hidden, so the model also
# learns to predict well when the visitor chooses "any player".
RECEIVER_MASK_RATE = 0.2
CATEGORY_VOCAB = SHOTS + [NONE]

PARAMS = dict(learning_rate=0.08, max_iter=200, max_leaf_nodes=12, min_samples_leaf=100,
              l2_regularization=5.0, early_stopping=True, validation_fraction=0.12,
              n_iter_no_change=25, random_state=SEED)


class ShuttleDestinyModel:
    """Small wrapper: turns readable inputs into the model's number columns."""

    def __init__(self, players):
        self.players = players
        self.columns = CATEGORICAL + ["receiver"] + NUMERIC
        self.clf = HistGradientBoostingClassifier(
            categorical_features=list(range(len(CATEGORICAL) + 1)), **PARAMS)

    def encode(self, df, mask_rate=0.0, rng=None):
        X = pd.DataFrame(index=df.index)
        for col in CATEGORICAL:
            X[col] = df[col].map({s: i for i, s in enumerate(CATEGORY_VOCAB)}).astype(float)
        names = df["receiver_name"] if "receiver_name" in df else pd.Series(np.nan, index=df.index)
        X["receiver"] = names.map({n: i for i, n in enumerate(self.players)}).astype(float)
        if mask_rate:
            X.loc[rng.random(len(X)) < mask_rate, "receiver"] = np.nan
        for col in NUMERIC:
            X[col] = df[col].astype(float)
        return X[self.columns]

    def fit(self, df, y):
        rng = np.random.default_rng(SEED)
        self.clf.fit(self.encode(df, RECEIVER_MASK_RATE, rng), y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, df):
        return self.clf.predict_proba(self.encode(df))
