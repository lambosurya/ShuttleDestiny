"""Ask the trained ShuttleDestiny model what shot might come next, from the terminal.

Usage:  python predict.py

Court positions are in metres, seen from the HITTER's side of the court:
  x = sideways from the centre line (negative = hitter's left, positive = right)
  y = distance from the net (hitter's half is negative, receiver's half positive)
The court is 6.1 m wide (x from -3.05 to 3.05) and 13.4 m long (y from -6.7 to 6.7).
The website (site/index.html) lets you click these positions on a court instead.
"""
import math
from pathlib import Path

import joblib
import pandas as pd

from features import ShuttleDestinyModel  # noqa: F401  (needed to load the model)

MODEL_FILE = Path(__file__).resolve().parent / "shuttledestiny_model.joblib"


def ask(prompt, default, cast=str):
    text = input(f"{prompt} [{default}]: ").strip()
    return cast(text) if text else cast(default)


def main():
    if not MODEL_FILE.exists():
        raise SystemExit("No trained model found. Run: python train.py")
    model = joblib.load(MODEL_FILE)  # only load model files you created or trust

    print("\nSHUTTLEDESTINY - NEXT SHOT PREDICTOR (press Enter to accept defaults)\n")
    shot = ask("Shot just played (e.g. clear, smash, drop)", "clear").lower()
    prev = ask("Shot before that, or 'none'", "lob").lower()
    prev2 = ask("Two shots before, or 'none'", "none").lower()
    ball_round = ask("Shot number in the rally", 4, int)
    land_x = ask("Landing spot x (m)", 1.5, float)
    land_y = ask("Landing spot y (m, receiver half is positive)", 5.5, float)
    rec_x = ask("Receiver position x (m)", 0.0, float)
    rec_y = ask("Receiver position y (m)", 3.5, float)
    hit_x = ask("Hitter position x (m)", 0.0, float)
    hit_y = ask("Hitter position y (m, negative)", -4.0, float)
    receiver = ask("Receiving player name, or 'any'", "any")

    row = {
        "shot": shot, "prev_shot": prev, "prev2_shot": prev2, "ball_round": ball_round,
        "hit_high": 1.0, "backhand": 0.0, "around_head": 0.0,
        "score_diff": 0.0, "points_total": 10.0,
        "land_x": land_x, "land_y": land_y, "land_depth": land_y,
        "hitter_x": hit_x, "hitter_y": hit_y, "receiver_x": rec_x, "receiver_y": rec_y,
        "receiver_to_land": math.hypot(land_x - rec_x, land_y - rec_y),
        "receiver_name": None if receiver.lower() == "any" else receiver,
    }
    probs = model.predict_proba(pd.DataFrame([row]))[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda pair: pair[1], reverse=True)
    print("\nMost likely replies:")
    for name, p in ranked[:5]:
        print(f"  {name:<24} {p:6.1%}  {'#' * round(p * 40)}")
    print("\nThese are estimates learned from professional singles matches, not guarantees.")


if __name__ == "__main__":
    main()
