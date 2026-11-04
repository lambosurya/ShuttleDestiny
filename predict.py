"""Ask the trained model what shot might come next in a rally."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
model_path = ROOT / "shuttle_model.joblib"
if not model_path.exists():
    raise SystemExit("No trained model found. Run: python train.py")

model = joblib.load(model_path)  # Only load model files that you created/trust.
print("\n🏸 SHUTTLESENSE — NEXT SHOT PREDICTOR")
previous_shot = input("Previous shot type (e.g., clear, smash, drop): ").strip().lower()
landing_area = input("Previous shot's numeric landing area (e.g., 5): ").strip()
shot_number = int(input("Previous shot's position in rally (e.g., 5): ").strip())

sample = pd.DataFrame([{
    "previous_shot": previous_shot,
    "landing_area": landing_area,
    "ball_round": shot_number,
}])
probs = model.predict_proba(sample)[0]
results = sorted(zip(model.classes_, probs), key=lambda pair: pair[1], reverse=True)
print("\nPredicted next shot probabilities:")
for shot, probability in results[:5]:
    print(f"  {shot:<24} {probability:.1%}")
print("\nThese are model estimates from historical singles rallies, not guarantees.")
