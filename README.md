# 🏸 ShuttleSense — Interactive Badminton Shot Predictor

Your existing trained **Random Forest** model is included in this project as `shuttle_model.joblib`. You **do not need to train it again** to use the app.

## Quick start on Mac (VS Code)

1. Unzip `ShuttleSense-Interactive.zip`.
2. Open Visual Studio Code → **File → Open Folder...** → select the extracted `ShuttleSense-Interactive` folder.
3. At the very top of the Mac screen, choose **Terminal → New Terminal**.
4. Run these commands **one at a time**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Your browser should automatically open the app, usually at **http://localhost:8501**.

**Important:** All terminal commands should be run while your terminal is inside the `ShuttleSense-Interactive` folder. You can check with `pwd`. When you come back to your project in the future, you only need:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

To stop the app: press **Control + C** in the terminal.

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Using the app

1. Select a **previous shot** (e.g., `clear`, `smash`, `net shot`).
2. Click an **in-court zone (1–9)** on the green court. The **net is at the top**; you're looking from behind the baseline.
3. Enter the **number of the previous stroke in the rally** (e.g., `5` for the fifth shot).
4. Click **Predict next shot**. The app shows the five most likely next shot types, plus the full distribution.
5. Open **Explore dataset** to see shot frequencies and optionally evaluate your saved model on the original held-out test matches.

Outside zones 10–16 are shown in brown; an out shot typically ends the rally, so prediction is intentionally disabled for those zones.

### Court numbering (from ShuttleSet Figure 4)

```text
                            NET
             ───────────────────────────
        10     2       7       1      16
        11     6       8       5      15
        12     4       9       3      14
             ─────── BASELINE ─────────
                      13 (OUT)
```

Zone numbers are dataset-specific. This half-court is a **schematic representation**, not to scale. See the paper's Figure 4 for the original full-court diagram.

## Project files

- `app.py` — Streamlit interface and the clickable court.
- `shuttle_model.joblib` — **Your already-trained model**. Do not delete it.
- `train.py` — Original training script, unchanged.
- `predict.py` — Original terminal-based predictor, unchanged.
- `requirements.txt` — Python libraries needed for the app, including the version of scikit-learn used to train your saved model.
- `data/ShuttleSet/set/` — Your original professional-match CSV files, preserved.
- `.streamlit/config.toml` — Application color theme.

## How the prediction is produced

The saved model expects exactly these features (with the original text encoding):

| Feature | Example | Type |
|---|---|---|
| `previous_shot` | `clear` | text |
| `landing_area` | `3` | **text** representation of zone number |
| `ball_round` | `5` | integer |

The app passes these features to `model.predict_proba()` and visualizes the returned shot probabilities. The original trained model stores two Chinese labels (`過度切球`, `未知球種`); the app translates them for display and maps the **passive drop** input back to the label the saved model expects. **No new machine learning model was created.**

**Limitations:** The model learns statistical shot transitions from a relatively small professional singles dataset; it does not track player movement or predict the optimal shot. Model probabilities should not be treated as certainties.

## Dataset credit

Wang, W.-Y., Huang, Y.-C., Ik, T.-U., & Peng, W.-C. (2023). *ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis*. KDD 2023. https://arxiv.org/pdf/2306.04948

Original dataset: https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet
