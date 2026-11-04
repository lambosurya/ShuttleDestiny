# 🏸 ShuttleSense

**Badminton shot prediction powered by machine learning.**

ShuttleSense is an interactive web app that estimates the **next shot in a badminton rally** based on the previous shot type, where it landed, and its position in the rally. It uses a Random Forest classifier trained on annotated professional singles matches from the [ShuttleSet dataset](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet).

**Live app:** *[ShuttleSense_AI.](https://shuttlesense-ai.streamlit.app/)*

## Features

- **Interactive court:** Select a landing zone on a numbered badminton court.
- **Shot prediction:** Choose from 18 shot types and estimate the next shot using the trained model.
- **Probability breakdown:** View the five most likely next shots and the full probability distribution.
- **Dataset explorer:** View match, rally, and stroke counts along with common shot types.
- **Model evaluation:** Calculate held-out test accuracy from the included match data.

## Using the website

1. Open the **live app** using the link above (once published).
2. On the **Predict next shot** tab, select the **previous shot type**.
3. Click the shuttle's **landing zone (1–9)** on the court. The **net is at the top** of the diagram.
4. Enter the **previous shot's position in the rally** (for example, `5` means the fifth stroke).
5. Select **Predict next shot** to see the model's predictions and probabilities.

The court also displays **zones 10–16**, which are outside the singles boundaries. Predictions are disabled for these zones because an out shot normally ends the rally.

To explore the underlying data, switch to **Explore dataset**. This tab shows summary statistics and shot frequencies. If the dataset is available, you can also expand **Check this saved model's held-out accuracy** and select **Calculate test accuracy**.

## Model and results

| Item | Details |
| --- | --- |
| Algorithm | Random Forest classifier (scikit-learn) |
| Inputs | Previous shot type, landing zone, rally shot number |
| Output | Probability distribution across possible next shots |
| Held-out test accuracy | **40.8%** |
| Most-common-shot baseline | **16.9%** |
| Training data | ShuttleSet professional singles match annotations |

The evaluation splits examples **by match**, so strokes from a match in the test set are not used for training. Accuracy measures whether the predicted next shot type matches the recorded shot; it is **not** a measure of how likely a player is to win the rally.

## Run locally

### Requirements

- Python 3
- Git (to clone the repository)

Clone the repository and enter the project folder:

```bash
git clone https://github.com/lambosurya/ShuttleSense-AI.git
cd ShuttleSense-AI
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501** if the browser doesn't open automatically. To stop the app, press **Ctrl+C** in the terminal.

**Note:** The trained model file, `shuttle_model.joblib`, must be in the repository root alongside `app.py`. You do **not** need to retrain the model to use the predictor.

## Repository structure

```text
ShuttleSense-AI/
├── app.py                  # Streamlit website and clickable court
├── shuttle_model.joblib    # Pretrained shot-prediction model
├── train.py                # Model training and evaluation
├── predict.py              # Command-line prediction tool
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # App theme
├── data/
│   └── ShuttleSet/
│       └── set/            # Match CSV files (if included)
└── README.md               # Project documentation
```

The **predictor** requires the saved model. The **dataset explorer and accuracy evaluation** also require the ShuttleSet CSV files. If those files aren't included in the repository, obtain them from the [original ShuttleSet project](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet) and place the `set` directory under `data/ShuttleSet/`.

To retrain the model on the provided match data, run `python train.py`. This **overwrites** `shuttle_model.joblib`, so keep a backup if you want to preserve the current model.

## Tech stack

**Python · Pandas · scikit-learn · Streamlit · Joblib**

## Limitations

ShuttleSense identifies **statistical shot patterns**; it does not analyze live video, track player movement, or recommend an objectively optimal shot. Predictions are estimates and may not generalize to every playing style or skill level.

## Data and acknowledgments

This project uses the **ShuttleSet** badminton dataset. Please consult the original project's terms before redistributing its data.

Wang, W.-Y., Huang, Y.-C., Ik, T.-U., & Peng, W.-C. (2023). *ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis.* KDD 2023. [Paper](https://arxiv.org/pdf/2306.04948) · [Dataset](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)
