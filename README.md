# ShuttleDestiny

**Choose your shuttle's destiny.** Place the shuttle and both players on a singles court, and a machine learning model trained on professional matches predicts the reply.

Live site: **https://shuttledestiny.com**

ShuttleDestiny is a machine learning model trained on 28,922 shots from 44 professional singles matches (2018–2021). On the website you drag the shuttle and both players around a singles court, choose the shot that was just played, and the model predicts the reply instantly. The model runs entirely in your browser, so the site needs no server.

## Results

Scored with 5-fold cross-validation **by match**: every match is held out in turn, and the model is only ever tested on matches it never saw during training.

| Model | Accuracy | Macro-F1 | Real reply in top 3 |
|---|---|---|---|
| Always guess the most common shot | 18.7% | – | – |
| v1: Random Forest, 3 inputs | 43.3% | 0.213 | 82.3% |
| **v2: gradient boosting, any player** | **44.8%** | **0.266** | **84.3%** |
| **v2: gradient boosting, receiver chosen** | **45.8%** | **0.282** | **85.6%** |

There are 15 possible replies, so exact-shot accuracy is a hard target. The bigger gain is macro-F1 (+32%): v2 predicts rarer shots like rushes, drives and wrist smashes much better, where v1 mostly guessed the common ones.

## What changed in v2

- **Real court positions.** ShuttleSet records positions as broadcast-video pixels, which differ for every camera setup. v2 uses each match's homography matrix to convert them to metres on a standard court. It also rotates the court so the hitter is always on the near side, which makes "back-left" mean the same thing at either end.
- **More context.** The landing spot, both players' positions, how far the receiver has to move, the two earlier shots, contact height, backhand, the score and the receiving player.
- **Honest features only.** Every input is known by the time the shot lands. `landing_height` was left out because it matches the *next* shot's contact height 97% of the time, so it would leak the receiver's choice. ShuttleSet records each rally's score *after* the rally, so the winner's point is taken back off to get the score at the start of the rally; otherwise the score would leak who wins.
- **Gradient-boosted trees** (scikit-learn `HistGradientBoostingClassifier`) instead of a Random Forest. The model is 1 MB instead of 32 MB.
- **Runs in the browser.** `train.py` exports the trees to `site/data/model.json`, and `site/js/model.js` evaluates them. It was checked against scikit-learn on 3,000 rows: identical top prediction on every row, probabilities within 0.000003.

## What the model relies on

Accuracy drop when each input is scrambled: the current shot type matters most (−21 pts), then the landing spot (−9 pts), the hitter's position, the receiving player, and the shot number and score.

## The court follows singles rules

Every match in ShuttleSet is singles, so the website's court does too:

- The side alleys are out. The shuttle can't be placed past the singles sidelines or the back boundary line.
- **Serves:** the server stands in the right service court when their score is even and the left when it's odd, and the serve must land in the diagonally opposite service court. Everything in front of the short service line is out on a serve. Serves are hit from below the net.
- **Rally order:** shot 1 is a serve, shot 2 replies to a serve, and no later shot can be a serve.

The data agrees: at an even score, 99.4% of serves come from the server's right, and 98% land in the diagonal box. That check also fixed v2's left/right orientation, so "right" in the model is the hitter's real right.

## Repository layout

```
features.py            Data loading, court-coordinate conversion, feature engineering, model class
train.py               Cross-validation, final training, export for the website
predict.py             Terminal predictor
shuttledestiny_model.joblib   Trained model (used by predict.py)
data/ShuttleSet/       ShuttleSet dataset (set/<match>/set*.csv, match.csv, homography.csv)
site/                  The website: upload this folder's contents to any web host
  index.html
  css/style.css
  js/model.js          Runs the exported model in the browser
  js/app.js            Court, controls, charts
  data/model.json      Exported model
  data/metrics.json    Evaluation results shown on the site
  data/analytics.json  Shot statistics for the "How the pros reply" section
```

## Run it yourself

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python train.py                    # about 1 minute: prints results, rewrites the model and site/data/*.json
python predict.py                  # try the model in the terminal
```

Preview the website locally (it needs a web server, because opening `index.html` straight from disk blocks loading the model file):

```bash
cd site
python -m http.server 8000         # then open http://localhost:8000
```

## Publishing the website

The site is plain HTML, CSS and JavaScript, so it works on any static host. For Hostinger, upload the **contents** of `site/` (so `index.html` is at the top level) into the subdomain's folder using hPanel → File Manager. After retraining, re-upload `site/data/` to update the live model.

## Data

ShuttleSet: Wang et al., *ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis*, KDD 2023. [Paper](https://arxiv.org/abs/2306.04948) · [Dataset](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)

## Limitations

- Trained on 44 elite singles matches with 27 players. It reflects how those pros play, not club badminton or doubles.
- Rarely played replies (driven flight, cross-court net shot, defensive lobs) are still hard to predict.
- Predictions are estimates of what tends to happen, not certainties.

Built by Surya Ilamurugu.
