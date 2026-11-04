"""ShuttleSense interactive app — uses the EXISTING shuttle_model.joblib.

Run from VS Code's terminal with:
    python -m streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "shuttle_model.joblib"

st.set_page_config(
    page_title="ShuttleSense | Badminton Intelligence",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Matches the English shot labels in train.py and in the ShuttleSet dataset.
SHOT_TYPES = [
    "net shot", "return net", "smash", "wrist smash", "lob",
    "defensive return lob", "clear", "drive", "driven flight",
    "back-court drive", "drop", "passive drop", "push", "rush",
    "defensive return drive", "cross-court net shot",
    "short service", "long service",
]

# Reference: ShuttleSet paper, Figure 4, https://arxiv.org/pdf/2306.04948
# Schematic half-court: net is at the TOP, baseline at the BOTTOM.
# Outer zones 10-16 are OUTSIDE the singles lines.
COURT_ROWS = [
    [10, 2, 7, 1, 16],
    [11, 6, 8, 5, 15],
    [12, 4, 9, 3, 14],
]
ZONE_LABELS = {
    1: "Front right", 2: "Front left", 3: "Rear right", 4: "Rear left",
    5: "Middle right", 6: "Middle left", 7: "Front center",
    8: "Middle center", 9: "Rear center", 10: "Out: front left",
    11: "Out: middle left", 12: "Out: rear left",
    13: "Out: past baseline", 14: "Out: rear right",
    15: "Out: middle right", 16: "Out: front right",
}
INSIDE_ZONES = set(range(1, 10))

# The user's saved model retains two uncommon Chinese labels that the original
# training translation dictionary did not normalize. Do NOT change the model:
# translate these labels only at the interface boundary.
DISPLAY_SHOTS = {
    "過度切球": "passive drop",
    "未知球種": "unknown / unclassified shot",
}
MODEL_INPUT_SHOTS = {"passive drop": "過度切球"}


def shot_to_display(shot: str) -> str:
    return DISPLAY_SHOTS.get(str(shot), str(shot))



@st.cache_resource(show_spinner="Loading your trained model…")
def load_model(path: str):
    # joblib uses pickle; load only the model you trained or otherwise trust.
    return joblib.load(path)


def choose_zone(zone: int) -> None:
    st.session_state.zone = zone


def predict_next(model, previous_shot: str, zone: int, shot_number: int):
    example = pd.DataFrame([{
        "previous_shot": MODEL_INPUT_SHOTS.get(previous_shot, previous_shot),
        "landing_area": str(zone),  # Original training pipeline used STRINGS here.
        "ball_round": int(shot_number),
    }])
    probabilities = model.predict_proba(example)[0]
    return sorted(
        ((shot_to_display(str(shot)), float(prob)) for shot, prob in zip(model.classes_, probabilities)),
        key=lambda pair: pair[1],
        reverse=True,
    )


def get_style(selected_zone: int):
    common = """
<style>
:root { --mint: #61e3a0; --subtle: #93abac; }
.stApp { background: radial-gradient(circle at 10% 0%, #123737 0%, #0b181f 38%, #09121a 100%); color: #effffb; }
.block-container { max-width: 1350px; padding-top: 2.4rem; padding-bottom: 4rem; }
h1,h2,h3 { letter-spacing: -.035em; }
h1 { color: #effff9 !important; font-weight: 800 !important; }
[data-testid="stMetricValue"] { color: #a0f3bf !important; }
[data-testid="stCaptionContainer"] p { color: #a8bfc0; }
.kicker { font-size: 0.72rem; font-weight: 800; letter-spacing: .18em; color: #79e9b1; text-transform: uppercase; margin-bottom: .35rem; }
.hero { padding: .15rem 0 1.25rem; }
.hero-title { font-size: clamp(2.1rem,4.3vw,3.6rem); font-weight: 850; line-height: 1.1; margin: 0; letter-spacing: -.055em; color: #f2fff9; }
.hero-title span { color: #64e1a5; }
.hero-sub { max-width: 820px; color: #a8bebc; margin-top: .72rem; line-height: 1.65; font-size: 1.01rem; }
.section-heading { letter-spacing: -.025em; font-weight: 750; font-size: 1.24rem; margin: .1rem 0 .55rem; }
.small-label { font-size: .71rem; color: #aec9c5; text-transform: uppercase; font-weight: 750; letter-spacing: .11em; margin-top: .45rem; }
.net-line { height: 22px; border-bottom: 3px dashed #dbf3e8; text-align: center; color: #b5e9d3; font-size: .71rem; letter-spacing: .3em; font-weight: 750; margin-bottom: .5rem; text-shadow: 0 0 14px #75e1b5; }
.baseline { border-top: 3px solid rgba(226,255,240,.65); text-align: center; color: #98b4ac; font-size: .68rem; letter-spacing: .18em; font-weight: 700; padding-top: .28rem; margin-top: .05rem; }
.out-banner { text-align: center; font-size: .68rem; letter-spacing: .12em; color: #d9a894; padding: .4rem 0 0; }
.legend { font-size: .77rem; color: #a9c4bf; line-height: 1.7; }
.pill { display: inline-block; border-radius: 100px; padding: .19rem .68rem; font-size: .7rem; letter-spacing: .08em; font-weight: 750; background: rgba(99,237,173,.12); border: 1px solid rgba(99,237,173,.25); color: #9cfdc7; margin-right: .45rem; }
.foot { color: #91a6ab; font-size: .77rem; border-top: 1px solid #284149; padding-top: .95rem; margin-top: 1.3rem; }
/* Every zone is a REAL Streamlit button, not a non-clickable image. */
"""
    for zone in range(1, 17):
        # One class per zone, available from the button's key in Streamlit.
        bg = ("linear-gradient(145deg, #176c62, #125249)" if zone in INSIDE_ZONES
              else "linear-gradient(145deg, #443634, #332d31)")
        border = ("rgba(151,239,206,.30)" if zone in INSIDE_ZONES else "rgba(243,178,133,.28)")
        common += f"""
.st-key-zone_{zone} button {{
    min-height: 75px; width: 100%; background: {bg};
    border: 1px solid {border}; border-radius: 8px;
    font-weight: 800; font-size: 1.45rem; color: #edfff6;
    box-shadow: inset 0 0 10px rgba(5,18,20,.14); transition: transform .12s ease, border-color .12s ease;
}}
.st-key-zone_{zone} button:hover {{ border-color: #74e7aa; color: white; transform: translateY(-1px); }}
.st-key-zone_{zone} button:focus {{ color: white; }}
"""
    common += f"""
.st-key-zone_{selected_zone} button {{
    border: 3px solid #f7ce6b !important;
    background: linear-gradient(145deg, #388a66, #236a5a) !important;
    box-shadow: 0 0 0 2px rgba(247,206,107,.15), 0 0 18px rgba(247,206,107,.17);
    color: #fff9dd !important;
}}
</style>
"""
    return common


def render_court() -> None:
    st.markdown('<div class="section-heading">Select a landing zone</div>', unsafe_allow_html=True)
    st.markdown('<div class="net-line">NET</div>', unsafe_allow_html=True)
    for row in COURT_ROWS:
        cols = st.columns([.83, 1.2, 1.2, 1.2, .83], gap="small")
        for col, zone in zip(cols, row):
            with col:
                st.button(
                    str(zone), key=f"zone_{zone}", help=ZONE_LABELS[zone],
                    on_click=choose_zone, args=(zone,), use_container_width=True,
                )
    st.markdown('<div class="baseline">BASELINE</div>', unsafe_allow_html=True)
    row = st.columns([.83, 1.2, 1.2, 1.2, .83], gap="small")
    with row[2]:
        st.button("13", key="zone_13", help=ZONE_LABELS[13],
                  on_click=choose_zone, args=(13,), use_container_width=True)
    st.markdown('<div class="out-banner">OUTSIDE THE BACK LINE</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="legend"><span class="pill">1–9 · IN</span> '
        '<span class="pill" style="background:rgba(211,148,107,.13);border-color:#7b5950;color:#ebc0a2">10–16 · OUT</span>'
        ' View is from behind the baseline toward the net. '
        'The zone diagram is schematic, not to scale.</p>',
        unsafe_allow_html=True,
    )


def render_predictions(saved: dict) -> None:
    predictions = saved["results"]
    st.divider()
    st.markdown('### 🎯 Next-shot forecast')
    st.caption(
        f"After a **{saved['previous_shot']}** to zone **{saved['zone']}** "
        f"on shot **#{saved['shot_number']}** of the rally"
    )
    top_shot, top_probability = predictions[0]
    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric("Most likely next shot", top_shot.title())
    with c2:
        st.metric("Estimated likelihood", f"{top_probability:.1%}")
    st.markdown("#### Top 5 possibilities")
    for shot, probability in predictions[:5]:
        label, value = st.columns([4, 1])
        with label:
            st.write(shot.title())
            st.progress(min(max(probability, 0.0), 1.0))
        with value:
            st.write(f"**{probability:.1%}**")
    with st.expander("See probabilities for every shot"):
        table = pd.DataFrame(predictions, columns=["Shot", "Probability (%)"])
        table["Probability (%)"] *= 100
        st.dataframe(
            table, hide_index=True, use_container_width=True,
            column_config={"Probability (%)": st.column_config.ProgressColumn(
                "Probability (%)", min_value=0, max_value=100, format="%.1f%%",
            )},
        )
    st.info(
        "These are estimates of what players DID next in historical professional singles "
        "rallies—not what they SHOULD do. Predictions are uncertain and may be inaccurate."
    )


@st.cache_data(show_spinner="Reading match annotations…")
def dataset_summary():
    source_files = sorted((ROOT / "data").rglob("set[123].csv"))
    if not source_files:
        return None
    freq = {}
    n_strokes = 0
    n_rallies = 0
    matches = set()
    try:
        from train import shot_name
        for p in source_files:
            table = pd.read_csv(p, encoding="utf-8-sig", usecols=lambda c: c in {"rally", "type"})
            if "type" not in table.columns:
                continue
            n_strokes += len(table)
            n_rallies += table["rally"].nunique() if "rally" in table else 0
            matches.add(str(p.parent))
            for shot, amount in table["type"].dropna().map(shot_name).value_counts().items():
                label = shot_to_display(shot)
                freq[label] = freq.get(label, 0) + int(amount)
        return {"matches": len(matches), "rallies": n_rallies,
                "shots": n_strokes, "counts": freq}
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def show_dataset_tab(model):
    st.markdown("### Professional singles stroke dataset")
    stats = dataset_summary()
    if stats is None:
        st.warning("No match CSVs found. Predictions still work using your saved trained model.")
        return
    a, b, c = st.columns(3)
    a.metric("Matches", f"{stats['matches']:,}")
    b.metric("Rallies", f"{stats['rallies']:,}")
    c.metric("Recorded strokes", f"{stats['shots']:,}")
    st.markdown("#### Most frequently recorded shots")
    counts = (pd.Series(stats["counts"], name="Strokes")
              .sort_values(ascending=False).head(12))
    st.bar_chart(counts, horizontal=True)
    st.caption("Counts are based on the CSV files included in this project.")
    with st.expander("Check this saved model's held-out accuracy"):
        st.write("Your original training script split examples **by match**, so the test matches were not used to fit the model.")
        if st.button("Calculate test accuracy", key="evaluate_model"):
            with st.spinner("Evaluating your already-trained model…"):
                try:
                    from sklearn.metrics import accuracy_score
                    from sklearn.model_selection import GroupShuffleSplit
                    from train import load_rallies
                    data = load_rallies()
                    features = data[["previous_shot", "landing_area", "ball_round"]]
                    train_i, test_i = next(GroupShuffleSplit(
                        n_splits=1, test_size=0.2, random_state=42
                    ).split(features, data["next_shot"], groups=data["match_group"]))
                    predictions = model.predict(features.iloc[test_i])
                    accuracy = accuracy_score(data["next_shot"].iloc[test_i], predictions)
                    baseline_shot = data["next_shot"].iloc[train_i].mode().iloc[0]
                    baseline = (data["next_shot"].iloc[test_i] == baseline_shot).mean()
                    st.success(f"Held-out model accuracy: **{accuracy:.1%}**")
                    st.write(f"Most-common-shot baseline: **{baseline:.1%}**")
                    st.caption("For exact comparability, this repeats the split from train.py. It does not retrain the model.")
                except Exception as exc:
                    st.error(f"Unable to evaluate this model: {exc}")


def main():
    if "zone" not in st.session_state:
        st.session_state.zone = 8
    st.markdown(get_style(st.session_state.zone), unsafe_allow_html=True)
    st.markdown('''
<div class="hero">
  <div class="kicker">🏸 ShuttleSense / Badminton intelligence</div>
  <h1 class="hero-title">Read the rally. <span>Predict the return.</span></h1>
  <div class="hero-sub">Explore shot tendencies from professional badminton matches.
  Select a landing zone, describe the previous shot, and let your existing
  machine learning model estimate what comes next.</div>
</div>
''', unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error("Missing trained model: shuttle_model.joblib. Keep the original model file alongside app.py.")
        st.stop()
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Your saved model could not be loaded: {exc}")
        st.info("Make sure you installed requirements.txt in your project's Python environment.")
        st.stop()

    tab_predict, tab_data = st.tabs(["🎯  Predict next shot", "📊  Explore dataset"])
    with tab_predict:
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            render_court()
        with right:
            st.markdown('<div class="section-heading">Build your rally situation</div>', unsafe_allow_html=True)
            shot = st.selectbox(
                "Previous shot type", SHOT_TYPES,
                index=SHOT_TYPES.index("clear"),
                help="Choose the shot your opponent just played.",
            )
            shot_number = st.number_input(
                "Previous shot's position in the rally", min_value=1,
                max_value=100, value=5, step=1,
            )
            zone = st.session_state.zone
            st.markdown('<div class="small-label">Selected landing zone</div>', unsafe_allow_html=True)
            st.markdown(f"### {zone:02d} · {ZONE_LABELS[zone]}")
            if zone not in INSIDE_ZONES:
                st.warning(
                    "This is outside the singles court. An out shot normally ends the rally, "
                    "so the model cannot meaningfully predict another shot. Choose a zone from 1–9."
                )
            st.markdown(" ")
            pressed = st.button(
                "🏸 Predict next shot", type="primary", key="submit_prediction",
                use_container_width=True, disabled=zone not in INSIDE_ZONES,
            )
            if pressed:
                try:
                    predicted = predict_next(model, shot, zone, int(shot_number))
                    st.session_state.prediction = {
                        "previous_shot": shot, "zone": zone,
                        "shot_number": int(shot_number), "results": predicted,
                    }
                except Exception as exc:
                    st.error(f"Unable to make a prediction: {exc}")
            st.caption("Model: saved Random Forest · Inputs: shot type + landing zone + rally position")
        if "prediction" in st.session_state:
            render_predictions(st.session_state.prediction)
    with tab_data:
        show_dataset_tab(model)
    st.markdown(
        '<div class="foot">Built with Python, Streamlit & scikit-learn · '
        'Data: ShuttleSet (Wang et al., KDD 2023) · '
        'Court numbering: ShuttleSet Figure 4 · No model retraining required</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
