import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SHAP_PATH = os.path.join(APP_DIR, "assets", "shap_values.json")

ENGINEERED_NAMES = [
    "recon_error_content",
    "recon_error_other",
    "behavioral_gaussian_score",
]


@st.cache_data
def load_precomputed_shap():
    if not os.path.exists(SHAP_PATH):
        return None
    with open(SHAP_PATH, "r") as f:
        return json.load(f)


def compute_live_shap(pipeline, x_125, predicted_class):
    """Opt-in only — called exclusively from the 'Generate Live SHAP'
    button click, never on page load."""
    import preprocess  # src/ on sys.path via main.py
    from classifier import LABEL_MAP  # noqa: E402

    feature_names = preprocess.get_feature_names(pipeline.artifacts) + ENGINEERED_NAMES
    x_df = pd.DataFrame([x_125], columns=feature_names)

    explainer = shap.TreeExplainer(pipeline.model)
    shap_expl = explainer(x_df)

    class_idx = {name: idx for idx, name in LABEL_MAP.items()}[predicted_class]
    values = shap_expl.values[0, :, class_idx]

    top_order = np.argsort(np.abs(values))[::-1][:10]
    return [
        {
            "feature": feature_names[j],
            "shap_value": float(values[j]),
            "feature_value": float(x_125[j]),
        }
        for j in top_order
    ]


def render_shap_bar_chart(top_features, predicted_class):
    ordered = sorted(top_features, key=lambda d: abs(d["shap_value"]))
    names = [f["feature"] for f in ordered]
    values = [f["shap_value"] for f in ordered]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"SHAP value (impact on '{predicted_class}' prediction)")
    ax.set_title("Top contributing features")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Green = pushed toward the predicted class. Red = pushed away from it.")


def render_summary(result: dict, true_class: str, top_features: list):
    attack_class = result["predicted_class"]
    confidence = result["probabilities"][attack_class]
    flagged = result["flagged_by_autoencoder"]
    recon_error = result["full_recon_error"]
    tau = result["tau"]
    top_names = ", ".join(f["feature"] for f in top_features[:3])

    summary = (
        f"This record was classified as {attack_class} with "
        f"{confidence * 100:.0f}% confidence. The autoencoder "
        f"{'flagged' if flagged else 'did not flag'} it "
        f"(reconstruction error: {recon_error:.3f}, threshold: {tau:.3f}). "
        f"Top contributing features: {top_names}."
    )
    st.info(summary)

    if true_class is not None:
        match = "matches" if true_class == attack_class else "does not match"
        st.caption(f"True class: {true_class} ({match} the prediction)")


def render(pipeline):
    st.subheader("Explainability")
    st.caption(
        "SHAP (SHapley Additive exPlanations): how much each feature "
        "pushed this prediction toward or away from the predicted class."
    )

    result = st.session_state.get("last_prediction")
    true_class = st.session_state.get("last_true_class")
    sample_index = st.session_state.get("last_sample_index")

    if result is None:
        st.info(
            "No prediction yet — classify a record on the Live Classifier "
            "page first, then come back here to see why."
        )
        return

    precomputed = load_precomputed_shap()
    top_features = st.session_state.get("live_shap_cache")

    if top_features is None and sample_index is not None and precomputed is not None:
        entry = next(
            (e for e in precomputed if e["sample_index"] == sample_index), None
        )
        if entry is not None and entry["predicted_class"] == result["predicted_class"]:
            top_features = entry["top_features"]
            st.caption("Precomputed SHAP values — zero live compute.")

    if top_features is None:
        st.info(
            "No precomputed SHAP for this record "
            + (
                "(only the 75 sample records have precomputed values)."
                if sample_index is None
                else "for its current prediction."
            )
            + " Generate it live instead."
        )

    if st.button(
        "Generate Live SHAP", type="primary" if top_features is None else "secondary"
    ):
        with st.spinner("Running shap.TreeExplainer on this record..."):
            top_features = compute_live_shap(
                pipeline, result["x_125"], result["predicted_class"]
            )
        st.session_state["live_shap_cache"] = top_features
        st.caption("Live SHAP — computed just now, not precomputed.")

    if top_features is None:
        return

    render_shap_bar_chart(top_features, result["predicted_class"])
    render_summary(result, true_class, top_features)
