import os

import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "models/threshold.txt")

RECON_ERROR_PATH = os.path.join(PROCESSED_DIR, "recon_error_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")


@st.cache_data
def load_recon_arrays():
    recon_error = np.load(RECON_ERROR_PATH)
    y_test = np.load(Y_TEST_PATH)
    y_binary = (y_test != 0).astype(int)  # 0 = Normal, 1 = Anomaly
    return recon_error, y_binary


@st.cache_data
def load_shipped_tau():
    """Default the slider to the tuned tau, falling back to the 97th
    percentile of recon_error if threshold.txt isn't found (e.g. app
    run standalone without the full models/ dir)."""
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            return float(f.read().strip())
    recon_error, _ = load_recon_arrays()
    return float(np.percentile(recon_error, 97))


def render(pipeline):
    if not (os.path.exists(RECON_ERROR_PATH) and os.path.exists(Y_TEST_PATH)):
        st.error(
            "Missing data/processed/recon_error_test.npy or y_test.npy — "
            "run notebook 03 (autoencoder) first."
        )
        return

    recon_error, y_binary = load_recon_arrays()
    shipped_tau = load_shipped_tau()

    st.subheader("Autoencoder threshold explorer")
    st.caption(
        "Binary view (Normal vs Anomaly) driven only by the autoencoder's "
        "full reconstruction error against a threshold — this is the "
        "autoencoder in isolation, not the final 5-class hybrid prediction."
    )

    slider_max = float(np.percentile(recon_error, 99.5))
    threshold = st.slider(
        "Threshold (tau)",
        min_value=0.0,
        max_value=round(slider_max, 6),
        value=round(min(shipped_tau, slider_max), 6),
        step=slider_max / 500,
        format="%.6f",
    )
    if abs(threshold - shipped_tau) < 1e-9:
        st.caption(f"Currently at the shipped tau ({shipped_tau:.6f}).")

    predicted = (recon_error > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_binary, predicted, labels=[0, 1]).ravel()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion matrix")
        st.table(
            {
                "Predicted Normal": [f"TN = {tn:,}", f"FN = {fn:,}"],
                "Predicted Anomaly": [f"FP = {fp:,}", f"TP = {tp:,}"],
            }
        )
        st.caption("Rows: True Normal (top), True Anomaly (bottom)")

    with col2:
        st.subheader("Metrics")
        precision = precision_score(y_binary, predicted, zero_division=0)
        recall = recall_score(y_binary, predicted, zero_division=0)
        f1 = f1_score(y_binary, predicted, zero_division=0)
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1", f"{f1:.4f}")

    st.caption(
        f"{len(recon_error):,} test records | "
        f"{int(y_binary.sum()):,} true anomalies | "
        f"{int((y_binary == 0).sum()):,} true normal"
    )
