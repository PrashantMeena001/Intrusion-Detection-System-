import os
import sys

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pipeline as pipeline_module  # noqa: E402

from sections import (  # noqa: E402
    anomaly_detector,
    architecture,
    data_explorer,
    explainability,
    live_classifier,
)

st.set_page_config(page_title="Hybrid NIDS", layout="wide")

# Order matches the context prompt's section numbering, with
# Architecture appended 5th per SECTION 5d.
PAGES = {
    "Data Explorer": data_explorer,
    "Anomaly Detector": anomaly_detector,
    "Live Classifier": live_classifier,
    "Explainability": explainability,
    "Architecture": architecture,
}


@st.cache_resource
def load_pipeline():
    # This print is the test for Step 2: it should appear in the
    # terminal exactly once per app process, not once per click/rerun.
    print("[main.py] Loading NIDSPipeline (should log once per process)...")
    pipe = pipeline_module.get_pipeline()

    # Warm-up: a model's FIRST .predict() call pays a one-time graph
    # tracing cost (can be 10-30s on CPU) totally separate from normal
    # inference speed. Paying that cost here — during app startup,
    # under the spinner Streamlit already shows for a cached resource —
    # means the user's first real Classify click is fast, not the
    # first thing that looks "stuck." Dummy zeros are fine: this only
    # exercises the autoencoder/XGBoost graphs, never touches
    # transform_raw_record() or real feature values.
    import numpy as np

    dummy_122 = np.zeros((1, 122), dtype=float)
    pipe.ae(
        dummy_122, training=False
    )  # direct call, not .predict() — see autoencoder.py
    dummy_125 = np.zeros((1, 125), dtype=float)
    pipe.model.predict(dummy_125)  # XGBoost — unaffected, not a Keras model

    return pipe


def main():
    st.sidebar.title("Hybrid NIDS")
    page_name = st.sidebar.radio("Navigate", list(PAGES.keys()))

    pipe = load_pipeline()

    st.title(page_name)
    PAGES[page_name].render(pipe)


if __name__ == "__main__":
    main()
