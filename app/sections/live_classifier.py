import json
import os

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_PATH = os.path.join(APP_DIR, "assets", "sample_records.json")


FEATURE_COLS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


@st.cache_data
def load_samples():
    with open(SAMPLES_PATH, "r") as f:
        return json.load(f)


def csv_row_to_raw_record(row: pd.Series) -> dict:
    """Same numeric/categorical cast rule as generate_sample_records.py."""
    record = {}
    for c in FEATURE_COLS:
        if c in CATEGORICAL_COLS:
            record[c] = str(row[c])
        else:
            record[c] = float(row[c])
    return record


def render_result_panel(result: dict, true_class: str = None, sample_index: int = None):
    """Shared output panel for both the sample path and the CSV path.

    sample_index is the record's absolute position in
    app/assets/sample_records.json — set only when the prediction came
    from the sample picker (None for CSV uploads). Explainability
    (Section 4) uses it to look up precomputed SHAP values, which are
    keyed by that same index.
    """
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction")
        st.metric("Predicted class", result["predicted_class"])

        if true_class is not None:
            if true_class == result["predicted_class"]:
                st.success(f"True class: {true_class} — match")
            else:
                st.error(f"True class: {true_class} — mismatch")

        probs = result["probabilities"]
        prob_df = pd.DataFrame(
            {"probability": [probs[c] for c in CLASS_ORDER]}, index=CLASS_ORDER
        )
        st.bar_chart(prob_df)

    with col2:
        st.subheader("Autoencoder signal")
        st.metric(
            "Full reconstruction error",
            f"{result['full_recon_error']:.6f}",
            delta=f"tau = {result['tau']:.6f}",
            delta_color="off",
        )
        if result["flagged_by_autoencoder"]:
            st.warning("Flagged as anomalous by the autoencoder (error > tau)")
        else:
            st.success("Not flagged by the autoencoder (error <= tau)")

        st.caption(
            f"content_error = {result['content_error']:.6f}  |  "
            f"other_error = {result['other_error']:.6f}  |  "
            f"behavioral_score = {result['behavioral_score']:.4f}"
        )

    # Handoff for Section 4 (Explainability) — reuse this exact
    # prediction's x_125 vector instead of recomputing anything.
    st.session_state["last_prediction"] = result
    st.session_state["last_true_class"] = true_class
    st.session_state["last_sample_index"] = sample_index
    # A new prediction invalidates any live-SHAP result computed for
    # the previous record — otherwise Explainability could show stale
    # values for a record no longer selected.
    st.session_state.pop("live_shap_cache", None)


def render_sample_picker(pipeline, samples):
    st.subheader("Sample records")

    class_filter = st.selectbox("Filter by true class", ["All"] + CLASS_ORDER)
    # Keep each sample's absolute index into the full `samples` list (not
    # just its position after filtering) — Explainability (Section 4)
    # looks up precomputed SHAP values by that same absolute index into
    # sample_records.json, so it has to survive the class filter here.
    filtered = [
        (abs_idx, s)
        for abs_idx, s in enumerate(samples)
        if class_filter == "All" or s["true_class"] == class_filter
    ]

    labels = [
        f"{s['true_class']} — {s['true_attack_type']} (#{abs_idx})"
        for abs_idx, s in filtered
    ]
    if not labels:
        st.info("No samples for this class.")
        return

    choice = st.selectbox("Pick a record", labels)
    choice_pos = labels.index(choice)
    sample_idx, sample = filtered[choice_pos]

    if st.button("Classify this record", type="primary"):
        with st.spinner("Running inference..."):
            result = pipeline.predict(sample["raw_features"])
        render_result_panel(
            result, true_class=sample["true_class"], sample_index=sample_idx
        )


def render_csv_uploader(pipeline):
    st.subheader("Custom prediction (CSV upload)")
    st.caption(
        "Upload a CSV with the 41 raw NSL-KDD feature columns. "
        "An optional `attack_type` column enables true-label comparison."
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return

    has_labels = "attack_type" in df.columns
    if has_labels:
        from label_map import get_true_class  # noqa: E402 (src/ on sys.path via main.py)

    if len(df) == 1:
        row = df.iloc[0]
        raw_record = csv_row_to_raw_record(row)
        true_class = None
        if has_labels:
            try:
                true_class = get_true_class(str(row["attack_type"]).strip())
            except KeyError:
                st.warning(
                    f"attack_type '{row['attack_type']}' not in ATTACK_MAP — "
                    f"showing prediction without true-label comparison."
                )

        if st.button("Classify uploaded record", type="primary"):
            with st.spinner("Running inference..."):
                result = pipeline.predict(raw_record)
            render_result_panel(result, true_class=true_class)
        return

    # Batch path: predict every row, show a summary table.
    if st.button(f"Classify all {len(df)} rows", type="primary"):
        rows_out = []
        with st.spinner(f"Running inference on {len(df)} rows..."):
            for _, row in df.iterrows():
                raw_record = csv_row_to_raw_record(row)
                result = pipeline.predict(raw_record)

                true_class = None
                if has_labels:
                    try:
                        true_class = get_true_class(str(row["attack_type"]).strip())
                    except KeyError:
                        true_class = "unmapped"

                rows_out.append(
                    {
                        "predicted_class": result["predicted_class"],
                        "true_class": true_class if true_class is not None else "-",
                        "flagged_by_autoencoder": result["flagged_by_autoencoder"],
                        "full_recon_error": round(result["full_recon_error"], 6),
                    }
                )

        st.dataframe(pd.DataFrame(rows_out), use_container_width=True)
        st.caption(
            "Batch view is summary-only. Upload a single-row CSV for the "
            "full probability/recon-error panel on one record."
        )


def render(pipeline):
    samples = load_samples()

    tab_sample, tab_csv = st.tabs(["Use a sample record", "Upload CSV"])
    with tab_sample:
        render_sample_picker(pipeline, samples)
    with tab_csv:
        render_csv_uploader(pipeline)
