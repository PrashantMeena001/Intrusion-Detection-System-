import os

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/raw/KDDTrain+.txt")

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
ALL_COLS = FEATURE_COLS + ["attack_type", "difficulty_level"]
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]

OVERVIEW_PNGS = [
    ("class_distribution.png", "Class distribution"),
    ("Protocol_type_by_class.png", "Protocol type by class"),
    ("Topservices_by_class.png", "Top services by class"),
    ("Guest_login_by_class_percentage.png", "Guest login % by class"),
]
FEATURE_PNGS = [
    ("Duration_by_flag_box.png", "Duration by flag"),
    ("bytes_box_plot.png", "Byte counts (src/dst)"),
    ("feature_variance_ranking.png", "Feature variance ranking"),
]


@st.cache_data
def load_train_df():
    from label_map import get_true_class  # src/ on sys.path via main.py

    df = pd.read_csv(TRAIN_PATH, header=None, names=ALL_COLS)
    df["attack_type"] = df["attack_type"].astype(str).str.strip().str.rstrip(".")

    def _safe(a):
        try:
            return get_true_class(a)
        except KeyError:
            return None

    df["true_class"] = df["attack_type"].apply(_safe)
    n_unmapped = df["true_class"].isna().sum()
    df = df[df["true_class"].notna()].copy()
    return df, n_unmapped


def render_png_grid(pairs, cols=2):
    columns = st.columns(cols)
    for i, (filename, caption) in enumerate(pairs):
        path = os.path.join(REPORTS_DIR, filename)
        with columns[i % cols]:
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True)
            else:
                st.warning(f"Missing: reports/{filename}")


def render_interactive(df: pd.DataFrame):
    st.subheader("Interactive: top services by class + protocol")

    col1, col2 = st.columns(2)
    with col1:
        class_filter = st.selectbox("Class", ["All"] + CLASS_ORDER, key="de_class")
    with col2:
        protocols = ["All"] + sorted(df["protocol_type"].unique().tolist())
        protocol_filter = st.selectbox("Protocol type", protocols, key="de_protocol")

    filtered = df
    if class_filter != "All":
        filtered = filtered[filtered["true_class"] == class_filter]
    if protocol_filter != "All":
        filtered = filtered[filtered["protocol_type"] == protocol_filter]

    st.caption(f"{len(filtered):,} records match this filter (of {len(df):,} total)")

    if filtered.empty:
        st.info("No records match this combination.")
        return

    top_services = filtered["service"].value_counts().head(10)
    st.bar_chart(top_services)


def render(pipeline):
    tab_overview, tab_features, tab_interactive = st.tabs(
        ["Overview", "Feature Distributions", "Interactive Explorer"]
    )

    with tab_overview:
        render_png_grid(OVERVIEW_PNGS)

    with tab_features:
        render_png_grid(FEATURE_PNGS)

    with tab_interactive:
        df, n_unmapped = load_train_df()
        if n_unmapped > 0:
            st.caption(
                f"Note: {n_unmapped} training rows had attack_types not in "
                f"label_map's ATTACK_MAP and were excluded (same rows the "
                f"model itself never trained on)."
            )
        render_interactive(df)
