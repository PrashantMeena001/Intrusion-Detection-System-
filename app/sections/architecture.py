import streamlit as st


def render(pipe=None):
    # `pipe` is accepted for signature consistency with the other
    # sections (main.py calls PAGES[page_name].render(pipe)), but
    # deliberately unused — this page is a static explainer and makes
    # no pipeline calls.
    st.caption(
        "A hybrid intrusion detection pipeline: an autoencoder's "
        "reconstruction-error signals feed into a cost-sensitive "
        "XGBoost multi-class classifier."
    )

    st.markdown(
        """
        > An autoencoder trained on normal traffic contributes two
        > reconstruction-error signals — content-feature error and
        > non-content/traffic-feature error — plus a diagonal-Gaussian
        > behavioral anomaly score, all engineered as extra features
        > for a cost-sensitive XGBoost multi-class attack classifier.
        """
    )

    st.divider()

    # ------------------------------------------------------------------
    # DIAGRAM
    # ------------------------------------------------------------------
    st.subheader("Data Flow")

    diagram = r"""
    digraph NIDS {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fontname="Helvetica",
              fontsize=11, color="#4B5563", fillcolor="#F3F4F6",
              fontcolor="#111827", margin="0.18,0.12"];
        edge [color="#9CA3AF", fontname="Helvetica", fontsize=9,
              fontcolor="#4B5563"];

        Raw [label="Raw Record\n(41 features)"];
        Prep [label="Preprocessing\nOneHotEncode + MinMaxScale"];

        subgraph cluster_ae {
            label="Autoencoder path\n(trained on Normal only)";
            style="rounded,dashed";
            color="#9CA3AF";
            fontsize=10;
            fontcolor="#4B5563";
            AE [label="Autoencoder\n(122-dim reconstruction)"];
            ContentErr [label="Content-feature\nreconstruction error", fillcolor="#DBEAFE"];
            TrafficErr [label="Non-content/traffic\nreconstruction error", fillcolor="#DBEAFE"];
        }

        subgraph cluster_gauss {
            label="Behavioral path";
            style="rounded,dashed";
            color="#9CA3AF";
            fontsize=10;
            fontcolor="#4B5563";
            Gauss [label="Diagonal-Gaussian\nover 5 behavioral columns", fillcolor="#DCFCE7"];
            GaussScore [label="Behavioral\nanomaly score", fillcolor="#DCFCE7"];
        }

        XGB [label="XGBoost\n(125 features, cost-sensitive)\nR2L/U2R upweighted",
             fillcolor="#FEF3C7"];
        Pred [label="Prediction\nNormal / DoS / Probe / R2L / U2R",
              shape=ellipse, fillcolor="#FEE2E2"];

        Raw -> Prep;
        Prep -> AE;
        Prep -> Gauss;
        Prep -> XGB [label="122 raw features", style=dashed];
        AE -> ContentErr;
        AE -> TrafficErr;
        Gauss -> GaussScore;
        ContentErr -> XGB [label="+1 feature"];
        TrafficErr -> XGB [label="+1 feature"];
        GaussScore -> XGB [label="+1 feature"];
        XGB -> Pred;
    }
    """
    st.graphviz_chart(diagram, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # WHY TWO MODELS
    # ------------------------------------------------------------------
    st.subheader("Why a hybrid design")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Autoencoder → catches volume-based attacks (DoS)")
        st.markdown(
            """
            - Trained **only** on Normal traffic.
            - Volume attacks look nothing like normal traffic in raw
              feature space, so the autoencoder reconstructs them
              poorly → high reconstruction error → flagged.
            - Reconstruction error is split into two signals instead
              of one blended number: **content-feature error** and
              **non-content/traffic-feature error**, using NSL-KDD's
              documented 3-way column taxonomy (basic / content /
              traffic).
            """
        )

    with col2:
        st.markdown("#### XGBoost → catches behavioral attacks (R2L, U2R)")
        st.markdown(
            """
            - R2L and U2R attacks look similar to normal traffic in
              *volume* but differ in subtle *behavior* — the
              autoencoder often won't flag them.
            - XGBoost has seen labeled examples of these patterns
              during training, and carries **extra class weight** on
              R2L/U2R to counter severe class imbalance.
            - Helped further by a **diagonal-covariance Gaussian**
              anomaly score over 5 behavioral columns
              (`num_failed_logins`, `num_root`, `num_shells`,
              `is_guest_login`, `num_access_files`) — R2L/U2R sit far
              above other classes in the tail of this score even when
              their medians look identical.
            """
        )

    st.info(
        "Neither model is sufficient alone. The autoencoder's error "
        "signals and the behavioral anomaly score are engineered as "
        "**extra features** for a single 125-feature XGBoost "
        "classifier — that XGBoost model is the actual hybrid, not a "
        "separate voting ensemble.",
        icon="💡",
    )

    st.divider()

    # ------------------------------------------------------------------
    # KEY DESIGN DECISIONS
    # ------------------------------------------------------------------
    st.subheader("Key design decisions")

    with st.expander("Why OneHotEncoder, not LabelEncoder, for categoricals"):
        st.markdown(
            """
            `protocol_type`, `service`, and `flag` are categorical.
            LabelEncoder would assign arbitrary integers (e.g.
            `http=14`, `ftp=3`), implying a false ordinal relationship
            that the autoencoder would treat as numerically close or
            far. OneHotEncoder gives each category its own binary
            column, which the autoencoder can reconstruct meaningfully
            as 0s and 1s.
            """
        )

    with st.expander("Why diagonal covariance for the Gaussian, not full"):
        st.markdown(
            """
            Several of the 5 behavioral columns are near-binary or
            near-zero for most non-U2R traffic. A full covariance
            matrix over these columns risks near-singularity, so a
            diagonal-covariance Gaussian (independent per-feature
            variance) is used instead. Per-row anomaly score is the
            sum of squared z-scores across the 5 columns.
            """
        )

    with st.expander("Why macro F1, not accuracy"):
        st.markdown(
            """
            NSL-KDD is severely imbalanced (Normal ≈ 67k, DoS ≈ 46k,
            Probe ≈ 11.6k, R2L ≈ 995, U2R ≈ 52 in training). A model
            that predicts Normal every time already scores ~53.5%
            accuracy — accuracy is misleading here. Macro F1 weighs
            every class equally regardless of size, so the model is
            actually rewarded for catching rare R2L/U2R attacks.
            """
        )

    with st.expander("Why multi-class, not binary (attack vs. normal)"):
        st.markdown(
            """
            Collapsing to binary would throw away exactly the
            distinction that matters operationally — a DoS flood and
            a U2R privilege-escalation attempt require completely
            different responses. The 5-class label (Normal, DoS,
            Probe, R2L, U2R) is preserved end to end.
            """
        )

    st.divider()

    st.subheader("Real-world relevance")
    st.markdown(
        """
        This mirrors the architecture used by companies like
        **Darktrace** and **Cisco** for production network monitoring:
        an unsupervised anomaly signal feeding a supervised classifier
        as engineered features. NSL-KDD is from 1999 and modern
        traffic looks different — that caveat is worth raising
        proactively, since it signals awareness of the dataset's
        limitations rather than treating it as production-ready.
        """
    )
