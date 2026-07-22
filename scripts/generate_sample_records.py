import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from label_map import get_true_class

# --- config -----------------------------------------------------------
N_PER_CLASS = 15
RANDOM_SEED = 42
# ------------------------------------------------------------------------
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
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def main():
    raw_path = os.path.join(PROJECT_ROOT, "data/raw/KDDTest+.txt")
    df = pd.read_csv(raw_path, header=None, names=ALL_COLS)

    df["attack_type"] = df["attack_type"].astype(str).str.strip().str.rstrip(".")

    def _safe_true_class(attack_type):
        try:
            return get_true_class(attack_type)
        except KeyError:
            return None

    df["true_class"] = df["attack_type"].apply(_safe_true_class)

    unmapped = df[df["true_class"].isna()]
    if not unmapped.empty:
        counts = unmapped["attack_type"].value_counts()
        print("Skipping attack_types not in ATTACK_MAP (never seen in training):")
        for atk, cnt in counts.items():
            print(f"  {atk:20s}: {cnt} rows in KDDTest+.txt, excluded from demo")
        df = df[df["true_class"].notna()]

    samples = []
    for cls in CLASS_ORDER:
        subset = df[df["true_class"] == cls]
        if subset.empty:
            raise ValueError(
                f"No rows found for true_class='{cls}' in KDDTest+.txt — "
                f"check label_map.py / the raw file."
            )

        n = min(N_PER_CLASS, len(subset))
        picked = subset.sample(n=n, random_state=RANDOM_SEED)

        for _, row in picked.iterrows():
            raw_features = {}
            for c in FEATURE_COLS:
                if c in CATEGORICAL_COLS:
                    raw_features[c] = str(row[c])
                else:
                    # int-like NSL-KDD columns and float rate columns
                    # both round-trip safely through float; the pipeline
                    # just needs numeric, not a specific dtype.
                    raw_features[c] = float(row[c])

            samples.append(
                {
                    "true_class": cls,
                    "true_attack_type": row["attack_type"],
                    "raw_features": raw_features,
                }
            )

        print(f"  {cls:8s}: picked {n:2d} / {len(subset):5d} available")

    out_path = os.path.join(PROJECT_ROOT, "app/assets/sample_records.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nWrote {len(samples)} total sample records to {out_path}")


if __name__ == "__main__":
    main()
