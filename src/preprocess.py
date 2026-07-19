import joblib
import numpy as np
import pandas as pd
import os

# The three categorical columns one-hot encoded in preprocessing.ipynb
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# The five raw columns the behavioral Gaussian anomaly score is fit on
BEHAVIORAL_COLS = [
    "num_failed_logins",
    "num_root",
    "num_shells",
    "is_guest_login",
    "num_access_files",
]

# NOTE: encoder_artifacts.pkl was saved with joblib.dump(), not pickle.dump().
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_artifacts(path: str = None) -> dict:
    """Load the fitted OneHotEncoder, fitted MinMaxScaler, and num_cols order."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data/processed/encoder_artifacts.pkl")
    artifacts = joblib.load(path)
    for key in ("ohe", "scaler", "num_cols"):
        if key not in artifacts:
            raise KeyError(
                f"encoder_artifacts.pkl missing key '{key}'. Found: {list(artifacts.keys())}"
            )
    return artifacts


def load_feature_taxonomy(processed_dir: str = None) -> dict:
    """Load the 3-way NSL-KDD feature taxonomy index arrays from preprocessing.ipynb."""
    if processed_dir is None:
        processed_dir = os.path.join(PROJECT_ROOT, "data/processed")
    return {
        "content_idx": np.load(f"{processed_dir}/content_idx.npy"),
        "traffic_idx": np.load(f"{processed_dir}/traffic_idx.npy"),
        "basic_idx": np.load(f"{processed_dir}/basic_idx.npy"),
    }


def get_feature_names(artifacts: dict) -> list:
    """Full ordered list of 122 post-OHE feature names: num_cols + OHE names."""
    num_cols = artifacts["num_cols"]
    ohe_names = list(artifacts["ohe"].get_feature_names_out(CATEGORICAL_COLS))
    return list(num_cols) + ohe_names


def transform_raw_record(raw_record: dict, artifacts: dict) -> np.ndarray:
    """
    Transform a single raw NSL-KDD record into the 122-feature, [0,1]-scaled
    vector the autoencoder/XGBoost were trained on. transform() only —
    ohe/scaler are already fit and must never be re-fit here.
    """
    num_cols = artifacts["num_cols"]
    ohe = artifacts["ohe"]
    scaler = artifacts["scaler"]

    missing_num = [c for c in num_cols if c not in raw_record]
    missing_cat = [c for c in CATEGORICAL_COLS if c not in raw_record]
    if missing_num or missing_cat:
        raise ValueError(
            f"Missing numeric: {missing_num}, missing categorical: {missing_cat}"
        )

    num_values = np.array([[raw_record[c] for c in num_cols]], dtype=float)

    cat_df = pd.DataFrame([{c: raw_record[c] for c in CATEGORICAL_COLS}])
    cat_values = ohe.transform(cat_df)  # dense already — sparse_output=False in the fit

    x_122 = np.hstack([num_values, cat_values])
    x_122_scaled = scaler.transform(x_122)

    return x_122_scaled.flatten()


def load_behavioral_gaussian_params(path: str = None) -> dict:
    """Load the {'mu', 'var'} fit on Normal-only raw behavioral columns."""
    if path is None:
        path = os.path.join(
            PROJECT_ROOT, "data/processed/behavioral_gaussian_params.pkl"
        )
    return joblib.load(path)


def get_behavioral_score(raw_record: dict, mu: np.ndarray, var: np.ndarray) -> float:
    """
    Diagonal-Gaussian anomaly score on RAW (unscaled) behavioral column
    values — matches preprocessing.ipynb exactly, which fits mu/var on
    df_train[behavioral_cols], not on the MinMax-scaled matrix.
    """
    x = np.array([raw_record[c] for c in BEHAVIORAL_COLS], dtype=float)
    z = (x - mu) / np.sqrt(var)
    return float(np.sum(z**2))


if __name__ == "__main__":
    artifacts = load_artifacts()
    names = get_feature_names(artifacts)
    print(f"Total feature count: {len(names)} (expect 122)")
    taxonomy = load_feature_taxonomy()
    for k, v in taxonomy.items():
        print(f"{k}: {len(v)} indices")
