import os
import numpy as np
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_autoencoder(path: str = None):
    """Load the frozen, trained autoencoder. Never retrain here."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "models/autoencoder.keras")
    return load_model(path)


def load_threshold(path: str = None) -> float:
    """Load tau — the reconstruction-error threshold, a single float."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "models/threshold.txt")
    with open(path, "r") as f:
        return float(f.read())


def get_non_content_idx(traffic_idx: np.ndarray, basic_idx: np.ndarray) -> np.ndarray:
    """Matches autoencoder.ipynb Cell 12: non_content_idx = concatenate([traffic, basic])."""
    return np.concatenate([traffic_idx, basic_idx])


def get_recon_error(
    x_122: np.ndarray, autoencoder, content_idx: np.ndarray, non_content_idx: np.ndarray
):
    """
    Compute content-only and non-content reconstruction error, matching
    autoencoder.ipynb Cell 12 exactly. Accepts either a single record
    (shape (122,)) or a batch (shape (n, 122)).

    Returns
    -------
    (content_error, other_error) — floats if input was a single record,
    else 1D arrays of shape (n,)
    """
    was_single = x_122.ndim == 1
    x = np.atleast_2d(x_122)

    # Direct call instead of .predict(): same forward pass, same
    # weights, same numeric output — but .predict() builds a full
    # tf.data pipeline (multiprocessing/threading) even for one tiny
    # array, which is unnecessary overhead here and can outright
    # deadlock in some terminal environments (observed hanging
    # indefinitely in the VS Code integrated terminal on macOS).
    reconstruction = autoencoder(x, training=False).numpy()
    squared_diff = np.square(x - reconstruction)

    content_error = np.mean(squared_diff[:, content_idx], axis=1)
    other_error = np.mean(squared_diff[:, non_content_idx], axis=1)

    if was_single:
        return float(content_error[0]), float(other_error[0])
    return content_error, other_error


def get_full_recon_error(x_122: np.ndarray, autoencoder):
    """
    Blended MSE over all 122 columns — this is what tau/threshold.txt
    was tuned against, and what the Anomaly Detector app section
    visualizes. NOT one of the 3 engineered features fed to XGBoost
    (that uses the content/other split instead).
    """
    was_single = x_122.ndim == 1
    x = np.atleast_2d(x_122)

    # See get_recon_error() above for why this is a direct call
    # instead of .predict() — identical math, avoids the tf.data
    # pipeline overhead/deadlock risk for small single-batch input.
    reconstruction = autoencoder(x, training=False).numpy()
    error = np.mean(np.square(x - reconstruction), axis=1)

    return float(error[0]) if was_single else error


if __name__ == "__main__":
    import preprocess

    ae = load_autoencoder()
    tau = load_threshold()
    print(f"Loaded autoencoder, tau = {tau:.6f}")

    taxonomy = preprocess.load_feature_taxonomy()
    non_content_idx = get_non_content_idx(
        taxonomy["traffic_idx"], taxonomy["basic_idx"]
    )
    print(
        f"content_idx: {len(taxonomy['content_idx'])}, non_content_idx: {len(non_content_idx)}"
    )
