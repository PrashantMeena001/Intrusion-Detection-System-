import os
import numpy as np
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_MAP = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}


def inject_recon_error(
    x_122: np.ndarray, content_err: float, other_err: float, behavioral_score: float
) -> np.ndarray:
    """
    Assemble the 125-feature vector, matching the exact order from
    customweights.ipynb: 122 raw + content_err + other_err + behavioral_score.
    (X_train_hybrid = [122, content, other] -> X_train_125 = [hybrid, behavioral])
    """
    return np.concatenate([x_122, [content_err], [other_err], [behavioral_score]])


def load_classifier(path: str = None) -> XGBClassifier:
    """Load the saved 125-feature cost-sensitive model. Never retrain here."""
    if path is None:
        path = os.path.join(
            PROJECT_ROOT, "models/custom_weights/xgboost_cost_final.json"
        )
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict(x_125: np.ndarray, model: XGBClassifier):
    """
    Predict on a single 125-feature record.
    Returns (predicted_class_name, class_probabilities_dict)
    """
    x = np.atleast_2d(x_125)
    pred_class = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    prob_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
    return LABEL_MAP[pred_class], prob_dict


if __name__ == "__main__":
    model = load_classifier()
    print(f"Loaded model, expects {model.n_features_in_} features (expect 125)")
