import json
import os
import sys

import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pipeline
import preprocess
from classifier import LABEL_MAP

SAMPLES_PATH = os.path.join(PROJECT_ROOT, "app/assets/sample_records.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "app/assets/shap_values.json")
TOP_N = 10

ENGINEERED_NAMES = [
    "recon_error_content",
    "recon_error_other",
    "behavioral_gaussian_score",
]

NAME_TO_CLASS_IDX = {name: idx for idx, name in LABEL_MAP.items()}


def main():
    with open(SAMPLES_PATH, "r") as f:
        samples = json.load(f)

    pipe = pipeline.get_pipeline()
    feature_names = preprocess.get_feature_names(pipe.artifacts) + ENGINEERED_NAMES

    explainer = shap.TreeExplainer(pipe.model)

    all_explanations = []
    for i, sample in enumerate(samples):
        result = pipe.predict(sample["raw_features"])
        class_idx = NAME_TO_CLASS_IDX[result["predicted_class"]]

        x_df = pd.DataFrame([result["x_125"]], columns=feature_names)
        shap_expl = explainer(x_df)  # shape (1, 125, 5)

        values = shap_expl.values[0, :, class_idx]
        feature_values = shap_expl.data[0, :]
        base_value = float(np.array(shap_expl.base_values[0]).reshape(-1)[class_idx])

        top_order = np.argsort(np.abs(values))[::-1][:TOP_N]
        top_features = [
            {
                "feature": feature_names[j],
                "shap_value": float(values[j]),
                "feature_value": float(feature_values[j]),
            }
            for j in top_order
        ]

        all_explanations.append(
            {
                "sample_index": i,
                "predicted_class": result["predicted_class"],
                "base_value": base_value,
                "top_features": top_features,
            }
        )

        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(samples)} records explained")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_explanations, f, indent=2)

    print(f"\nWrote {len(all_explanations)} SHAP explanations to {OUT_PATH}")


if __name__ == "__main__":
    main()
