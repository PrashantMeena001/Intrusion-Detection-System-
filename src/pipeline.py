import preprocess
import autoencoder
import classifier


class NIDSPipeline:
    """
    Loads all fitted artifacts (encoder, scaler, autoencoder, taxonomy,
    Gaussian params, XGBoost model) ONCE, then exposes predict() for
    repeated use — this is what makes it fast enough for a live Streamlit
    demo instead of reloading a Keras model on every click.
    """

    def __init__(self):
        self.artifacts = preprocess.load_artifacts()
        self.taxonomy = preprocess.load_feature_taxonomy()
        self.non_content_idx = autoencoder.get_non_content_idx(
            self.taxonomy["traffic_idx"], self.taxonomy["basic_idx"]
        )
        self.gauss_params = preprocess.load_behavioral_gaussian_params()
        self.ae = autoencoder.load_autoencoder()
        self.tau = autoencoder.load_threshold()
        self.model = classifier.load_classifier()

    def predict(self, raw_record: dict) -> dict:
        """
        Full pipeline: raw record -> 122-feature transform -> recon errors
        + behavioral score -> 125-feature vector -> XGBoost prediction.

        Returns a dict with everything the Streamlit app's 4 sections need —
        not just the predicted class, but the intermediate signals too
        (Section 2 needs full_recon_error + tau, Section 4/SHAP will need
        the 125-feature vector itself).
        """
        x_122 = preprocess.transform_raw_record(raw_record, self.artifacts)

        content_err, other_err = autoencoder.get_recon_error(
            x_122, self.ae, self.taxonomy["content_idx"], self.non_content_idx
        )
        full_recon_error = autoencoder.get_full_recon_error(x_122, self.ae)
        behavioral_score = preprocess.get_behavioral_score(
            raw_record, self.gauss_params["mu"], self.gauss_params["var"]
        )

        x_125 = classifier.inject_recon_error(
            x_122, content_err, other_err, behavioral_score
        )
        pred_class, probs = classifier.predict(x_125, self.model)

        return {
            "predicted_class": pred_class,
            "probabilities": probs,
            "content_error": content_err,
            "other_error": other_err,
            "behavioral_score": behavioral_score,
            "full_recon_error": full_recon_error,
            "tau": self.tau,
            "flagged_by_autoencoder": bool(full_recon_error > self.tau),
            "x_125": x_125,  # kept for later SHAP use in Section 4
        }


_pipeline_instance = None


def get_pipeline() -> NIDSPipeline:
    """Lazy singleton — loads everything on first call, reuses after."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = NIDSPipeline()
    return _pipeline_instance


def predict(raw_record: dict) -> dict:
    """The single function app/sections/live_classifier.py should call."""
    return get_pipeline().predict(raw_record)


if __name__ == "__main__":
    import time

    t0 = time.time()
    pipe = get_pipeline()
    print(f"Pipeline loaded in {time.time() - t0:.2f}s")
