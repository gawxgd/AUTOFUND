import pandas as pd
import cloudpickle
import os

# Define your project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_predictor(model_type: str):
    """Loads the meta model from disk."""
    if model_type == "weighted":
        path = os.path.join(PROJECT_ROOT, "models/meta_models/weighted_meta_model/weighted_meta_model.pkl")
    elif model_type == "lgbm":
        path = os.path.join(PROJECT_ROOT, "models/meta_models/lgbm_meta_model/lgbm_meta_model.pkl")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    with open(path, "rb") as f:
        return cloudpickle.load(f)

def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
    """Numerai-compatible predict function."""
    prediction_function = load_predictor("weighted")  # or "lgbm"
    predictions = prediction_function(live_features)
    return predictions.to_frame("prediction")

# === Serialize the predict function ===
if __name__ == "__main__":
    with open("predict.pkl", "wb") as f:
        cloudpickle.dump(predict, f)
    print("✅ Saved predict.pkl")
