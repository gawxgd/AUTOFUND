import pandas as pd
import cloudpickle

# === Load both meta models locally ===
with open("models/meta_models/weighted_meta_model/weighted_meta_model.pkl", "rb") as f:
    weighted_model = cloudpickle.load(f)
    print("weighted_model_loaded")

with open("models/meta_models/lgbm_meta_model/lgbm_meta_model.pkl", "rb") as f:
    lgbm_model = cloudpickle.load(f)
    print("lgbm_model_loaded")

# === Embed both inside the predict function ===
def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
    preds_weighted = weighted_model(live_features)
    preds_lgbm = lgbm_model(live_features)
    print("Predicted")
    # Average predictions (omega model)
    omega_preds = (preds_weighted + preds_lgbm) / 2

    return omega_preds.to_frame("prediction")

# === Save the predict function ===
if __name__ == "__main__":
    with open("predictv2.pkl", "wb") as f:
        cloudpickle.dump(predict, f)
    print("✅ Saved predict.pkl with both weighted and lgbm models embedded.")
