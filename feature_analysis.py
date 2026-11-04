
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "real_estate_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

importance = regressor.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))

top_features = importance_df.head(15).sort_values("Importance")

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.title("Top 15 Features Affecting Property Prices")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "feature_importance.png")
plt.close()

print("\nFeature analysis completed!")
