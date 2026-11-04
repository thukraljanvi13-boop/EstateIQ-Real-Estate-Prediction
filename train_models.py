
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "real_estate_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()

X = df.drop(columns=["Price_INR_Lakhs", "Monthly_Rent_INR"])
y = df["Price_INR_Lakhs"]


# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()


# Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_columns),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns)
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=22,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )
}


results = []
trained_models = {}


# Train and evaluate models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Regression metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Prediction accuracy within 10% tolerance
    actual_values = np.asarray(y_test)

    relative_error = np.abs(
        (actual_values - predictions) /
        np.maximum(np.abs(actual_values), 1e-10)
    )

    accuracy = np.mean(relative_error <= 0.10) * 100

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 4),
        "Prediction Accuracy (%)": round(accuracy, 2)
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("-" * 40)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 4))
    print("Prediction Accuracy:", round(accuracy, 2), "%")


# Create comparison table
results_df = pd.DataFrame(results)


# Select best model based on lowest RMSE
best_result = min(results, key=lambda x: x["RMSE"])
best_model_name = best_result["Model"]

best_model = trained_models[best_model_name]


# Find model with highest prediction accuracy
best_accuracy_result = max(
    results,
    key=lambda x: x["Prediction Accuracy (%)"]
)

best_accuracy_model = best_accuracy_result["Model"]
overall_accuracy = best_accuracy_result["Prediction Accuracy (%)"]


# Save best RMSE model
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)


# Save all trained models
joblib.dump(
    trained_models,
    MODEL_DIR / "all_models.pkl"
)


# Save comparison results
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)


# Display final results
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


print("\n" + "=" * 60)
print("BEST MODEL BASED ON RMSE")
print("=" * 60)

print("Best Model:", best_model_name)
print("RMSE:", best_result["RMSE"])


print("\n" + "=" * 60)
print("OVERALL PREDICTION ACCURACY")
print("=" * 60)

print("Best Accuracy Model:", best_accuracy_model)
print(f"Overall Prediction Accuracy: {overall_accuracy:.2f}%")


print("\nModels saved successfully!")
print("Comparison saved at:", OUTPUT_DIR / "model_comparison.csv")
