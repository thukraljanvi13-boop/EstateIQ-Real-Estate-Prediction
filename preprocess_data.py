
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "real_estate_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates()

# Separate features and target
X = df.drop(columns=["Price_INR_Lakhs", "Monthly_Rent_INR"])
y = df["Price_INR_Lakhs"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# Split before fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save processed datasets
pd.DataFrame(
    X_train_processed.toarray() if hasattr(X_train_processed, "toarray")
    else X_train_processed
).to_csv(OUTPUT_DIR / "X_train.csv", index=False)

pd.DataFrame(
    X_test_processed.toarray() if hasattr(X_test_processed, "toarray")
    else X_test_processed
).to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Processed feature count:", X_train_processed.shape[1])
print("Files saved in outputs folder.")
