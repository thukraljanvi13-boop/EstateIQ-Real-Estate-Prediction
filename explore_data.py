
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "real_estate_data.csv"
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

plt.figure(figsize=(10, 6))
sns.histplot(df["Price_INR_Lakhs"], kde=True)
plt.title("Distribution of Property Prices")
plt.xlabel("Price (INR Lakhs)")
plt.savefig(output_dir / "price_distribution.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="City", y="Price_INR_Lakhs")
plt.xticks(rotation=45)
plt.title("Property Prices Across Cities")
plt.tight_layout()
plt.savefig(output_dir / "city_price_comparison.png")
plt.close()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Area_sqft",
    y="Price_INR_Lakhs",
    alpha=0.5
)
plt.title("Property Area vs Price")
plt.savefig(output_dir / "area_vs_price.png")
plt.close()

print("\nEDA completed successfully!")
print("Charts saved in the outputs folder.")
