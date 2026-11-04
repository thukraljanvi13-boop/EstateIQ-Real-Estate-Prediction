
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "real_estate_data.csv"

np.random.seed(42)
N = 12000

city_prices = {
    "Chandigarh": 7500,
    "Delhi": 10000,
    "Mumbai": 15000,
    "Bangalore": 8500,
    "Pune": 6500,
    "Hyderabad": 6500,
    "Chennai": 6200,
    "Ludhiana": 4500,
    "Jaipur": 4200,
    "Ahmedabad": 4800
}

locality_factors = {
    "Prime": 1.50,
    "Developed": 1.15,
    "Suburban": 0.90,
    "Emerging": 0.75
}

property_factors = {
    "Apartment": 1.00,
    "Independent House": 1.20,
    "Villa": 1.55
}

furnishing_factors = {
    "Unfurnished": 1.00,
    "Semi-Furnished": 1.04,
    "Fully-Furnished": 1.09
}

city = np.random.choice(list(city_prices), N)

property_type = np.random.choice(
    list(property_factors),
    N,
    p=[0.65, 0.25, 0.10]
)

locality_type = np.random.choice(
    list(locality_factors),
    N,
    p=[0.15, 0.35, 0.30, 0.20]
)

bhk = np.random.choice(
    [1, 2, 3, 4, 5],
    N,
    p=[0.10, 0.35, 0.32, 0.17, 0.06]
)

area_sqft = np.clip(
    bhk * np.random.normal(520, 95, N),
    400,
    3500
).round(0)

bathrooms = np.clip(
    bhk + np.random.choice([-1, 0, 1], N, p=[0.15, 0.70, 0.15]),
    1,
    6
).astype(int)

property_age = np.random.randint(0, 31, N)
total_floors = np.random.randint(1, 31, N)

floor_number = np.array([
    np.random.randint(0, floors)
    for floors in total_floors
])

metro_distance = np.round(
    np.random.exponential(2.5, N) + 0.2, 2
)

school_distance = np.round(
    np.random.uniform(0.2, 8, N), 2
)

hospital_distance = np.round(
    np.random.uniform(0.3, 10, N), 2
)

parking = np.random.choice([0, 1], N, p=[0.25, 0.75])
lift = np.random.choice([0, 1], N, p=[0.25, 0.75])
security = np.random.choice([0, 1], N, p=[0.35, 0.65])

furnishing = np.random.choice(
    list(furnishing_factors),
    N,
    p=[0.35, 0.40, 0.25]
)

base_price = pd.Series(city).map(city_prices).to_numpy()

locality_factor = (
    pd.Series(locality_type)
    .map(locality_factors)
    .to_numpy()
)

property_factor = (
    pd.Series(property_type)
    .map(property_factors)
    .to_numpy()
)

furnishing_factor = (
    pd.Series(furnishing)
    .map(furnishing_factors)
    .to_numpy()
)

amenity_factor = (
    1 + parking * 0.025 + lift * 0.025 + security * 0.020
)

location_factor = (
    1
    + 0.10 * np.exp(-metro_distance / 3)
    + 0.025 * np.exp(-school_distance / 2)
    + 0.020 * np.exp(-hospital_distance / 3)
)

age_factor = np.maximum(0.78, 1 - property_age * 0.008)

noise = np.random.normal(1, 0.05, N)

price = (
    area_sqft
    * base_price
    * locality_factor
    * property_factor
    * furnishing_factor
    * amenity_factor
    * location_factor
    * age_factor
    * noise
    / 100000
)

price = np.maximum(price, 10).round(2)

rental_yield = np.clip(
    np.random.normal(0.045, 0.008, N),
    0.025,
    0.075
)

monthly_rent = (
    price * 100000 * rental_yield / 12
    * np.random.normal(1, 0.06, N)
)

monthly_rent = np.maximum(monthly_rent, 3000).round(0)

df = pd.DataFrame({
    "City": city,
    "Property_Type": property_type,
    "Locality_Type": locality_type,
    "BHK": bhk,
    "Area_sqft": area_sqft,
    "Bathrooms": bathrooms,
    "Property_Age_Years": property_age,
    "Floor_Number": floor_number,
    "Total_Floors": total_floors,
    "Metro_Distance_km": metro_distance,
    "School_Distance_km": school_distance,
    "Hospital_Distance_km": hospital_distance,
    "Parking": parking,
    "Lift": lift,
    "Security": security,
    "Furnishing": furnishing,
    "Price_INR_Lakhs": price,
    "Monthly_Rent_INR": monthly_rent
})

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 55)
print("REAL ESTATE DATASET GENERATED SUCCESSFULLY")
print("=" * 55)

print(f"\nTotal properties: {len(df):,}")
print(f"Total features: {len(df.columns)}")
print(f"Saved at: {OUTPUT_PATH}")

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nPrice statistics:")
print(df["Price_INR_Lakhs"].describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset generation completed!")
