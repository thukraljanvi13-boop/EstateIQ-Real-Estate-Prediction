
import pandas as pd
import joblib

from pathlib import Path
from investment_analyzer import analyze_investment

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load trained machine learning model
model = joblib.load(MODEL_PATH)


def predict_property_investment(
    property_details,
    monthly_rent,
    annual_expenses,
    investment_cost=None
):
    # Convert property details into DataFrame
    property_data = pd.DataFrame([property_details])

    # Predict property price
    predicted_price = float(model.predict(property_data)[0])

    # Analyze investment opportunity
    investment_result = analyze_investment(
        property_price=predicted_price,
        monthly_rent=monthly_rent,
        annual_expenses=annual_expenses,
        investment_cost=investment_cost
    )

    # Combine prediction and investment results
    result = {
        "Predicted Property Price (INR Lakhs)": round(predicted_price, 2),
        **investment_result
    }

    return result


if __name__ == "__main__":

    sample_property = {
        "City": "Chandigarh",
        "Property_Type": "Apartment",
        "Locality_Type": "Developed",
        "BHK": 3,
        "Area_sqft": 1500,
        "Bathrooms": 3,
        "Property_Age_Years": 5,
        "Floor_Number": 2,
        "Total_Floors": 10,
        "Metro_Distance_km": 2.0,
        "School_Distance_km": 1.5,
        "Hospital_Distance_km": 2.0,
        "Parking": 1,
        "Lift": 1,
        "Security": 1,
        "Furnishing": "Semi-Furnished"
    }

    result = predict_property_investment(
        property_details=sample_property,
        monthly_rent=25000,
        annual_expenses=50000
    )

    print("\n========== PROPERTY INVESTMENT REPORT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

    # Save report
    pd.DataFrame([result]).to_csv(
        OUTPUT_DIR / "complete_investment_report.csv",
        index=False
    )

    print("\nComplete investment report saved successfully!")
