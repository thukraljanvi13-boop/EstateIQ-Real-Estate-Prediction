
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_investment(
    property_price,
    monthly_rent,
    annual_expenses,
    investment_cost=None
):
    # Validate inputs
    if property_price <= 0:
        raise ValueError("Property price must be greater than zero.")

    if monthly_rent < 0:
        raise ValueError("Monthly rent cannot be negative.")

    if annual_expenses < 0:
        raise ValueError("Annual expenses cannot be negative.")

    if investment_cost is None:
        investment_cost = property_price

    if investment_cost <= 0:
        raise ValueError("Investment cost must be greater than zero.")

    # Convert property price from lakhs to INR
    property_price_inr = property_price * 100000
    investment_cost_inr = investment_cost * 100000

    # Calculate rental income
    annual_rent = monthly_rent * 12

    # Calculate investment metrics
    gross_yield = (annual_rent / property_price_inr) * 100

    net_income = annual_rent - annual_expenses

    net_roi = (net_income / investment_cost_inr) * 100

    # Investment recommendation
    if net_roi >= 6:
        recommendation = "High Investment Potential"
    elif net_roi >= 3:
        recommendation = "Moderate Investment Potential"
    else:
        recommendation = "Low Investment Potential"

    # Prepare results
    result = {
        "Property Price (INR Lakhs)": round(property_price, 2),
        "Monthly Rent (INR)": round(monthly_rent, 2),
        "Annual Rental Income (INR)": round(annual_rent, 2),
        "Annual Expenses (INR)": round(annual_expenses, 2),
        "Net Annual Income (INR)": round(net_income, 2),
        "Gross Rental Yield (%)": round(gross_yield, 2),
        "Net ROI (%)": round(net_roi, 2),
        "Recommendation": recommendation
    }

    return result


if __name__ == "__main__":

    result = analyze_investment(
        property_price=80,
        monthly_rent=30000,
        annual_expenses=60000
    )

    print("\n========== INVESTMENT ANALYSIS ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

    # Save analysis report
    report = pd.DataFrame([result])

    report.to_csv(
        OUTPUT_DIR / "investment_analysis.csv",
        index=False
    )

    print("\nInvestment analysis completed!")
    print("Report saved in outputs/investment_analysis.csv")
