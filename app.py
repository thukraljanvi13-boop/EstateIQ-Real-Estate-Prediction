
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

from prediction_engine import predict_property_investment

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "real_estate_data.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="EstateIQ | Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.metric-card {
    background-color: #172033;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263449;
}
h1, h2, h3 {
    color: #38BDF8;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Sidebar
st.sidebar.title("🏙️ EstateIQ")
st.sidebar.caption("Real Estate Investment Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Property Prediction",
        "Investment Analyzer",
        "Market Insights",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Based Decision Support")

# Dashboard
if page == "Dashboard":

    st.title("EstateIQ Dashboard")
    st.write("Real Estate Investment Intelligence Platform")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Properties", f"{len(df):,}")
    col2.metric("Cities Covered", df["City"].nunique())
    col3.metric("Average Price", f"₹{df['Price_INR_Lakhs'].mean():.2f} L")
    col4.metric("Average Rent", f"₹{df['Monthly_Rent_INR'].mean():,.0f}")

    st.markdown("## Property Market Overview")

    city_prices = (
        df.groupby("City")["Price_INR_Lakhs"]
        .mean()
        .reset_index()
        .sort_values("Price_INR_Lakhs", ascending=False)
    )

    fig = px.bar(
        city_prices,
        x="City",
        y="Price_INR_Lakhs",
        title="Average Property Prices by City",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Recent Property Records")
    st.dataframe(df.head(10), use_container_width=True)

# Property Prediction
elif page == "Property Prediction":

    st.title("Property Price Prediction")
    st.write("Enter property details to estimate its market price.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            city = st.selectbox("City", sorted(df["City"].unique()))
            property_type = st.selectbox(
                "Property Type",
                ["Apartment", "Independent House", "Villa"]
            )
            locality = st.selectbox(
                "Locality Type",
                ["Prime", "Developed", "Suburban", "Emerging"]
            )
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
            area = st.number_input(
                "Area (sq. ft.)", 400, 3500, 1500
            )
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6])
            age = st.slider("Property Age (Years)", 0, 30, 5)
            furnishing = st.selectbox(
                "Furnishing",
                ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
            )

        with col2:
            floor = st.number_input("Floor Number", 0, 30, 2)
            total_floors = st.number_input("Total Floors", 1, 30, 10)
            metro = st.number_input("Metro Distance (km)", 0.0, 30.0, 2.0)
            school = st.number_input("School Distance (km)", 0.0, 30.0, 1.5)
            hospital = st.number_input("Hospital Distance (km)", 0.0, 30.0, 2.0)
            parking = st.selectbox("Parking", [1, 0], format_func=lambda x: "Yes" if x else "No")
            lift = st.selectbox("Lift", [1, 0], format_func=lambda x: "Yes" if x else "No")
            security = st.selectbox("Security", [1, 0], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button(
            "Predict Property Price",
            use_container_width=True
        )

    if submitted:

        if floor >= total_floors:
            st.error("Floor number must be less than total floors.")
        else:
            property_details = {
                "City": city,
                "Property_Type": property_type,
                "Locality_Type": locality,
                "BHK": bhk,
                "Area_sqft": area,
                "Bathrooms": bathrooms,
                "Property_Age_Years": age,
                "Floor_Number": floor,
                "Total_Floors": total_floors,
                "Metro_Distance_km": metro,
                "School_Distance_km": school,
                "Hospital_Distance_km": hospital,
                "Parking": parking,
                "Lift": lift,
                "Security": security,
                "Furnishing": furnishing
            }

            predicted_price = float(model.predict(
                pd.DataFrame([property_details])
            )[0])

            st.success("Property prediction completed!")

            st.metric(
                "Estimated Property Price",
                f"₹{predicted_price:.2f} Lakhs"
            )

            st.info(
                "This is a model-based estimate using synthetic training data."
            )

# Investment Analyzer
elif page == "Investment Analyzer":

    st.title("Investment Analyzer")
    st.write("Evaluate the rental investment potential of a property.")

    price = st.number_input(
        "Property Price (INR Lakhs)", min_value=1.0, value=80.0
    )

    rent = st.number_input(
        "Monthly Rent (INR)", min_value=0, value=30000
    )

    expenses = st.number_input(
        "Annual Expenses (INR)", min_value=0, value=60000
    )

    if st.button("Analyze Investment", use_container_width=True):

        from investment_analyzer import analyze_investment

        result = analyze_investment(
            property_price=price,
            monthly_rent=rent,
            annual_expenses=expenses
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Gross Rental Yield", f"{result['Gross Rental Yield (%)']}%")
        col2.metric("Net ROI", f"{result['Net ROI (%)']}%")
        col3.metric("Net Annual Income", f"₹{result['Net Annual Income (INR)']:,.0f}")

        st.subheader("Investment Recommendation")
        st.success(result["Recommendation"])

        st.dataframe(
            pd.DataFrame([result]),
            use_container_width=True
        )

        pd.DataFrame([result]).to_csv(
            OUTPUT_DIR / "investment_analysis.csv",
            index=False
        )

# Market Insights
elif page == "Market Insights":

    st.title("Real Estate Market Insights")

    city = st.selectbox(
        "Select City",
        sorted(df["City"].unique())
    )

    city_df = df[df["City"] == city]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Price",
        f"₹{city_df['Price_INR_Lakhs'].mean():.2f} L"
    )

    col2.metric(
        "Average Rent",
        f"₹{city_df['Monthly_Rent_INR'].mean():,.0f}"
    )

    col3.metric(
        "Properties",
        f"{len(city_df):,}"
    )

    fig = px.box(
        city_df,
        x="Property_Type",
        y="Price_INR_Lakhs",
        color="Property_Type",
        template="plotly_dark",
        title=f"Property Price Distribution in {city}"
    )

    st.plotly_chart(fig, use_container_width=True)

    bhk_prices = (
        city_df.groupby("BHK")["Price_INR_Lakhs"]
        .mean()
        .reset_index()
    )

    fig2 = px.line(
        bhk_prices,
        x="BHK",
        y="Price_INR_Lakhs",
        markers=True,
        template="plotly_dark",
        title="Average Price by BHK"
    )

    st.plotly_chart(fig2, use_container_width=True)

# Model Performance
elif page == "Model Performance":

    st.title("Machine Learning Model Performance")

    comparison_path = OUTPUT_DIR / "model_comparison.csv"

    if comparison_path.exists():

        results = pd.read_csv(comparison_path)

        st.dataframe(results, use_container_width=True)

        fig = px.bar(
            results,
            x="Model",
            y="R2 Score",
            color="Model",
            title="Model R² Score Comparison",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        importance_path = OUTPUT_DIR / "feature_importance.csv"

        if importance_path.exists():

            importance = pd.read_csv(importance_path).head(15)

            fig2 = px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Important Features",
                template="plotly_dark"
            )

            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Train the models first to view performance.")