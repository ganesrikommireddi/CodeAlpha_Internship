"""
app.py
------
Streamlit web interface for the Real Estate Appraisal System.

Loads the trained model pipeline (models/best_model.pkl) and lets a user
enter property details to get an estimated market price.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Real Estate Appraisal System", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def main():
    st.title("🏠 Real Estate Appraisal System")
    st.write(
        "Estimate a property's market value using a machine learning model "
        "trained on historical housing data."
    )

    if not MODEL_PATH.exists():
        st.error(
            "No trained model found. Please run `python src/train.py` first "
            "to generate `models/best_model.pkl`."
        )
        return

    model = load_model()
    metadata = load_metadata()
    cat_options = metadata["categorical_options"]

    st.header("Property Details")

    col1, col2 = st.columns(2)
    with col1:
        square_footage = st.number_input("Square footage", 300, 10000, 2000, step=50)
        bedrooms = st.slider("Bedrooms", 1, 8, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 6.0, 2.0, step=0.5)
        lot_size = st.number_input("Lot size (sq ft)", 500, 50000, 8000, step=100)
        house_age = st.slider("House age (years)", 0, 120, 20)
        distance_to_city_center = st.slider("Distance to city center (miles)", 0.0, 60.0, 8.0)
        garage_spaces = st.selectbox("Garage spaces", [0, 1, 2, 3], index=2)

    with col2:
        school_rating = st.slider("School rating (1-10)", 1, 10, 7)
        crime_rate = st.slider("Crime rate (per 1,000 residents)", 0.0, 20.0, 3.0)
        has_pool = st.checkbox("Has pool")
        has_garden = st.checkbox("Has garden", value=True)
        renovated = st.checkbox("Recently renovated")
        neighborhood_quality = st.selectbox("Neighborhood quality", cat_options["neighborhood_quality"])
        condition = st.selectbox("Condition", cat_options["condition"])
        property_type = st.selectbox("Property type", cat_options["property_type"])

    if st.button("Estimate Price", type="primary"):
        rooms_total = bedrooms + bathrooms
        lot_to_house_ratio = lot_size / square_footage
        amenity_score = int(has_pool) + int(has_garden) + int(renovated) + int(garage_spaces > 0)
        is_new = int(house_age <= 5)

        input_df = pd.DataFrame([{
            "square_footage": square_footage,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "lot_size": lot_size,
            "house_age": house_age,
            "distance_to_city_center": distance_to_city_center,
            "garage_spaces": garage_spaces,
            "has_pool": int(has_pool),
            "has_garden": int(has_garden),
            "school_rating": school_rating,
            "crime_rate": crime_rate,
            "renovated": int(renovated),
            "rooms_total": rooms_total,
            "lot_to_house_ratio": lot_to_house_ratio,
            "amenity_score": amenity_score,
            "is_new": is_new,
            "neighborhood_quality": neighborhood_quality,
            "condition": condition,
            "property_type": property_type,
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated Price: ${prediction:,.0f}")
        st.caption(
            f"Model used: {metadata['best_model']} "
            f"(test R² = {metadata['results'][metadata['best_model']]['R2']:.3f})"
        )

    st.divider()
    with st.expander("About this model"):
        st.write(f"**Best performing model:** {metadata['best_model']}")
        results_df = pd.DataFrame(metadata["results"]).T
        st.dataframe(results_df.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}"}))


if __name__ == "__main__":
    main()
