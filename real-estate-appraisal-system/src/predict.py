"""
predict.py
----------
Small command-line utility to load the saved model and predict the price
of a single property, useful for quick testing without launching Streamlit.

Example:
    python src/predict.py
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"


def predict_price(property_dict: dict) -> float:
    """Load the saved pipeline and return a predicted price for one property.

    property_dict must contain all the feature keys the model was trained on
    (see README.md "Usage" section for an example).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/best_model.pkl not found. Run `python src/train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([property_dict])
    return float(model.predict(df)[0])


if __name__ == "__main__":
    example_property = {
        "square_footage": 2200,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "lot_size": 9000,
        "house_age": 10,
        "distance_to_city_center": 6.5,
        "garage_spaces": 2,
        "has_pool": 0,
        "has_garden": 1,
        "school_rating": 8,
        "crime_rate": 2.5,
        "renovated": 1,
        "rooms_total": 6.5,
        "lot_to_house_ratio": 9000 / 2200,
        "amenity_score": 3,
        "is_new": 0,
        "neighborhood_quality": "High",
        "condition": "Good",
        "property_type": "Single Family",
    }

    price = predict_price(example_property)
    print(f"Predicted price for example property: ${price:,.0f}")
